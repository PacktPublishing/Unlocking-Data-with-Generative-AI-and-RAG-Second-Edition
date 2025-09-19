# generic_procedural_memory.py
import json
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from domain_agent import DomainAgent, DomainProcedure

class ProceduralMemory:
    """Generic procedural memory that works with any domain agent"""
    
    def __init__(self, llm, domain_agent: DomainAgent):
        self.llm = llm
        self.domain_agent = domain_agent
        
        # Get domain-specific configuration
        self.procedure_class = domain_agent.get_procedure_class()
        self.community_definitions = domain_agent.get_community_definitions()
        
        # SEGMENTED STORAGE
        self.user_procedures = {}
        self.community_procedures = {}
        self.global_procedures = {}
        self.task_procedures = {}
        
        # Keep old flat storage for backward compatibility
        self.procedures = {}
        
        # Track learning at each level
        self.user_learning_history = defaultdict(list)
        self.community_learning_history = defaultdict(list)
        self.global_learning_history = []
        self.learning_history = []  # backward compatibility
        
        # Track community membership
        self.user_communities = {}
        self.community_members = defaultdict(set)
        
        # Track discovered segments
        self.segments_discovered = set()
        
        # Create learning prompts from domain agent
        self._create_learning_prompts()
        
        self.json_parser = JsonOutputParser()
    
    def _create_learning_prompts(self):
        """Create prompts from domain agent"""
        prompts = self.domain_agent.get_learning_prompts()
        
        self.global_learning_prompt = PromptTemplate.from_template(prompts["global"])
        self.user_learning_prompt = PromptTemplate.from_template(prompts["user"])
        self.community_learning_prompt = PromptTemplate.from_template(prompts["community"])
        self.task_learning_prompt = PromptTemplate.from_template(prompts["task"])
        
        # Keep backward compatibility
        self.learning_prompt = self.global_learning_prompt
    
    def _identify_task_type(self, query):
        """Delegate to domain agent"""
        return self.domain_agent.identify_task_type(query)
    
    def learn_from_interaction(self, query: str, interaction_data: Dict,
                              user_id: Optional[str] = None,
                              user_profile: Optional[Dict] = None) -> Dict:
        """Learn at multiple levels based on interaction success."""
        learned_items = {}
        

        def try_learn(prompt, params, scope, scope_id=None):
            try:
                chain = prompt | self.llm | self.json_parser
                result = chain.invoke(params)
                if result.get("procedure"):
                    proc_data = result["procedure"]
                    
                    # Clean segments if they're dicts instead of strings
                    segments = proc_data.get("segments", [])
                    if segments and isinstance(segments[0], dict):
                        # Extract string from dict objects
                        segments = [
                            str(s.get("segment_title", s.get("description", s.get("name", str(s)))))
                            if isinstance(s, dict) else str(s)
                            for s in segments
                        ]
                        proc_data["segments"] = segments
                    
                    # Clean domain_metrics to ensure only numeric values
                    domain_metrics = proc_data.get("domain_metrics", {})
                    clean_metrics = {}
                    for k, v in domain_metrics.items():
                        try:
                            # Try to convert to float
                            if isinstance(v, (int, float)):
                                clean_metrics[k] = float(v)
                            elif isinstance(v, str) and v.replace('.','').replace('-','').isdigit():
                                clean_metrics[k] = float(v)
                            # Skip non-numeric values
                        except:
                            pass
                    proc_data["domain_metrics"] = clean_metrics
                    
                    return proc_data, result.get("confidence", 0.85), clean_metrics
            except Exception as e:
                print(f"Learning failed in {scope}: {e}")
                pass
            return None, None, None

        # 1. Task learning
        task_type = self._identify_task_type(query)
        if task_type != "general" and task_type not in self.task_procedures:
            self.task_procedures[task_type] = {}
        
        if task_type != "general" and len(self.task_procedures[task_type]) < 2:
            proc_data, conf, metrics = try_learn(
                self.task_learning_prompt,
                {"task_type": task_type, "task": query, 
                 "execution_data": json.dumps(interaction_data)},
                "task", task_type
            )
            if proc_data:
                pattern = f"{task_type}_strategy_{len(self.task_procedures[task_type])}"
                self.task_procedures[task_type][pattern] = self.procedure_class(
                    strategy_pattern=pattern,
                    steps=proc_data.get("steps", ["Execute task"]),
                    segments=proc_data.get("segments", []),
                    success_rate=conf,
                    scope="task",
                    scope_id=task_type,
                    domain_metrics=metrics
                )
                learned_items["task_learned"] = pattern
        
        # 2. User learning
        if user_id and user_id not in self.user_procedures:
            self.user_procedures[user_id] = {}
        
        if user_id and len(self.user_procedures[user_id]) == 0:
            proc_data, conf, metrics = try_learn(
                self.user_learning_prompt,
                {"user_id": user_id, "task": query,
                 "execution_data": json.dumps(interaction_data)},
                "user", user_id
            )
            if proc_data:
                pattern = f"user_{user_id}_preference_0"
                self.user_procedures[user_id][pattern] = self.procedure_class(
                    strategy_pattern=pattern,
                    steps=proc_data.get("steps", [f"Personalized for {user_id}"]),
                    segments=proc_data.get("segments", []),
                    success_rate=conf,
                    scope="user",
                    scope_id=user_id,
                    domain_metrics=metrics
                )
                learned_items["user_learned"] = pattern
        
        # 3. Community assignment and learning
        if user_id and user_id not in self.user_communities:
            community = self.domain_agent.identify_community(user_id, user_profile)
            self.user_communities[user_id] = [community]
            self.community_members[community].add(user_id)
        
        # Learn for community
        if user_id:
            for community_id in self.user_communities[user_id]:
                if community_id not in self.community_procedures:
                    self.community_procedures[community_id] = {}
                
                if len(self.community_procedures[community_id]) == 0:
                    proc_data, conf, metrics = try_learn(
                        self.community_learning_prompt,
                        {"community_segment": community_id, "task": query,
                         "execution_data": json.dumps(interaction_data)},
                        "community", community_id
                    )
                    if proc_data:
                        pattern = f"{community_id}_pattern_0"
                        self.community_procedures[community_id][pattern] = self.procedure_class(
                            strategy_pattern=pattern,
                            steps=proc_data.get("steps", [f"Community approach"]),
                            segments=proc_data.get("segments", []),
                            success_rate=conf,
                            scope="community",
                            scope_id=community_id,
                            domain_metrics=metrics
                        )
                        learned_items["community_learned"] = pattern
        
        # 4. Global learning
        if len(self.global_procedures) < 5:
            proc_data, conf, metrics = try_learn(
                self.global_learning_prompt,
                {"task": query, "execution_data": json.dumps(interaction_data),
                 "existing": json.dumps(list(self.global_procedures.keys()))},
                "global"
            )
            if proc_data:
                pattern = proc_data.get("strategy_pattern", f"global_strategy_{len(self.global_procedures)}")
                if pattern not in self.global_procedures:
                    self.global_procedures[pattern] = self.procedure_class(
                        strategy_pattern=pattern,
                        steps=proc_data.get("steps", ["Analyze", "Execute"]),
                        segments=proc_data.get("segments", ["general"]),
                        success_rate=conf,
                        scope="global",
                        domain_metrics=metrics
                    )
                    self.procedures[pattern] = self.global_procedures[pattern]
                    learned_items["global_learned"] = pattern
                    
                    # Track discovered segments
                    self.segments_discovered.update(proc_data.get("segments", []))
        
        return learned_items if learned_items else {"learned": None}
    
    def _search_procedures(self, procedures_dict, query, profile):
        """Helper method to search through a procedures dictionary"""
        best_match = None
        best_score = 0
        
        for pattern, proc in procedures_dict.items():
            # Simple keyword matching
            query_match = 1.0 if any(
                word in query.lower() 
                for word in pattern.lower().split()
            ) else 0.5
            
            # Check segment compatibility
            segment_match = 0
            if profile:
                for segment in proc.segments:
                    if segment in str(profile.values()).lower():
                        segment_match += 0.5
            
            # Combined score with success rate weighting
            score = (query_match + segment_match) * proc.success_rate
            
            if score > best_score:
                best_score = score
                best_match = proc
        
        if best_match:
            best_match.usage_count += 1
            return {
                "strategy": best_match.strategy_pattern,
                "steps": best_match.steps,
                "confidence": best_match.success_rate,
                "usage_count": best_match.usage_count,
                "match_score": best_score,
                "domain_metrics": best_match.domain_metrics
            }
        
        return None
    
    def get_investment_strategy(self, query: str, client_profile: Dict,
                               user_id: Optional[str] = None) -> Optional[Dict]:
        """Hierarchical retrieval: user → community → task → global"""
        
        # 1. User-specific strategies
        if user_id and user_id in self.user_procedures:
            user_strategy = self._search_procedures(
                self.user_procedures[user_id], query, client_profile
            )
            if user_strategy:
                user_strategy["source"] = f"personalized for {user_id}"
                user_strategy["scope"] = "user"
                return user_strategy
        
        # 2. Community strategies
        if user_id and user_id in self.user_communities:
            for community_id in self.user_communities.get(user_id, []):
                if community_id in self.community_procedures:
                    community_strategy = self._search_procedures(
                        self.community_procedures[community_id], 
                        query, client_profile
                    )
                    if community_strategy:
                        community_strategy["source"] = f"learned from {community_id} community"
                        community_strategy["scope"] = "community"
                        return community_strategy
        
        # 3. Task-specific strategies
        task_type = self._identify_task_type(query)
        if task_type in self.task_procedures:
            task_strategy = self._search_procedures(
                self.task_procedures[task_type], query, client_profile
            )
            if task_strategy:
                task_strategy["source"] = f"specialized for {task_type}"
                task_strategy["scope"] = "task"
                return task_strategy
        
        # 4. Global strategies
        global_strategy = self._search_procedures(
            self.global_procedures, query, client_profile
        )
        if global_strategy:
            global_strategy["source"] = "general best practices"
            global_strategy["scope"] = "global"
            return global_strategy
        
        return None
    
    def update_from_performance(self, strategy: str, performance_data: Dict,
                               scope: str = "global",
                               scope_id: Optional[str] = None) -> Dict:
        """Process performance feedback with scope awareness"""
        
        # Select the right procedure store
        if scope == "user" and scope_id and scope_id in self.user_procedures:
            procedures = self.user_procedures[scope_id]
        elif scope == "community" and scope_id and scope_id in self.community_procedures:
            procedures = self.community_procedures[scope_id]
        elif scope == "task" and scope_id and scope_id in self.task_procedures:
            procedures = self.task_procedures[scope_id]
        else:
            procedures = self.global_procedures
        
        # Find and update the matching procedure
        for pattern, proc in procedures.items():
            if pattern.lower() in strategy.lower() or strategy.lower() in pattern.lower():
                # Use domain agent to calculate success score
                success_score = self.domain_agent.calculate_success_score(performance_data)
                
                # Update success rate
                old_rate = proc.success_rate
                proc.success_rate = min(1.0, proc.success_rate * 0.8 + success_score * 0.2)
                
                # Update domain-specific metrics
                self.domain_agent.update_domain_metrics(proc, performance_data)
                
                # Version history
                proc.adaptations.append({
                    "timestamp": datetime.now().isoformat(),
                    "performance": performance_data,
                    "old_rate": old_rate,
                    "new_rate": proc.success_rate,
                    "success_score": success_score
                })
                
                return {
                    "updated": pattern,
                    "scope": scope,
                    "scope_id": scope_id,
                    "new_success_rate": round(proc.success_rate, 2),
                    "performance_trend": "improving" if proc.success_rate > old_rate else "declining",
                    "total_adaptations": len(proc.adaptations)
                }
        
        return {"updated": None}
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        total_procedures = (
            len(self.global_procedures) +
            sum(len(procs) for procs in self.user_procedures.values()) +
            sum(len(procs) for procs in self.community_procedures.values()) +
            sum(len(procs) for procs in self.task_procedures.values())
        )
        
        if total_procedures == 0:
            return {
                "total_strategies": 0,
                "by_scope": {
                    "global": 0,
                    "user": 0,
                    "community": 0,
                    "task": 0
                },
                "avg_success_rate": 0,
                "total_adaptations": 0,
                "segments": []
            }
        
        # Collect all procedures
        all_procedures = []
        for _, proc in self.global_procedures.items():
            all_procedures.append(proc)
        for _, user_procs in self.user_procedures.items():
            for _, proc in user_procs.items():
                all_procedures.append(proc)
        for _, comm_procs in self.community_procedures.items():
            for _, proc in comm_procs.items():
                all_procedures.append(proc)
        for _, task_procs in self.task_procedures.items():
            for _, proc in task_procs.items():
                all_procedures.append(proc)
        
        avg_success = sum(p.success_rate for p in all_procedures) / len(all_procedures) if all_procedures else 0
        total_adaptations = sum(len(p.adaptations) for p in all_procedures)
        
        return {
            "total_strategies": total_procedures,
            "by_scope": {
                "global": len(self.global_procedures),
                "user": sum(len(procs) for procs in self.user_procedures.values()),
                "community": sum(len(procs) for procs in self.community_procedures.values()),
                "task": sum(len(procs) for procs in self.task_procedures.values())
            },
            "avg_success_rate": round(avg_success, 2),
            "total_adaptations": total_adaptations,
            "segments": list(self.segments_discovered)
        }
    
    def show_strategy_performance(self) -> None:
        """Visualize strategy effectiveness by segmentation level"""
        
        print("\n📊 Strategy Performance by Scope:")
        print("=" * 60)
        
        # Global strategies
        print("\n🌍 GLOBAL STRATEGIES (Universal Best Practices):")
        if self.global_procedures:
            for strategy_name, proc in self.global_procedures.items():
                success_bar = "█" * int(proc.success_rate * 10)
                empty = "░" * (10 - int(proc.success_rate * 10))
                print(f"  {strategy_name[:30]:<30} {success_bar}{empty} {proc.success_rate:.1%}")
                if proc.usage_count > 0:
                    print(f"    Used {proc.usage_count}x | Segments: {', '.join(proc.segments[:2])}")
        else:
            print("  No global strategies learned yet")
        
        # User-specific strategies
        print("\n👤 USER-SPECIFIC STRATEGIES:")
        if self.user_procedures:
            total_user_procs = sum(len(procs) for procs in self.user_procedures.values())
            print(f"  Total users with personalized strategies: {len(self.user_procedures)}")
            print(f"  Total personalized procedures: {total_user_procs}")
            
            if self.user_procedures:
                sample_user = list(self.user_procedures.keys())[0]
                sample_procs = self.user_procedures[sample_user]
                if sample_procs:
                    print(f"\n  Example - User {sample_user}:")
                    for pattern, proc in list(sample_procs.items())[:2]:
                        print(f"    • {pattern[:40]} (success: {proc.success_rate:.1%})")
        else:
            print("  No user-specific strategies learned yet")
        
        # Community strategies
        print("\n👥 COMMUNITY STRATEGIES:")
        if self.community_procedures:
            for community_id, procedures in self.community_procedures.items():
                member_count = len(self.community_members.get(community_id, set()))
                avg_success = sum(p.success_rate for p in procedures.values()) / len(procedures) if procedures else 0
                print(f"  {community_id}: {len(procedures)} strategies")
                print(f"    Members: {member_count} | Avg success: {avg_success:.1%}")
        else:
            print("  No community strategies learned yet")
        
        # Task-specific strategies
        print("\n📋 TASK-SPECIFIC STRATEGIES:")
        if self.task_procedures:
            for task_type, procedures in self.task_procedures.items():
                avg_success = sum(p.success_rate for p in procedures.values()) / len(procedures) if procedures else 0
                print(f"  {task_type}: {len(procedures)} procedures (avg success: {avg_success:.1%})")
        else:
            print("  No task-specific strategies learned yet")