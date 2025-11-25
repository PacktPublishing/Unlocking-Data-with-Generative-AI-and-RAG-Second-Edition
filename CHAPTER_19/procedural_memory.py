# procedural_memory.py
"""
LangMem-powered procedural memory system.
Uses create_memory_manager for extraction and create_prompt_optimizer for continuous improvement.
"""

import asyncio
import json
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import nest_asyncio
from pydantic import BaseModel, Field
from langmem import create_memory_manager, create_prompt_optimizer

from domain_agent import DomainAgent, DomainProcedure

nest_asyncio.apply()


class ExtractedProcedure(BaseModel):
    """Schema for LangMem procedure extraction"""
    strategy_pattern: str = Field(description="Name/description of the strategy pattern")
    steps: List[str] = Field(description="Ordered steps to execute this strategy")
    applicable_segments: List[str] = Field(description="User segments this applies to")
    reasoning: str = Field(description="Why this strategy works")


class ProceduralMemory:
    """LangMem-powered procedural memory with hierarchical learning"""
    
    def __init__(self, llm, domain_agent: DomainAgent, optimization_algorithm: str = "prompt_memory"):
        self.llm = llm
        self.domain_agent = domain_agent
        self.procedure_class = domain_agent.get_procedure_class()
        self.community_definitions = domain_agent.get_community_definitions()
        
        self.user_procedures: Dict[str, Dict] = {}
        self.community_procedures: Dict[str, Dict] = {}
        self.global_procedures: Dict[str, DomainProcedure] = {}
        self.task_procedures: Dict[str, Dict] = {}
        self.procedures: Dict[str, DomainProcedure] = {}
        
        self.user_learning_history = defaultdict(list)
        self.community_learning_history = defaultdict(list)
        self.global_learning_history: List = []
        self.learning_history: List = []
        
        self.user_communities: Dict[str, List[str]] = {}
        self.community_members: Dict[str, set] = defaultdict(set)
        self.segments_discovered: set = set()
        
        self.optimization_algorithm = optimization_algorithm
        self.model_str = f"openai:{llm.model_name}" if hasattr(llm, 'model_name') else "openai:gpt-4.1-mini"
        
        prompts = self.domain_agent.get_learning_prompts()
        self.extraction_instructions = {
            "global": prompts.get("global", "Extract procedural strategies from this interaction."),
            "user": prompts.get("user", "Extract user-specific preferences from this interaction."),
            "community": prompts.get("community", "Extract community-level patterns from this interaction."),
            "task": prompts.get("task", "Extract task-specific procedures from this interaction.")
        }
        
        self._init_memory_managers()
        self._init_optimizer()
        
        self.conversation_buffer: List[Dict] = []
        self.optimization_threshold = 5
        self.total_optimizations = 0
        self.optimization_history: List[Dict] = []
    
    def _init_memory_managers(self):
        """Initialize LangMem memory managers"""
        self.global_memory_manager = create_memory_manager(
            self.model_str,
            schemas=[ExtractedProcedure],
            instructions=self.extraction_instructions["global"]
        )
        self.user_memory_manager = create_memory_manager(
            self.model_str,
            schemas=[ExtractedProcedure],
            instructions=self.extraction_instructions["user"]
        )
        self.community_memory_manager = create_memory_manager(
            self.model_str,
            schemas=[ExtractedProcedure],
            instructions=self.extraction_instructions["community"]
        )
        self.task_memory_manager = create_memory_manager(
            self.model_str,
            schemas=[ExtractedProcedure],
            instructions=self.extraction_instructions["task"]
        )
    
    def _init_optimizer(self):
        """Initialize LangMem prompt optimizer"""
        if self.optimization_algorithm == "metaprompt":
            self.optimizer = create_prompt_optimizer(
                self.model_str,
                kind="metaprompt",
                config={"max_reflection_steps": 3, "min_reflection_steps": 1}
            )
        elif self.optimization_algorithm == "gradient":
            self.optimizer = create_prompt_optimizer(self.model_str, kind="gradient")
        else:
            self.optimizer = create_prompt_optimizer(self.model_str, kind="prompt_memory")
    
    def _identify_task_type(self, query: str) -> str:
        return self.domain_agent.identify_task_type(query)
    
    def _run_async(self, coro):
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)
    
    async def _extract_with_langmem(
        self,
        memory_manager,
        query: str,
        interaction_data: Dict,
        scope: str,
        scope_id: Optional[str] = None
    ) -> Optional[Dict]:
        """Extract procedure using LangMem memory manager"""
        try:
            conversation = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": json.dumps(interaction_data)}
            ]
            
            memories = await memory_manager(conversation)
            
            if memories and len(memories) > 0:
                memory_tuple = memories[0]
                
                if isinstance(memory_tuple, tuple) and len(memory_tuple) >= 2:
                    memory_content = memory_tuple[1]
                else:
                    memory_content = memory_tuple
                
                if isinstance(memory_content, ExtractedProcedure):
                    return {
                        "strategy_pattern": memory_content.strategy_pattern,
                        "steps": memory_content.steps,
                        "segments": memory_content.applicable_segments,
                        "confidence": 0.85,
                        "domain_metrics": {}
                    }
                
                if isinstance(memory_content, dict):
                    return {
                        "strategy_pattern": memory_content.get("strategy_pattern", f"{scope}_strategy"),
                        "steps": memory_content.get("steps", [str(memory_content)]),
                        "segments": memory_content.get("applicable_segments", memory_content.get("segments", [scope])),
                        "confidence": 0.85,
                        "domain_metrics": {}
                    }
                
                if isinstance(memory_content, str):
                    return {
                        "strategy_pattern": f"{scope}_strategy",
                        "steps": [memory_content],
                        "segments": [scope],
                        "confidence": 0.8,
                        "domain_metrics": {}
                    }
                    
        except Exception as e:
            print(f"LangMem extraction failed for {scope}: {e}")
        
        return None
    
    def learn_from_interaction(
        self,
        query: str,
        interaction_data: Dict,
        user_id: Optional[str] = None,
        user_profile: Optional[Dict] = None
    ) -> Dict:
        """Learn procedures at multiple hierarchy levels using LangMem"""
        learned_items = {}
        
        task_type = self._identify_task_type(query)
        if task_type != "general":
            if task_type not in self.task_procedures:
                self.task_procedures[task_type] = {}
            
            if len(self.task_procedures[task_type]) < 2:
                proc_data = self._run_async(
                    self._extract_with_langmem(self.task_memory_manager, query, interaction_data, "task", task_type)
                )
                if proc_data:
                    pattern = f"{task_type}_strategy_{len(self.task_procedures[task_type])}"
                    self.task_procedures[task_type][pattern] = self.procedure_class(
                        strategy_pattern=pattern,
                        steps=proc_data.get("steps", ["Execute task"]),
                        segments=proc_data.get("segments", []),
                        success_rate=proc_data.get("confidence", 0.85),
                        scope="task",
                        scope_id=task_type,
                        domain_metrics=proc_data.get("domain_metrics", {})
                    )
                    learned_items["task_learned"] = pattern
        
        if user_id:
            if user_id not in self.user_procedures:
                self.user_procedures[user_id] = {}
            
            if len(self.user_procedures[user_id]) == 0:
                proc_data = self._run_async(
                    self._extract_with_langmem(self.user_memory_manager, query, interaction_data, "user", user_id)
                )
                if proc_data:
                    pattern = f"user_{user_id}_preference_0"
                    self.user_procedures[user_id][pattern] = self.procedure_class(
                        strategy_pattern=pattern,
                        steps=proc_data.get("steps", [f"Personalized for {user_id}"]),
                        segments=proc_data.get("segments", []),
                        success_rate=proc_data.get("confidence", 0.9),
                        scope="user",
                        scope_id=user_id,
                        domain_metrics=proc_data.get("domain_metrics", {})
                    )
                    learned_items["user_learned"] = pattern
        
        if user_id:
            if user_id not in self.user_communities:
                community = self.domain_agent.identify_community(user_id, user_profile)
                self.user_communities[user_id] = [community]
                self.community_members[community].add(user_id)
            
            for community_id in self.user_communities[user_id]:
                if community_id not in self.community_procedures:
                    self.community_procedures[community_id] = {}
                
                if len(self.community_procedures[community_id]) == 0:
                    proc_data = self._run_async(
                        self._extract_with_langmem(self.community_memory_manager, query, interaction_data, "community", community_id)
                    )
                    if proc_data:
                        pattern = f"{community_id}_pattern_0"
                        self.community_procedures[community_id][pattern] = self.procedure_class(
                            strategy_pattern=pattern,
                            steps=proc_data.get("steps", ["Community approach"]),
                            segments=proc_data.get("segments", []),
                            success_rate=proc_data.get("confidence", 0.87),
                            scope="community",
                            scope_id=community_id,
                            domain_metrics=proc_data.get("domain_metrics", {})
                        )
                        learned_items["community_learned"] = pattern
        
        if len(self.global_procedures) < 5:
            proc_data = self._run_async(
                self._extract_with_langmem(self.global_memory_manager, query, interaction_data, "global")
            )
            if proc_data:
                pattern = proc_data.get("strategy_pattern", f"global_strategy_{len(self.global_procedures)}")
                if pattern not in self.global_procedures:
                    self.global_procedures[pattern] = self.procedure_class(
                        strategy_pattern=pattern,
                        steps=proc_data.get("steps", ["Analyze", "Execute"]),
                        segments=proc_data.get("segments", ["general"]),
                        success_rate=proc_data.get("confidence", 0.85),
                        scope="global",
                        domain_metrics=proc_data.get("domain_metrics", {})
                    )
                    self.procedures[pattern] = self.global_procedures[pattern]
                    learned_items["global_learned"] = pattern
                    self.segments_discovered.update(proc_data.get("segments", []))
        
        self._buffer_for_optimization(query, interaction_data, learned_items)
        
        return learned_items if learned_items else {"learned": None}
    
    def _buffer_for_optimization(self, query: str, interaction_data: Dict, learned: Dict) -> None:
        """Buffer interaction for batch optimization"""
        self.conversation_buffer.append({
            "messages": [
                {"role": "user", "content": query},
                {"role": "assistant", "content": json.dumps(interaction_data)}
            ],
            "feedback": {
                "success": interaction_data.get("success", True),
                "satisfaction_score": interaction_data.get("client_satisfaction", 5),
                "learned": learned
            },
            "timestamp": datetime.now().isoformat()
        })
        
        if len(self.conversation_buffer) >= self.optimization_threshold:
            self._run_async(self.run_optimization())
    
    async def run_optimization(self) -> Optional[Dict]:
        """Run LangMem prompt optimizer"""
        if not self.conversation_buffer:
            return None
        
        print(f"\n🔄 Running LangMem {self.optimization_algorithm} optimization...")
        print(f"   Processing {len(self.conversation_buffer)} buffered interactions")
        
        try:
            trajectories = []
            for conv in self.conversation_buffer:
                messages = conv.get("messages", [])
                feedback = conv.get("feedback", {})
                feedback_str = f"Success: {feedback.get('success', True)}, Satisfaction: {feedback.get('satisfaction_score', 5)}"
                trajectories.append((messages, feedback_str))
            
            current_prompt = self.extraction_instructions["global"]
            result = await self.optimizer.ainvoke({"trajectories": trajectories, "prompt": current_prompt})
            
            improved_prompt = result if isinstance(result, str) else result
            self.extraction_instructions["global"] = improved_prompt
            
            self.global_memory_manager = create_memory_manager(
                self.model_str,
                schemas=[ExtractedProcedure],
                instructions=improved_prompt
            )
            
            self.total_optimizations += 1
            self.optimization_history.append({
                "timestamp": datetime.now().isoformat(),
                "algorithm": self.optimization_algorithm,
                "conversations_processed": len(self.conversation_buffer)
            })
            
            print(f"✅ LangMem optimization #{self.total_optimizations} complete")
            
            conversations_processed = len(self.conversation_buffer)
            self.conversation_buffer.clear()
            
            return {"success": True, "optimizations_total": self.total_optimizations, "conversations_processed": conversations_processed}
            
        except Exception as e:
            print(f"⚠️ LangMem optimization failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def force_optimization(self) -> Optional[Dict]:
        """Force optimization even if threshold not reached"""
        if not self.conversation_buffer:
            print("⚠️ No conversations buffered for optimization")
            return None
        return await self.run_optimization()
    
    def _search_procedures(self, procedures_dict: Dict, query: str, profile: Dict) -> Optional[Dict]:
        """Search procedures for best match"""
        best_match = None
        best_score = 0
        
        for pattern, proc in procedures_dict.items():
            query_match = 1.0 if any(word in query.lower() for word in pattern.lower().split()) else 0.5
            
            segment_match = 0
            if profile:
                for segment in proc.segments:
                    if segment in str(profile.values()).lower():
                        segment_match += 0.5
            
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
    
    def get_investment_strategy(self, query: str, client_profile: Dict, user_id: Optional[str] = None) -> Optional[Dict]:
        """Hierarchical retrieval: user → community → task → global"""
        if user_id and user_id in self.user_procedures:
            strategy = self._search_procedures(self.user_procedures[user_id], query, client_profile)
            if strategy:
                strategy["source"] = f"personalized for {user_id}"
                strategy["scope"] = "user"
                return strategy
        
        if user_id and user_id in self.user_communities:
            for community_id in self.user_communities.get(user_id, []):
                if community_id in self.community_procedures:
                    strategy = self._search_procedures(self.community_procedures[community_id], query, client_profile)
                    if strategy:
                        strategy["source"] = f"learned from {community_id} community"
                        strategy["scope"] = "community"
                        return strategy
        
        task_type = self._identify_task_type(query)
        if task_type in self.task_procedures:
            strategy = self._search_procedures(self.task_procedures[task_type], query, client_profile)
            if strategy:
                strategy["source"] = f"specialized for {task_type}"
                strategy["scope"] = "task"
                return strategy
        
        strategy = self._search_procedures(self.global_procedures, query, client_profile)
        if strategy:
            strategy["source"] = "general best practices"
            strategy["scope"] = "global"
            return strategy
        
        return None
    
    def update_from_performance(self, strategy: str, performance_data: Dict, scope: str = "global", scope_id: Optional[str] = None) -> Dict:
        """Update procedure based on performance feedback"""
        if scope == "user" and scope_id and scope_id in self.user_procedures:
            procedures = self.user_procedures[scope_id]
        elif scope == "community" and scope_id and scope_id in self.community_procedures:
            procedures = self.community_procedures[scope_id]
        elif scope == "task" and scope_id and scope_id in self.task_procedures:
            procedures = self.task_procedures[scope_id]
        else:
            procedures = self.global_procedures
        
        for pattern, proc in procedures.items():
            if pattern.lower() in strategy.lower() or strategy.lower() in pattern.lower():
                success_score = self.domain_agent.calculate_success_score(performance_data)
                old_rate = proc.success_rate
                proc.success_rate = min(1.0, proc.success_rate * 0.8 + success_score * 0.2)
                self.domain_agent.update_domain_metrics(proc, performance_data)
                
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
        total = (
            len(self.global_procedures) +
            sum(len(p) for p in self.user_procedures.values()) +
            sum(len(p) for p in self.community_procedures.values()) +
            sum(len(p) for p in self.task_procedures.values())
        )
        
        if total == 0:
            return {
                "total_strategies": 0,
                "by_scope": {"global": 0, "user": 0, "community": 0, "task": 0},
                "avg_success_rate": 0,
                "total_adaptations": 0,
                "segments": [],
                "langmem": {
                    "algorithm": self.optimization_algorithm,
                    "total_optimizations": self.total_optimizations,
                    "pending_conversations": len(self.conversation_buffer)
                }
            }
        
        all_procs = list(self.global_procedures.values())
        for procs in self.user_procedures.values():
            all_procs.extend(procs.values())
        for procs in self.community_procedures.values():
            all_procs.extend(procs.values())
        for procs in self.task_procedures.values():
            all_procs.extend(procs.values())
        
        avg_success = sum(p.success_rate for p in all_procs) / len(all_procs)
        total_adaptations = sum(len(p.adaptations) for p in all_procs)
        
        return {
            "total_strategies": total,
            "by_scope": {
                "global": len(self.global_procedures),
                "user": sum(len(p) for p in self.user_procedures.values()),
                "community": sum(len(p) for p in self.community_procedures.values()),
                "task": sum(len(p) for p in self.task_procedures.values())
            },
            "avg_success_rate": round(avg_success, 2),
            "total_adaptations": total_adaptations,
            "segments": list(self.segments_discovered),
            "langmem": {
                "algorithm": self.optimization_algorithm,
                "total_optimizations": self.total_optimizations,
                "pending_conversations": len(self.conversation_buffer)
            }
        }
    
    def show_strategy_performance(self) -> None:
        """Display strategy performance visualization"""
        print("\n📊 Strategy Performance by Scope:")
        print("=" * 60)
        
        print("\n🌍 GLOBAL STRATEGIES:")
        if self.global_procedures:
            for name, proc in self.global_procedures.items():
                bar = "█" * int(proc.success_rate * 10) + "░" * (10 - int(proc.success_rate * 10))
                print(f"  {name[:30]:<30} {bar} {proc.success_rate:.1%}")
                if proc.usage_count > 0:
                    print(f"    Used {proc.usage_count}x | Segments: {', '.join(proc.segments[:2])}")
        else:
            print("  No global strategies yet")
        
        print("\n👤 USER-SPECIFIC STRATEGIES:")
        if self.user_procedures:
            total = sum(len(p) for p in self.user_procedures.values())
            print(f"  {len(self.user_procedures)} users, {total} procedures")
        else:
            print("  No user strategies yet")
        
        print("\n👥 COMMUNITY STRATEGIES:")
        if self.community_procedures:
            for cid, procs in self.community_procedures.items():
                members = len(self.community_members.get(cid, set()))
                print(f"  {cid}: {len(procs)} strategies, {members} members")
        else:
            print("  No community strategies yet")
        
        print("\n📋 TASK-SPECIFIC STRATEGIES:")
        if self.task_procedures:
            for task, procs in self.task_procedures.items():
                print(f"  {task}: {len(procs)} procedures")
        else:
            print("  No task strategies yet")
        
        print("\n🧠 LANGMEM STATUS:")
        print(f"  Algorithm: {self.optimization_algorithm}")
        print(f"  Optimizations completed: {self.total_optimizations}")
        print(f"  Pending conversations: {len(self.conversation_buffer)}")