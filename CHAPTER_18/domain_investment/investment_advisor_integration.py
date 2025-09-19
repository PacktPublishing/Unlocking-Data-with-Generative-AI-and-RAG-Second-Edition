# investment_advisor_integration.py (updated)
"""
Integration layer to connect investment advisor data with CoALA full agent
Demonstrates episodic, semantic, and procedural memory learning
"""

import os
import sys
import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coala_agent import CoALAAgent, SemanticFact
from langchain_core.messages import HumanMessage, AIMessage
from .investment_advisor_agent import InvestmentAdvisorAgent


class InvestmentAdvisorIntegration:
    """
    Integration layer that connects the investment advisor data 
    with the CoALA agent for complete memory demonstration
    """
    
    def __init__(self, 
                 agent: Optional[CoALAAgent] = None,
                 data_dir: str = None):
        """
        Initialize the integration layer
        
        Args:
            agent: Existing CoALA agent or None to create new
            data_dir: Directory containing the generated data
        """
        # Set domain directory
        self.domain_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Use domain data directory if not specified
        if data_dir is None:
            self.data_dir = os.path.join(self.domain_dir, "investment_advisor_data")
        else:
            self.data_dir = data_dir
        
        # Create domain agent
        domain_agent = InvestmentAdvisorAgent()
        
        # Create CoALA agent with domain agent
        self.agent = agent or CoALAAgent(
            domain_agent=domain_agent,
            model_name="gpt-4.1-mini",
            persist_directory=domain_agent.memory_dir  # Use domain memory directory
        )
        
        # Load the generated data
        self.load_data()
        
        # Track learning progress
        self.learning_stats = {
            "conversations_processed": 0,
            "facts_extracted": 0,
            "episodes_stored": 0,
            "patterns_identified": 0,
            "procedural_optimizations": 0,
            "current_phase": "baseline"
        }
        
        # Track user-specific memories
        self.user_memories = {}
    
    def load_data(self):
        """Load all generated data files"""
        try:
            # Load conversations from JSONL format
            self.conversations = []
            with open(f"{self.data_dir}/conversations.jsonl", "r") as f:
                for line in f:
                    self.conversations.append(json.loads(line))
            
            # Load extracted patterns
            with open(f"{self.data_dir}/extracted_patterns.json", "r") as f:
                self.extracted_patterns = json.load(f)
            
            # Load test scenarios
            with open(f"{self.data_dir}/test_scenarios.json", "r") as f:
                self.test_scenarios = json.load(f)
            
            # Load statistics
            with open(f"{self.data_dir}/statistics.json", "r") as f:
                self.statistics = json.load(f)
            
            print(f"✅ Loaded data from {self.data_dir}")
            print(f"   • {len(self.conversations)} conversations")
            print(f"   • {len(self.extracted_patterns['universal_rules'])} universal patterns")
            print(f"   • {len(self.test_scenarios)} test scenarios")
            print(f"   • Success rate in data: {self.statistics['success_rate']:.1%}")
            
        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")
            print("   Please run the investment_advisor_data.py first to generate data")
            raise
    
    async def ingest_conversation_with_procedural(self, 
                                                 conversation: Dict,
                                                 store_episodic: bool = True,
                                                 extract_semantic: bool = True,
                                                 add_to_procedural: bool = True) -> Dict[str, Any]:
        """
        Ingest a single conversation into all memory systems including procedural
        
        Args:
            conversation: Conversation data from generated dataset
            store_episodic: Whether to store as episodic memory
            extract_semantic: Whether to extract semantic facts
            add_to_procedural: Whether to add to procedural learning buffer
            
        Returns:
            Summary of what was stored
        """
        # Handle new data structure
        conv_id = conversation["conversation_id"]
        user_id = str(conversation["user_id"])
        messages = conversation["messages"]
        feedback = conversation["feedback"]
        behavioral_signals = conversation["behavioral_signals"]
        metadata = conversation["metadata"]
        
        # Convert messages to agent format
        agent_messages = []
        for msg in messages:
            if msg["role"] == "user":
                agent_messages.append(HumanMessage(content=msg["content"]))
            else:
                agent_messages.append(AIMessage(content=msg["content"]))
        
        result = {
            "conversation_id": conv_id,
            "user_id": user_id,
            "episodic_stored": False,
            "semantic_facts": [],
            "behavioral_patterns": [],
            "procedural_triggered": False
        }
        
        # Store episodic memory
        if store_episodic:
            # Create conversation summary based on feedback
            satisfaction = feedback["satisfaction_score"]
            success = feedback["success"]
            outcome = "successful" if success else "unsuccessful"
            summary = f"{outcome} {metadata['topic']} discussion - satisfaction: {satisfaction:.1f}/5.0"
            
            self.agent.store_episodic_memory(
                conversation_id=conv_id,
                messages=agent_messages,
                summary=summary
            )
            result["episodic_stored"] = True
            self.learning_stats["episodes_stored"] += 1
        
        # Extract and store semantic facts
        if extract_semantic:
            # Extract facts using agent's extraction
            extracted_facts = self.agent.extract_semantic_facts(agent_messages)
            
            # Also create facts from the conversation metadata
            if metadata.get("query_type") == "performance_analysis":
                extracted_facts.append(SemanticFact(
                    subject=f"User_{user_id}",
                    predicate="interested_in",
                    object="performance_tracking",
                    confidence=0.8,
                    source="conversation"
                ))
            
            if extracted_facts:
                stored_count = self.agent.store_semantic_facts(
                    extracted_facts, 
                    user_id=user_id
                )
                result["semantic_facts"] = [
                    f"{f.subject} {f.predicate} {f.object}" 
                    for f in extracted_facts
                ]
                self.learning_stats["facts_extracted"] += stored_count
        
        # Add to procedural memory buffer
        if add_to_procedural:
            # Convert to format expected by procedural memory
            procedural_messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in messages
            ]
            
            # Create feedback dict for procedural learning
            procedural_feedback = {
                "success": feedback["success"],
                "satisfaction": feedback["satisfaction_score"]
            }
            
            # Add to buffer and check if optimization triggered
            optimization_triggered = await self.agent.add_conversation_for_learning(
                procedural_messages,
                procedural_feedback
            )
            
            if optimization_triggered:
                result["procedural_triggered"] = True
                self.learning_stats["procedural_optimizations"] += 1
                print(f"   🔧 Procedural optimization triggered!")
        
        # Track behavioral patterns
        for signal, value in behavioral_signals.items():
            if value:
                result["behavioral_patterns"].append(signal)
        
        self.learning_stats["patterns_identified"] += len(result["behavioral_patterns"])
        
        # Track user-specific patterns
        if user_id not in self.user_memories:
            self.user_memories[user_id] = {
                "conversations": 0,
                "satisfaction_avg": 0,
                "topics": set()
            }
        
        self.user_memories[user_id]["conversations"] += 1
        self.user_memories[user_id]["topics"].add(metadata["topic"])
        
        # Update running average of satisfaction
        prev_avg = self.user_memories[user_id]["satisfaction_avg"]
        n = self.user_memories[user_id]["conversations"]
        self.user_memories[user_id]["satisfaction_avg"] = (
            (prev_avg * (n - 1) + feedback["satisfaction_score"]) / n
        )
        
        self.learning_stats["conversations_processed"] += 1
        
        return result
    
    async def batch_ingest_conversations(self, 
                                        conversations: Optional[List[Dict]] = None,
                                        limit: Optional[int] = None,
                                        filter_criteria: Optional[Dict] = None,
                                        include_procedural: bool = True) -> Dict:
        """
        Ingest multiple conversations in batch with procedural learning
        
        Args:
            conversations: List of conversations or None for all data
            limit: Maximum number to process
            filter_criteria: Filter conversations (e.g., by satisfaction, success)
            include_procedural: Whether to include procedural memory learning
            
        Returns:
            Summary statistics
        """
        if conversations is None:
            conversations = self.conversations
        
        # Apply filters if specified
        if filter_criteria:
            filtered = []
            for conv in conversations:
                include = True
                
                if "min_satisfaction" in filter_criteria:
                    if conv["feedback"]["satisfaction_score"] < filter_criteria["min_satisfaction"]:
                        include = False
                
                if "success" in filter_criteria:
                    if conv["feedback"]["success"] != filter_criteria["success"]:
                        include = False
                
                if "query_type" in filter_criteria:
                    if conv["metadata"]["query_type"] != filter_criteria["query_type"]:
                        include = False
                
                if include:
                    filtered.append(conv)
            
            conversations = filtered
        
        # Apply limit
        if limit:
            conversations = conversations[:limit]
        
        print(f"\n📝 Batch ingesting {len(conversations)} conversations...")
        if include_procedural:
            print(f"   (Including procedural memory with threshold: {self.agent.optimization_threshold})")
        
        results = {
            "total_processed": 0,
            "episodic_stored": 0,
            "semantic_facts_extracted": 0,
            "patterns_identified": 0,
            "procedural_optimizations": 0,
            "by_user": {}
        }
        
        for i, conv in enumerate(conversations):
            if i % 10 == 0 and i > 0:
                print(f"   Processed {i}/{len(conversations)} conversations...")
            
            result = await self.ingest_conversation_with_procedural(
                conv, 
                add_to_procedural=include_procedural
            )
            
            results["total_processed"] += 1
            if result["episodic_stored"]:
                results["episodic_stored"] += 1
            results["semantic_facts_extracted"] += len(result["semantic_facts"])
            results["patterns_identified"] += len(result["behavioral_patterns"])
            if result["procedural_triggered"]:
                results["procedural_optimizations"] += 1
            
            # Track by user
            user_id = result["user_id"]
            if user_id not in results["by_user"]:
                results["by_user"][user_id] = {
                    "conversations": 0,
                    "facts": 0,
                    "patterns": 0
                }
            results["by_user"][user_id]["conversations"] += 1
            results["by_user"][user_id]["facts"] += len(result["semantic_facts"])
            results["by_user"][user_id]["patterns"] += len(result["behavioral_patterns"])
        
        print(f"✅ Batch ingestion complete:")
        print(f"   • Episodes stored: {results['episodic_stored']}")
        print(f"   • Facts extracted: {results['semantic_facts_extracted']}")
        print(f"   • Patterns identified: {results['patterns_identified']}")
        print(f"   • Procedural optimizations: {results['procedural_optimizations']}")
        print(f"   • Users processed: {len(results['by_user'])}")
        
        return results
    
    def demonstrate_memory_recall(self, query: str, user_id: Optional[str] = None):
        """
        Demonstrate how the agent recalls different types of memory
        
        Args:
            query: The query to search memories
            user_id: Optional user ID for personalized recall
        """
        print(f"\n🔍 Memory Recall for: '{query}'")
        print("=" * 60)
        
        # Episodic recall
        print("\n📚 EPISODIC MEMORY (Past Conversations):")
        episodes = self.agent.retrieve_episodic_memories(query, k=2)
        if episodes:
            for i, episode in enumerate(episodes, 1):
                print(f"\n   Episode {i}:")
                print(f"   Conversation: {episode.metadata.get('conversation_id', 'Unknown')}")
                print(f"   Time: {episode.metadata.get('timestamp', 'Unknown')}")
                print(f"   Preview: {episode.page_content[:150]}...")
        else:
            print("   No relevant episodes found")
        
        # Semantic recall
        print("\n💡 SEMANTIC MEMORY (Known Facts):")
        if user_id:
            self.agent.current_user_id = user_id
        facts = self.agent.retrieve_semantic_facts(query, user_id=user_id, k=5)
        if facts:
            for fact in facts:
                conf = fact.get('confidence', 1.0)
                print(f"   • {fact['subject']} {fact['predicate']} {fact['object']} (confidence: {conf:.2f})")
        else:
            print("   No relevant facts found")
        
        # Procedural memory (current system prompt)
        print("\n🔧 PROCEDURAL MEMORY (Learned System Prompt):")
        current_prompt = self.agent.current_system_prompt
        print(f"   Current prompt: {current_prompt[:200]}...")
        print(f"   Total optimizations: {self.agent.total_optimizations}")
        
        # Procedural patterns from extracted data
        print("\n📋 IDENTIFIED PATTERNS (from data analysis):")
        relevant_patterns = self._get_relevant_patterns(query)
        if relevant_patterns:
            for pattern in relevant_patterns:
                print(f"   • {pattern}")
        else:
            print("   No relevant patterns identified")
    
    def _get_relevant_patterns(self, query: str) -> List[str]:
        """Get patterns relevant to the query from extracted patterns"""
        patterns = []
        query_lower = query.lower()
        
        # Check universal rules
        for rule in self.extracted_patterns.get("universal_rules", []):
            if any(word in rule.lower() for word in query_lower.split()):
                patterns.append(rule)
        
        # Add specific patterns based on query content
        if "volatility" in query_lower or "risk" in query_lower:
            patterns.append("Acknowledge concerns when users express worry")
            
        if "beginner" in query_lower or "start" in query_lower:
            patterns.append("Break down complex financial terms into plain language")
        
        return patterns[:3]  # Limit to top 3 patterns
    
    def test_agent_with_scenario(self, scenario: Dict) -> Dict[str, Any]:
        """
        Test the agent with a specific scenario
        
        Args:
            scenario: Test scenario from generated data
            
        Returns:
            Test results including response and evaluation
        """
        print(f"\n🧪 Testing Scenario: {scenario['scenario_id']}")
        print(f"   Description: {scenario['description']}")
        print(f"   User: {scenario['user_message']}")
        
        # Get agent response
        response = self.agent.process_message(
            scenario["user_message"],
            user_id="test_user",
            conversation_id=f"test_{scenario['scenario_id']}"
        )
        
        print(f"   Agent Response: {response[:200]}...")
        
        # Evaluate response
        evaluation = self._evaluate_response(
            response, 
            scenario.get("expected_improvements", []),
            scenario.get("improved_response", "")
        )
        
        print(f"   Evaluation:")
        for criterion, met in evaluation["criteria_met"].items():
            status = "✓" if met else "✗"
            print(f"      {status} {criterion}")
        print(f"   Overall Score: {evaluation['score']:.1%}")
        
        return {
            "scenario": scenario,
            "response": response,
            "evaluation": evaluation
        }
    
    def _evaluate_response(self, 
                          response: str,
                          expected_improvements: List[str],
                          improved_response: str) -> Dict:
        """Evaluate response against expected improvements"""
        response_lower = response.lower()
        
        criteria_met = {}
        for improvement in expected_improvements:
            # Check for various improvement patterns
            if improvement == "ask_clarification":
                criteria_met[improvement] = "?" in response
            elif improvement == "provide_options":
                criteria_met[improvement] = any(word in response_lower for word in ["or", "would you", "are you"])
            elif improvement == "explain_concept":
                criteria_met[improvement] = any(phrase in response_lower for phrase in ["this means", "in other words", "let me explain"])
            elif improvement == "acknowledge_concern":
                criteria_met[improvement] = any(word in response_lower for word in ["understand", "concern", "worry"])
            elif improvement == "provide_perspective":
                criteria_met[improvement] = any(word in response_lower for word in ["historical", "typically", "perspective"])
            else:
                criteria_met[improvement] = False
        
        score = sum(criteria_met.values()) / len(expected_improvements) if expected_improvements else 0.5
        
        return {
            "criteria_met": criteria_met,
            "score": score,
            "expected": improved_response[:100] + "..."
        }
    
    async def run_progressive_learning_demo(self):
        """
        Run a complete demonstration of progressive learning with all memory types
        Shows how the agent improves through different phases
        """
        print("\n" + "="*80)
        print("INVESTMENT ADVISOR PROGRESSIVE LEARNING DEMONSTRATION")
        print("WITH FULL MEMORY INTEGRATION (Episodic + Semantic + Procedural)")
        print("="*80)
        
        # Phase 1: Baseline (test before any learning)
        print("\n📊 PHASE 1: Baseline Performance")
        print("-" * 40)
        self.learning_stats["current_phase"] = "baseline"
        print("Testing agent with no conversation history...")
        baseline_results = []
        for scenario in self.test_scenarios[:2]:
            result = self.test_agent_with_scenario(scenario)
            baseline_results.append(result)
        baseline_score = sum(r["evaluation"]["score"] for r in baseline_results) / len(baseline_results) if baseline_results else 0
        print(f"\n📈 Baseline Performance: {baseline_score:.1%}")
        
        # Show initial procedural state
        print(f"\n🔧 Initial System Prompt: {self.agent.current_system_prompt}")
        
        # Phase 2: Learn from failed conversations
        print("\n📊 PHASE 2: Learning from Failed Interactions")
        print("-" * 40)
        self.learning_stats["current_phase"] = "early_learning"
        
        # Ingest failed conversations
        failed_convs = [c for c in self.conversations if not c["feedback"]["success"]][:10]
        print(f"Ingesting {len(failed_convs)} failed conversations to learn from mistakes...")
        await self.batch_ingest_conversations(failed_convs, include_procedural=True)
        
        # Check if procedural optimization occurred
        if self.agent.total_optimizations > 0:
            print(f"\n🔧 Procedural Memory Updated after failures!")
            print(f"   New prompt: {self.agent.current_system_prompt[:150]}...")
        
        # Test after learning from failures
        print("\nTesting after learning from failures...")
        early_results = []
        for scenario in self.test_scenarios[:2]:
            result = self.test_agent_with_scenario(scenario)
            early_results.append(result)
        early_score = sum(r["evaluation"]["score"] for r in early_results) / len(early_results) if early_results else 0
        print(f"\n📈 Early Learning Performance: {early_score:.1%}")
        print(f"   Improvement: +{(early_score - baseline_score):.1%}")
        
        # Phase 3: Learn from successful patterns
        print("\n📊 PHASE 3: Learning from Successful Patterns")
        print("-" * 40)
        self.learning_stats["current_phase"] = "pattern_learning"
        
        # Ingest successful conversations
        successful_convs = [c for c in self.conversations 
                          if c["feedback"]["success"] and c["feedback"]["satisfaction_score"] >= 4.0][:10]
        print(f"Ingesting {len(successful_convs)} successful conversations...")
        await self.batch_ingest_conversations(successful_convs, include_procedural=True)
        
        # Check procedural optimization
        if self.agent.total_optimizations > 1:
            print(f"\n🔧 Procedural Memory Updated with successful patterns!")
            print(f"   Total optimizations: {self.agent.total_optimizations}")
        
        # Test after learning successful patterns
        print("\nTesting after learning successful patterns...")
        pattern_results = []
        for scenario in self.test_scenarios[:3]:
            result = self.test_agent_with_scenario(scenario)
            pattern_results.append(result)
        pattern_score = sum(r["evaluation"]["score"] for r in pattern_results) / len(pattern_results) if pattern_results else 0
        print(f"\n📈 Pattern Learning Performance: {pattern_score:.1%}")
        print(f"   Improvement: +{(pattern_score - baseline_score):.1%}")
        
        # Phase 4: Query-specific optimization
        print("\n📊 PHASE 4: Query-Type Specific Optimization")
        print("-" * 40)
        self.learning_stats["current_phase"] = "optimization"
        
        # Force a procedural optimization if we haven't had one yet
        if self.agent.total_optimizations == 0:
            print("\n🔧 Forcing procedural optimization to demonstrate capability...")
            await self.agent.force_optimization()
        
        # Ingest conversations by query type
        for query_type in ["performance_analysis", "exposure_analysis"]:
            type_convs = [c for c in self.conversations 
                         if c["metadata"]["query_type"] == query_type and c["feedback"]["success"]][:5]
            if type_convs:
                print(f"\nIngesting {len(type_convs)} {query_type} conversations...")
                await self.batch_ingest_conversations(type_convs, include_procedural=True)
        
        # Final Summary
        print("\n" + "="*80)
        print("LEARNING PROGRESSION SUMMARY")
        print("="*80)
        print(f"Baseline:         {baseline_score:.1%}")
        print(f"Early Learning:   {early_score:.1%} (+{(early_score-baseline_score):.1%})")
        print(f"Pattern Learning: {pattern_score:.1%} (+{(pattern_score-baseline_score):.1%})")
        
        print("\n📊 MEMORY STATISTICS:")
        all_stats = self.agent.get_all_memory_stats()
        print(f"\n   Episodic Memory:")
        print(f"      • Episodes stored: {self.learning_stats['episodes_stored']}")
        print(f"\n   Semantic Memory:")
        print(f"      • Facts extracted: {self.learning_stats['facts_extracted']}")
        print(f"\n   Procedural Memory:")
        procedural_stats = all_stats.get('procedural', {})
        print(f"      • Total optimizations: {procedural_stats.get('total_optimizations', 0)}")
        print(f"      • Algorithm used: {procedural_stats.get('algorithm', 'N/A')}")
        print(f"      • Avg success rate: {procedural_stats.get('average_success_rate', 0):.1%}")
        print(f"      • Avg satisfaction: {procedural_stats.get('average_satisfaction', 0):.1f}/5.0")
        
        # Show optimization history
        if self.agent.optimization_history:
            print(f"\n🔧 PROCEDURAL OPTIMIZATION HISTORY:")
            for i, opt in enumerate(self.agent.optimization_history[-3:], 1):
                print(f"\n   Optimization {i}:")
                print(f"      Algorithm: {opt['algorithm']}")
                print(f"      Conversations used: {opt['conversations_used']}")
                print(f"      Time: {opt['timestamp'].strftime('%H:%M:%S')}")
        
        # Show extracted patterns being used
        print("\n📋 LEARNED PATTERNS APPLIED:")
        for rule in self.extracted_patterns.get("universal_rules", [])[:3]:
            print(f"   • {rule}")
        
        # Demonstrate memory recall
        print("\n" + "="*80)
        print("MEMORY RECALL DEMONSTRATION")
        print("="*80)
        
        test_queries = [
            "portfolio performance",
            "investment risk",
            "dividend yields"
        ]
        
        for query in test_queries:
            self.demonstrate_memory_recall(query, user_id="3001")
    
    def export_learning_metrics(self, output_file: str = None):
        """Export comprehensive learning metrics for analysis"""
        if output_file is None:
            output_file = os.path.join(self.domain_dir, "learning_metrics.json")
        
        all_stats = self.agent.get_memory_stats()
        procedural_stats = all_stats.get('procedural', {})
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "learning_stats": self.learning_stats,
            "users_processed": len(self.user_memories),
            "avg_satisfaction_by_user": {
                uid: data["satisfaction_avg"] 
                for uid, data in self.user_memories.items()
            },
            "memory_stats": {
                "episodic_semantic": all_stats.get("episodic_semantic", {}),
                "procedural": procedural_stats
            },
            "patterns_learned": len(self.extracted_patterns.get("universal_rules", [])),
            "antipatterns_identified": len(self.extracted_patterns.get("antipatterns", []))
        }
        
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"\n📊 Metrics exported to {output_file}")
        return metrics


async def main():
    """Main demonstration function"""
    print("="*80)
    print("INVESTMENT ADVISOR AGENT - FULL MEMORY INTEGRATION DEMO")
    print("WITH EPISODIC, SEMANTIC, AND PROCEDURAL MEMORY")
    print("="*80)
    
    # Create integration (will automatically use domain directories)
    print("\n🚀 Initializing Investment Advisor Agent with Full Memory...")
    print("   • Episodic Memory: Stores past conversations")
    print("   • Semantic Memory: Extracts and stores facts")
    print("   • Procedural Memory: Learns and adapts strategies")
    
    integration = InvestmentAdvisorIntegration()
    
    # Check if data exists
    if not os.path.exists(integration.data_dir):
        print(f"\n⚠️  Data directory not found: {integration.data_dir}")
        print("Generating data now...")
        from .investment_advisor_data import EnhancedInvestmentAdvisorDataGenerator
        generator = EnhancedInvestmentAdvisorDataGenerator()
        generator.export_realistic_data()
        # Reload data
        integration.load_data()
    
    # Run the progressive learning demonstration
    await integration.run_progressive_learning_demo()
    
    # Export metrics
    integration.export_learning_metrics()
    
    print("\n✅ Demonstration complete!")

if __name__ == "__main__":
    asyncio.run(main())