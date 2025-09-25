"""Test scenarios and data for investment memory hierarchical retrieval demonstration."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from procedural_memory import DomainProcedure

def setup_hierarchy_demo(investment_memory):
    """Setup all test data for hierarchical retrieval demonstration."""
    
    # Community procedures
    investment_memory.community_procedures["moderate_professionals"] = {
        "moderate_prof_strategy": DomainProcedure(
            strategy_pattern="moderate_prof_strategy",
            steps=["Community step 1: Balanced approach", "Community step 2: Diversify"],
            segments=["moderate_professionals"],
            success_rate=0.85,
            scope="community",
            scope_id="moderate_professionals"
        )
    }
    investment_memory.user_communities["user_003"] = ["moderate_professionals"]
    
    # Task procedures
    investment_memory.task_procedures["rebalancing"] = {
        "rebalancing_specialist": DomainProcedure(
            strategy_pattern="rebalancing_specialist",
            steps=["Task step 1: Analyze drift", "Task step 2: Execute trades"],
            segments=["rebalancing"],
            success_rate=0.88,
            scope="task",
            scope_id="rebalancing"
        )
    }
    
    # Global procedures
    investment_memory.global_procedures["general_investment"] = DomainProcedure(
        strategy_pattern="general_investment",
        steps=["Global step 1: Assess situation", "Global step 2: Provide guidance"],
        segments=["general"],
        success_rate=0.75,
        scope="global"
    )

def get_test_cases():
    """Return test cases for demonstrating retrieval hierarchy."""
    return [
        ("user_001", "I want to rebalance", {"age": 35, "risk_tolerance": "moderate"}, 
         "Should retrieve USER scope (user_001 has personalized strategy)"),
        ("user_003", "Need investment advice", {"age": 45, "risk_tolerance": "moderate"},
         "Should retrieve COMMUNITY scope (user_003 in moderate_professionals)"),
        ("user_004", "Time to rebalance my portfolio", {"age": 50, "risk_tolerance": "conservative"},
         "Should retrieve TASK scope (rebalancing task identified)"),
        ("user_005", "General investment question", {"age": 25, "risk_tolerance": "aggressive"},
         "Should retrieve GLOBAL scope (no specific matches)")
    ]

def get_feedback_rounds():
    """Return feedback rounds for performance testing."""
    return [
        {"client_satisfaction": 9, "returns": 12.5, "goals_achieved": True},   # Good
        {"client_satisfaction": 8, "returns": 8.0, "goals_achieved": True},    # Good
        {"client_satisfaction": 4, "returns": -2.0, "goals_achieved": False},  # Bad
        {"client_satisfaction": 7, "returns": 5.0, "goals_achieved": True},    # OK
    ]

def run_performance_feedback(investment_memory, domain_agent, strategy_key="general_investment"):
    """Run performance feedback demonstration with the given strategy."""
    test_strategy = investment_memory.global_procedures[strategy_key]
    initial_success = test_strategy.success_rate
    initial_metrics = test_strategy.domain_metrics.copy()
    
    print(f"\nInitial state of '{strategy_key}' strategy:")
    print(f"  Success rate: {initial_success:.0%}")
    print(f"  Domain metrics: {initial_metrics}")
    print(f"  Adaptations: {len(test_strategy.adaptations)}")
    
    print("\n📈 Applying feedback rounds:")
    for i, feedback in enumerate(get_feedback_rounds(), 1):
        expected_score = domain_agent.calculate_success_score(feedback)
        old_rate = test_strategy.success_rate
        
        result = investment_memory.update_from_performance(
            strategy=strategy_key,
            performance_data=feedback,
            scope="global"
        )
        
        print(f"\nRound {i}: Satisfaction={feedback['client_satisfaction']}, "
              f"Returns={feedback['returns']:+.1f}%")
        print(f"  Success score: {expected_score:.0%}")
        
        if result.get('updated'):
            print(f"  Success rate: {old_rate:.0%} → {result['new_success_rate']:.0%}")
            print(f"  Trend: {result['performance_trend']}")
            
            if 'avg_portfolio_performance' in test_strategy.domain_metrics:
                print(f"  Avg portfolio: {test_strategy.domain_metrics['avg_portfolio_performance']:.1f}%")
        else:
            print(f"  ✗ Strategy not found for update")
    
    # Show adaptation history
    print(f"\n📚 Adaptation History:")
    print(f"  Total adaptations: {len(test_strategy.adaptations)}")
    if test_strategy.adaptations:
        for i, adaptation in enumerate(test_strategy.adaptations[-2:], 1):
            print(f"\n  Adaptation {i}:")
            print(f"    Time: {adaptation['timestamp'][:19]}")
            print(f"    Old rate: {adaptation['old_rate']:.0%}")
            print(f"    New rate: {adaptation['new_rate']:.0%}")
            print(f"    Success score: {adaptation['success_score']:.0%}")

def identify_user_community(conv):
    """Identify user's community from metadata."""
    metadata = conv.get('metadata', {})
    age = metadata.get('client_age', 35)
    risk = metadata.get('risk_tolerance', 'moderate')
    
    if age >= 60 and risk == "conservative":
        community = "conservative_retirees"
    elif age <= 40 and risk == "aggressive":
        community = "aggressive_millennials"
    else:
        community = "moderate_professionals"
    
    return [community], {"age": age, "risk_tolerance": risk}

def process_baseline_conversations(full_agent, conversations, num_baseline=30):
    """Process baseline conversations and return learning summary."""
    from collections import defaultdict
    
    baseline_conversations = conversations[:num_baseline]
    learning_summary = defaultdict(int)
    processed = defaultdict(set)
    facts_extracted = 0
    extraction_errors = 0
    
    for i, conv in enumerate(baseline_conversations):
        user_id = str(conv['user_id'])
        messages = [(msg['role'], msg['content']) for msg in conv['messages']]
        
        # Store episodic memory
        full_agent.store_episodic_memory(
            conversation_id=conv['conversation_id'],
            messages=messages,
            summary=f"Investment discussion - satisfaction: {conv['feedback']['satisfaction_score']}/5.0"
        )
        
        # Extract semantic facts
        try:
            facts = full_agent.extract_semantic_facts(messages)
            if facts:
                facts_extracted += full_agent.store_semantic_facts(facts, user_id=user_id)
        except Exception:
            extraction_errors += 1
        
        # Learn from successful conversations
        if conv['feedback']['success'] and conv['feedback']['satisfaction_score'] >= 4.0:
            communities, user_profile = identify_user_community(conv)
            
            # Assign user to community
            for community in communities:
                if user_id not in full_agent.procedural_memory.user_communities:
                    full_agent.procedural_memory.user_communities[user_id] = []
                if community not in full_agent.procedural_memory.user_communities[user_id]:
                    full_agent.procedural_memory.user_communities[user_id].append(community)
                    full_agent.procedural_memory.community_members[community].add(user_id)
                    processed['communities'].add(community)
            
            # Learn patterns
            learning_result = full_agent.procedural_memory.learn_from_interaction(
                query=conv['messages'][0]['content'],
                interaction_data={
                    'messages': conv['messages'],
                    'success': True,
                    'client_satisfaction': conv['feedback']['satisfaction_score'],
                    'query_type': conv['metadata'].get('query_type', 'unknown')
                },
                user_id=user_id,
                user_profile=user_profile
            )
            
            # Track learning
            for key in ['global_learned', 'user_learned', 'community_learned', 'task_learned']:
                if learning_result.get(key):
                    scope = key.replace('_learned', '')
                    learning_summary[scope] += 1
                    if scope == 'user':
                        processed['users'].add(user_id)
                    elif scope == 'task':
                        task = full_agent.procedural_memory._identify_task_type(conv['messages'][0]['content'])
                        processed['tasks'].add(task)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(baseline_conversations)}...")
    
    return {
        'baseline_count': len(baseline_conversations),
        'facts_extracted': facts_extracted,
        'extraction_errors': extraction_errors,
        'learning_summary': dict(learning_summary),
        'processed': {k: len(v) for k, v in processed.items()}
    }

def get_test_queries():
    """Return test queries for agent testing."""
    return [
        ("I want to invest in sustainable companies", "3001", {"age": 35, "risk_tolerance": "moderate"}),
        ("Time to rebalance my portfolio?", "3002", {"age": 65, "risk_tolerance": "conservative"})
    ]

def process_performance_feedback(full_agent, conversations, start_idx=30, end_idx=40):
    """Process performance feedback from conversations and trigger adaptations."""
    evolution_convs = conversations[start_idx:end_idx]
    adaptations = 0
    adaptation_details = []
    
    for conv in evolution_convs:
        user_id = str(conv['user_id'])
        
        # Get user profile from conversation metadata
        metadata = conv.get('metadata', {})
        profile = {
            "age": metadata.get('client_age', 35),
            "risk_tolerance": metadata.get('risk_tolerance', 'moderate')
        }
        
        strategy = full_agent.procedural_memory.get_investment_strategy(
            conv['messages'][0]['content'],
            profile,
            user_id
        )
        
        if strategy:
            # Update performance based on actual conversation feedback
            result = full_agent.procedural_memory.update_from_performance(
                strategy['strategy'],
                {
                    "client_satisfaction": conv['feedback']['satisfaction_score'],
                    "returns": 8.5 if conv['feedback']['success'] else -2.0,
                    "goals_achieved": conv['feedback']['success']
                },
                strategy.get('scope', 'global'),
                strategy.get('scope_id')
            )
            if result.get('updated'):
                adaptations += 1
                adaptation_details.append({
                    'strategy': result['updated'][:20] + "...",
                    'new_rate': result['new_success_rate']
                })
    
    return adaptations, adaptation_details

def test_agent_with_queries(full_agent):
    """Test the agent with predefined queries and return results."""
    results = []
    for query, user_id, profile in get_test_queries():
        response = full_agent.process_message(query, user_id=user_id)
        strategy = full_agent.procedural_memory.get_investment_strategy(query, profile, user_id)
        
        results.append({
            'query': query,
            'response': response[:150] + "...",
            'strategy': strategy
        })
    
    return results

def process_remaining_conversations(full_agent, conversations, start_idx=50, end_idx=100):
    """Process remaining conversations for learning progression."""
    remaining = conversations[start_idx:end_idx]
    learned = {"global": 0, "user": 0, "community": 0, "task": 0}
    
    for i, conv in enumerate(remaining):
        if i % 10 == 0 and i > 0:
            print(f"   Processed {i}/{len(remaining)}...")
        
        if conv['feedback']['success'] and conv['feedback']['satisfaction_score'] >= 4.5:
            user_id = str(conv['user_id'])
            metadata = conv.get('metadata', {})
            user_profile = {
                "age": metadata.get('client_age', 35),
                "risk_tolerance": metadata.get('risk_tolerance', 'moderate')
            }
            
            result = full_agent.procedural_memory.learn_from_interaction(
                conv['messages'][0]['content'],
                {'messages': conv['messages'], 'success': True, 
                 'client_satisfaction': conv['feedback']['satisfaction_score'],
                 'query_type': metadata.get('query_type', 'unknown')},
                user_id=user_id,
                user_profile=user_profile
            )
            for key in learned:
                if result.get(f'{key}_learned'):
                    learned[key] += 1
    
    return len(remaining), learned

def get_test_users():
    """Return test users for hierarchical retrieval demonstration."""
    return [
        ("3001", "Experienced user", {"age": 35, "risk_tolerance": "moderate"}),
        ("3010", "Conservative user", {"age": 65, "risk_tolerance": "conservative"}),
        ("new_user_9999", "New user", {"age": 30, "risk_tolerance": "aggressive"})
    ]

def get_key_achievements():
    """Return list of key achievements for the demonstration."""
    return [
        "User-level personalization: Power users get customized strategies",
        "Community learning: New users benefit from their peer group's patterns",
        "Task specialization: Different strategies for different investment tasks",
        "Hierarchical fallback: User → Community → Task → Global",
        "Continuous adaptation: Strategies improve based on performance"
    ]

def test_hierarchical_retrieval(full_agent, query="I'm worried about market volatility. Should I move to safer investments?"):
    """Test hierarchical retrieval with different user types."""
    results = []
    for user_id, desc, profile in get_test_users():
        strategy = full_agent.procedural_memory.get_investment_strategy(query, profile, user_id)
        results.append({
            'user_id': user_id,
            'description': desc,
            'profile': profile,
            'strategy': strategy
        })
    return results

def create_test_agent_for_algorithm(baseline_agent, test_domain_agent, algorithm_name="prompt_memory"):
    """Create a test agent with copied baseline state for algorithm testing."""
    import os
    import shutil
    from coala_agent import CoALAAgent
    
    # Use a separate directory for this test
    test_dir = os.path.join(test_domain_agent.domain_dir, f"{algorithm_name}_test")
    os.makedirs(test_dir, exist_ok=True)
    
    # Copy the baseline agent's state if it exists
    baseline_dir = baseline_agent.domain_agent.memory_dir
    if os.path.exists(baseline_dir):
        shutil.copytree(baseline_dir, test_dir, dirs_exist_ok=True)
        print(f"  Copied baseline state to {test_dir}")
    
    # Create new agent
    test_agent = CoALAAgent(
        domain_agent=test_domain_agent,
        model_name="gpt-4o-mini",
        temperature=0,
        persist_directory=test_dir
    )
    
    return test_agent, test_dir

def apply_prompt_memory_rules(agent, rules):
    """Apply learned rules from prompt_memory optimization to agent."""
    # No need to import here since it's already imported at the top
    applied_count = 0
    for rule in rules:
        pattern_name = f"prompt_memory_rule_{len(agent.procedural_memory.global_procedures)}"
        agent.procedural_memory.global_procedures[pattern_name] = DomainProcedure(
            strategy_pattern=pattern_name,
            steps=[rule['rule']],
            segments=["general"],
            success_rate=0.85,
            scope="global"
        )
        applied_count += 1
    
    return applied_count

def test_optimizer_efficiency(optimizer, test_conversations, num_runs=3):
    """Test the efficiency of an optimizer with multiple runs."""
    from datetime import datetime
    
    test_start = datetime.now()
    for i in range(num_runs):
        batch = test_conversations[i*5:(i+1)*5] if i*5 < len(test_conversations) else test_conversations[:5]
        _ = optimizer.optimize(batch, {"avg_success_rate": 0.7})
    
    total_time = (datetime.now() - test_start).total_seconds()
    avg_time = total_time / num_runs
    
    return {
        'avg_time': avg_time,
        'efficiency': 1/avg_time if avg_time > 0 else 0,
        'num_runs': num_runs
    }

def run_prompt_memory_test(baseline_agent, test_domain_agent, prompt_optimizer, test_conversations):
    """Run complete prompt_memory optimization test."""
    from datetime import datetime
    
    # Create test agent
    print("🔬 Creating test agent for prompt_memory algorithm...")
    prompt_memory_agent, test_dir = create_test_agent_for_algorithm(
        baseline_agent, test_domain_agent, "prompt_memory"
    )
    
    print("✅ Prompt_memory optimizer initialized")
    print("  Characteristics: Single-pass, minimal overhead, rapid iteration")
    
    # Run optimization
    print("\n🚀 Running prompt_memory optimization...")
    
    if not test_conversations:
        print("❌ No test conversations available. Run Lab 18-1 first.")
        return None
    
    start_time = datetime.now()
    prompt_result = prompt_optimizer.optimize(
        test_conversations[:10],
        baseline_agent.procedural_memory.get_stats()
    )
    optimization_time = (datetime.now() - start_time).total_seconds()
    
    # Apply learned rules
    applied_count = apply_prompt_memory_rules(prompt_memory_agent, prompt_result['rules'])
    
    # Test efficiency
    efficiency_results = test_optimizer_efficiency(prompt_optimizer, test_conversations)
    
    return {
        'agent': prompt_memory_agent,
        'optimization_time': optimization_time,
        'prompt_result': prompt_result,
        'applied_rules': applied_count,
        'efficiency': efficiency_results
    }


def apply_gradient_improvements(agent, improvements):
    """Apply gradient improvements to agent's procedural memory."""
    # No need to import here since it's already imported at the top
    applied_count = 0
    for improvement in improvements:
        pattern_name = f"gradient_improvement_{len(agent.procedural_memory.global_procedures)}"
        agent.procedural_memory.global_procedures[pattern_name] = DomainProcedure(
            strategy_pattern=pattern_name,
            steps=[improvement.get('rule', 'Improvement rule')],
            segments=["general"],
            success_rate=0.85,
            scope="global"
        )
        applied_count += 1
    
    return applied_count

def test_gradient_issue_detection(gradient_optimizer):
    """Test gradient's ability to identify specific issues."""
    import json
    
    test_critique = gradient_optimizer.critique_prompt | gradient_optimizer.llm | gradient_optimizer.parser
    
    # Test with a problematic conversation
    test_conv = [{
        "query": "Am I losing money?",
        "response": "Your portfolio is fine.",
        "success": False,
        "satisfaction": 2.0,
        "signals": []
    }]
    
    critique_only = test_critique.invoke({
        "conversations": json.dumps(test_conv, indent=2)
    })
    
    return critique_only

def run_gradient_test(baseline_agent, test_domain_agent, gradient_optimizer, test_conversations):
    """Run complete gradient optimization test."""
    from datetime import datetime
    
    # Create test agent
    print("🔬 Creating test agent for gradient algorithm...")
    gradient_agent, test_dir = create_test_agent_for_algorithm(
        baseline_agent, test_domain_agent, "gradient"
    )
    
    print("✅ Gradient optimizer initialized")
    print("  Characteristics: Two-phase, separated critique/proposal, focused improvements")
    
    if not test_conversations:
        print("❌ No test conversations available. Run Lab 18-1 first.")
        return None
    
    # Run optimization
    print("\n🚀 Running gradient optimization...")
    
    start_time = datetime.now()
    gradient_result = gradient_optimizer.optimize(
        test_conversations[:10],
        baseline_agent.procedural_memory.get_stats()
    )
    optimization_time = (datetime.now() - start_time).total_seconds()
    
    # Apply improvements
    applied_count = apply_gradient_improvements(gradient_agent, gradient_result['improvements'])
    
    # Test issue detection
    issue_detection = test_gradient_issue_detection(gradient_optimizer)
    
    return {
        'agent': gradient_agent,
        'optimization_time': optimization_time,
        'gradient_result': gradient_result,
        'applied_improvements': applied_count,
        'issue_detection': issue_detection
    }

def apply_metaprompt_rules(agent, comprehensive_rules):
    """Apply comprehensive rules from metaprompt optimization to agent."""
    # No need to import here since it's already imported at the top
    applied_count = 0
    for rule in comprehensive_rules:
        pattern_name = f"metaprompt_rule_{len(agent.procedural_memory.global_procedures)}"
        agent.procedural_memory.global_procedures[pattern_name] = DomainProcedure(
            strategy_pattern=pattern_name,
            steps=[rule.get('rule', 'Comprehensive rule')],
            segments=["general"],
            success_rate=rule.get('confidence', 0.85),
            scope="global"
        )
        applied_count += 1
    
    return applied_count

def test_metaprompt_reflection(metaprompt_optimizer):
    """Test metaprompt's reflection capability with pattern discovery."""
    test_conversations_with_pattern = [
        {
            "messages": [
                {"role": "user", "content": "Quick question about my portfolio"},
                {"role": "assistant", "content": "Your portfolio is up 2.3% this month."}
            ],
            "feedback": {"success": False, "satisfaction_score": 2.5},
            "behavioral_signals": {"provided_specific_data": True, "asked_clarification": False},
            "metadata": {"query_type": "performance_analysis"}
        },
        {
            "messages": [
                {"role": "user", "content": "I'm worried about the market"},
                {"role": "assistant", "content": "I understand your concern. Let's review your defensive positions..."}
            ],
            "feedback": {"success": True, "satisfaction_score": 4.8},
            "behavioral_signals": {"showed_empathy": True, "provided_specific_data": False},
            "metadata": {"query_type": "risk_assessment"}
        }
    ]
    
    reflection_test = metaprompt_optimizer.optimize(
        test_conversations_with_pattern,
        {"avg_success_rate": 0.5}
    )
    
    return reflection_test

def run_metaprompt_test(baseline_agent, test_domain_agent, metaprompt_optimizer, test_conversations):
    """Run complete metaprompt optimization test."""
    from datetime import datetime
    
    # Create test agent
    print("🔬 Creating test agent for metaprompt algorithm...")
    metaprompt_agent, test_dir = create_test_agent_for_algorithm(
        baseline_agent, test_domain_agent, "metaprompt"
    )
    
    print("✅ Metaprompt optimizer initialized")
    print("  Characteristics: Multi-stage, reflection-based, discovers hidden patterns")
    
    if not test_conversations:
        print("❌ No test conversations available. Run Lab 18-1 first.")
        return None
    
    # Run optimization
    print("\n🚀 Running metaprompt optimization...")
    
    start_time = datetime.now()
    metaprompt_result = metaprompt_optimizer.optimize(
        test_conversations[:10],
        baseline_agent.procedural_memory.get_stats()
    )
    optimization_time = (datetime.now() - start_time).total_seconds()
    
    # Apply comprehensive rules
    applied_count = apply_metaprompt_rules(metaprompt_agent, metaprompt_result['comprehensive_rules'])
    
    # Test reflection capability
    reflection_test = test_metaprompt_reflection(metaprompt_optimizer)
    
    return {
        'agent': metaprompt_agent,
        'optimization_time': optimization_time,
        'metaprompt_result': metaprompt_result,
        'applied_rules': applied_count,
        'reflection_test': reflection_test
    }

def get_agent_test_queries():
    """Return test queries for evaluating optimized agents."""
    return [
        {
            "query": "I'm worried about inflation eating my savings. What should I do?",
            "context": "Risk-averse investor concern"
        },
        {
            "query": "Should I sell everything and go to cash?",
            "context": "Panic response to volatility"
        },
        {
            "query": "Explain P/E ratios in simple terms",
            "context": "Educational request from beginner"
        }
    ]

def evaluate_response_qualities(response):
    """Evaluate response for key qualities."""
    response_lower = response.lower()
    return {
        "has_empathy": any(word in response_lower for word in ["understand", "concern", "worry", "appreciate"]),
        "has_specifics": "$" in response or "%" in response,
        "has_education": any(phrase in response_lower for phrase in ["this means", "in other words", "essentially", "simply put"]),
        "response_length": len(response)
    }

def test_agents_with_queries(agents_to_test):
    """Test multiple agents with standard queries."""
    test_queries = get_agent_test_queries()
    results_by_algo = {}
    
    for algo_name, agent in agents_to_test:
        print(f"\n📋 {algo_name.upper()} Agent Responses:")
        print("-" * 50)
        
        algo_results = []
        for test in test_queries:
            try:
                response = agent.process_message(test["query"], user_id=f"test_{algo_name}")
                result = evaluate_response_qualities(response)
                result["query"] = test["query"][:50]
                algo_results.append(result)
                
                print(f"\nQuery: {test['query'][:50]}...")
                print(f"Response: {response[:200]}...")
                qualities = []
                if result["has_empathy"]: qualities.append("[Empathy]")
                if result["has_specifics"]: qualities.append("[Specifics]")
                if result["has_education"]: qualities.append("[Education]")
                print(f"Qualities: {''.join(qualities)}")
                
            except Exception as e:
                print(f"Error processing query: {e}")
                algo_results.append({
                    "query": test["query"][:50],
                    "has_empathy": False,
                    "has_specifics": False,
                    "has_education": False,
                    "response_length": 0
                })
        
        results_by_algo[algo_name] = algo_results
    
    return results_by_algo

def test_response_consistency(agents_to_test, test_query="What's my risk level?", num_runs=3):
    """Test response consistency across multiple runs."""
    import numpy as np
    
    consistency_test = {}
    
    for algo_name, agent in agents_to_test:
        responses = []
        for i in range(num_runs):
            try:
                response = agent.process_message(test_query, user_id=f"consistency_test_{i}")
                responses.append(len(response))
            except:
                responses.append(0)
        
        valid_responses = [r for r in responses if r > 0]
        if valid_responses:
            consistency_test[algo_name] = {
                "mean_length": np.mean(valid_responses),
                "std_length": np.std(valid_responses)
            }
        else:
            consistency_test[algo_name] = {
                "mean_length": 0,
                "std_length": 0
            }
    
    return consistency_test

def compare_agent_performance(results_by_algo):
    """Generate performance comparison statistics."""
    stats = {}
    
    for algo_name, results in results_by_algo.items():
        stats[algo_name] = {
            "empathy_count": sum(1 for r in results if r["has_empathy"]),
            "specifics_count": sum(1 for r in results if r["has_specifics"]),
            "education_count": sum(1 for r in results if r["has_education"]),
            "avg_length": sum(r["response_length"] for r in results) / len(results) if results else 0,
            "total_queries": len(results)
        }
    
    # Calculate improvements over baseline
    if "baseline" in stats:
        baseline_empathy = stats["baseline"]["empathy_count"]
        for algo_name in stats:
            if algo_name != "baseline":
                stats[algo_name]["empathy_improvement"] = stats[algo_name]["empathy_count"] > baseline_empathy
    
    return stats