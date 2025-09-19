# domain_investment/investment_advisor_prompts.py
"""Investment advisor specific prompts"""

# ============== INVESTMENT LEARNING PROMPTS ==============

INVESTMENT_GLOBAL_PROMPT = """
Analyze this investment advisory interaction.

Query: {task}
Interaction: {execution_data}
Existing: {existing}

Output JSON with EXACT format:
{{
    "procedure": {{
        "strategy_pattern": "investment pattern name",
        "steps": ["step 1", "step 2", "step 3"],
        "segments": ["moderate_risk", "millennials"],
        "confidence": 0.85,
        "domain_metrics": {{"avg_portfolio_performance": 8.5}}
    }}
}}

CRITICAL FORMAT RULES:
- segments: array of simple strings like ["aggressive", "retirees"]
- domain_metrics: dict with float values like {{"returns": 10.5, "sharpe": 1.2}}
- NO complex objects, NO descriptions in arrays
"""

INVESTMENT_USER_PROMPT = """
Analyze user {user_id}'s investment preferences.

Query: {task}
Interaction: {execution_data}

Output JSON:
{{
    "procedure": {{
        "strategy_pattern": "user investment preference",
        "steps": ["personalized step 1", "personalized step 2"],
        "segments": ["user_preference"],
        "confidence": 0.9,
        "domain_metrics": {{}}
    }}
}}

segments must be simple strings, domain_metrics must have numeric values only
"""

INVESTMENT_COMMUNITY_PROMPT = """
Analyze {community_segment} investment patterns.

Interaction: {execution_data}

Output JSON:
{{
    "procedure": {{
        "strategy_pattern": "{community_segment} approach",
        "steps": ["community step 1", "community step 2"],
        "segments": ["{community_segment}"],
        "confidence": 0.87,
        "domain_metrics": {{"avg_returns": 0.0}}
    }}
}}
"""

INVESTMENT_TASK_PROMPT = """
Analyze this {task_type} investment task.

Task Type: {task_type}
Query: {task}
Interaction: {execution_data}

Output JSON:
{{
    "procedure": {{
        "strategy_pattern": "{task_type} investment strategy",
        "steps": ["task step 1", "task step 2"],
        "segments": ["{task_type}"],
        "confidence": 0.88,
        "domain_metrics": {{}}
    }}
}}
"""

INVESTMENT_RESPONSE_PROMPT = """
You are an experienced investment advisor with access to client history and proven strategies.

{semantic_context}

{episodic_context}

{procedural_context}

Current conversation:
{messages}

Provide personalized investment advice using your knowledge of the client and proven strategies.
Be specific and actionable. If you have a recommended strategy, follow its steps.
"""

INVESTMENT_SEMANTIC_EXTRACTION_PROMPT = """
Analyze this investment conversation and extract important facts.
Focus on: client information, investment preferences, risk tolerance, goals.

Conversation: {conversation}

Extract facts in JSON format:
{{"facts": [{{"subject": "...", "predicate": "...", "object": "...", 
              "confidence": 0.0-1.0, "source": "user or assistant"}}]}}

Only extract clear facts. Output valid JSON only.
"""

# ============== OPTIMIZATION ALGORITHM PROMPTS ==============

PROMPT_MEMORY_OPTIMIZATION = """
Analyze these investment advisory conversations and extract procedural improvements in a single pass.

CONVERSATIONS:
{conversations}

CURRENT PERFORMANCE:
{current_performance}

Extract patterns and generate rules in one analysis:
1. Identify what makes conversations successful
2. Note common failure points
3. Generate specific procedural rules

Output JSON:
{{
    "patterns_found": [
        {{"pattern": "description", "frequency": "how often", "impact": "success correlation"}}
    ],
    "procedural_rules": [
        {{"rule": "specific instruction", "condition": "when to apply", "priority": "high/medium/low"}}
    ],
    "optimization_summary": "overall learning from this batch"
}}
"""

GRADIENT_CRITIQUE = """
Critically analyze these investment advisory conversations to identify specific problems and successes.

CONVERSATIONS:
{conversations}

Provide an objective critique:
1. What specific behaviors led to failures?
2. What patterns correlate with high satisfaction?
3. What are the most critical gaps in current responses?

Be specific and analytical. Focus on actionable observations.

Output JSON:
{{
    "failures": [
        {{"issue": "specific problem", "frequency": "how often", "impact": "severity"}}
    ],
    "successes": [
        {{"behavior": "what worked", "correlation": "success rate", "context": "when it works"}}
    ],
    "critical_gaps": [
        {{"gap": "what's missing", "importance": "why it matters"}}
    ]
}}
"""

GRADIENT_PROPOSAL = """
Based on this critique, generate specific improvements for investment advisory behavior.

CRITIQUE:
{critique}

Generate targeted improvements that address the identified issues:
1. Create specific rules to prevent failures
2. Codify successful behaviors
3. Fill critical gaps

Output JSON:
{{
    "improvements": [
        {{
            "target_issue": "which problem this solves",
            "rule": "specific behavioral instruction",
            "expected_impact": "how this improves performance",
            "implementation": "when and how to apply"
        }}
    ],
    "optimization_strategy": "overall approach to improvement"
}}
"""

METAPROMPT_SURFACE = """
Perform initial analysis of investment advisory conversations.

CONVERSATIONS:
{conversations}

Identify obvious patterns and immediate observations.
Output JSON:
{{
    "surface_patterns": ["pattern1", "pattern2"],
    "initial_correlations": {{"behavior": "outcome"}},
    "obvious_issues": ["issue1", "issue2"]
}}
"""

METAPROMPT_DEEP = """
Reflect on this surface analysis and dig deeper.

SURFACE ANALYSIS:
{surface_analysis}

CONVERSATIONS:
{conversations}

Question your assumptions. What patterns are you missing?
What counterintuitive relationships exist?

Output JSON:
{{
    "hidden_patterns": [
        {{"pattern": "description", "evidence": "supporting data", "confidence": 0.0-1.0}}
    ],
    "causal_relationships": [
        {{"cause": "behavior", "effect": "outcome", "mechanism": "why this happens"}}
    ],
    "counterintuitive_findings": [
        {{"finding": "unexpected pattern", "explanation": "why it works"}}
    ]
}}
"""

METAPROMPT_SYNTHESIS = """
Synthesize all findings into comprehensive procedural rules.

SURFACE ANALYSIS:
{surface_analysis}

DEEP ANALYSIS:
{deep_analysis}

Create rules with examples and edge cases.
Output JSON:
{{
    "comprehensive_rules": [
        {{
            "rule": "specific instruction",
            "rationale": "why this works based on analysis",
            "example_application": "concrete example",
            "edge_cases": ["exception1", "exception2"],
            "confidence": 0.0-1.0
        }}
    ],
    "meta_insights": "overarching principle discovered",
    "implementation_priority": ["highest_impact_rule", "second_priority"]
}}
"""

# ============== OPTIMIZATION PROMPTS REGISTRY ==============

OPTIMIZATION_PROMPTS = {
    "prompt_memory": PROMPT_MEMORY_OPTIMIZATION,
    "gradient_critique": GRADIENT_CRITIQUE,
    "gradient_proposal": GRADIENT_PROPOSAL,
    "metaprompt_surface": METAPROMPT_SURFACE,
    "metaprompt_deep": METAPROMPT_DEEP,
    "metaprompt_synthesis": METAPROMPT_SYNTHESIS
}

def get_optimization_prompt(algorithm: str) -> str:
    """Get optimization prompt for a specific algorithm"""
    return OPTIMIZATION_PROMPTS.get(algorithm, "")

# ============== DOMAIN PROMPTS REGISTRY ==============

INVESTMENT_PROMPTS = {
    "global": INVESTMENT_GLOBAL_PROMPT,
    "user": INVESTMENT_USER_PROMPT,
    "community": INVESTMENT_COMMUNITY_PROMPT,
    "task": INVESTMENT_TASK_PROMPT,  # Use the investment-specific task prompt
    "response": INVESTMENT_RESPONSE_PROMPT,
    "semantic": INVESTMENT_SEMANTIC_EXTRACTION_PROMPT,
    "semantic_extraction": INVESTMENT_SEMANTIC_EXTRACTION_PROMPT  # Alias
}

# If TASK_LEARNING_PROMPT is generic, import it from parent
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from prompts import TASK_LEARNING_PROMPT
except ImportError:
    # Define investment-specific task prompt if generic not available
    TASK_LEARNING_PROMPT = """
    Analyze this {task_type} investment task.
    
    Task Type: {task_type}
    Query: {task}
    Interaction: {execution_data}
    
    Output JSON:
    {{
        "procedure": {{
            "strategy_pattern": "{task_type} investment strategy",
            "steps": ["task step 1", "task step 2"],
            "segments": ["{task_type}"],
            "confidence": 0.88,
            "domain_metrics": {{}}
        }}
    }}
    """
    INVESTMENT_PROMPTS["task"] = TASK_LEARNING_PROMPT