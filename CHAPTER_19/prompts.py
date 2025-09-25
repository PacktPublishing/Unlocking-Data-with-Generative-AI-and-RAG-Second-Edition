# prompts.py - general prompts for any domain

# ============== GENERIC LEARNING PROMPTS ==============

GLOBAL_LEARNING_PROMPT = """
Analyze this {domain} interaction for UNIVERSAL patterns.

Query: {task}
Interaction: {execution_data}
Existing Strategies: {existing}

Extract a GENERAL strategy that would work for ANY user with similar query.

Output JSON with EXACT format:
{{
    "procedure": {{
        "strategy_pattern": "string describing the pattern",
        "steps": ["step 1 as string", "step 2 as string"],
        "segments": ["segment1 as string", "segment2 as string"],
        "confidence": 0.85,
        "domain_metrics": {{"metric_name": 0.0}}
    }}
}}

CRITICAL: 
- segments must be array of STRINGS only
- domain_metrics must be dict with STRING keys and FLOAT values only
"""

USER_LEARNING_PROMPT = """
Analyze this SPECIFIC USER's preferences.

User: {user_id}
Query: {task}
Interaction: {execution_data}

Extract patterns UNIQUE to this user.

Output JSON with EXACT format:
{{
    "procedure": {{
        "strategy_pattern": "user preference description",
        "steps": ["step 1 string", "step 2 string"],
        "segments": ["preference1", "preference2"],
        "confidence": 0.9,
        "domain_metrics": {{}}
    }}
}}
"""

COMMUNITY_LEARNING_PROMPT = """
Analyze patterns for {community_segment} community.

Community: {community_segment}
Query: {task}
Interaction: {execution_data}

Extract patterns for this USER GROUP.

Output JSON with EXACT format:
{{
    "procedure": {{
        "strategy_pattern": "community approach",
        "steps": ["step 1", "step 2"],
        "segments": ["{community_segment}"],
        "confidence": 0.87,
        "domain_metrics": {{}}
    }}
}}
"""

TASK_LEARNING_PROMPT = """
Analyze this {task_type} task pattern.

Task Type: {task_type}
Query: {task}
Interaction: {execution_data}

Output JSON with EXACT format:
{{
    "procedure": {{
        "strategy_pattern": "{task_type} strategy",
        "steps": ["step 1", "step 2"],
        "segments": ["{task_type}"],
        "confidence": 0.88,
        "domain_metrics": {{}}
    }}
}}
"""

GRADIENT_CRITIQUE_TEMPLATE = """
Analyze these conversations to identify problems and successes.

CONVERSATIONS:
{conversations}

CURRENT PERFORMANCE:
{current_performance}

Identify:
1. What behaviors led to failures?
2. What patterns correlate with satisfaction?
3. Critical gaps in responses?

Output JSON:
{{
    "failures": [list of issues],
    "successes": [list of what worked],
    "gaps": [list of missing capabilities]
}}
"""

GRADIENT_PROPOSAL_TEMPLATE = """
Based on this critique, propose improvements.

CRITIQUE:
{critique}

Generate specific procedural rules to address issues.

Output JSON:
{{
    "procedural_rules": [
        {{"rule": "specific instruction", "rationale": "why"}}
    ],
    "optimization_summary": "overall strategy"
}}
"""

METAPROMPT_SURFACE_TEMPLATE = """
Initial analysis of conversations.

{conversations}

Identify obvious patterns.

Output JSON:
{{
    "patterns_found": ["pattern1", "pattern2"],
    "initial_observations": ["observation1", "observation2"]
}}
"""

METAPROMPT_DEEP_TEMPLATE = """
Deeper analysis based on surface patterns.

SURFACE ANALYSIS:
{surface_analysis}

CONVERSATIONS:
{conversations}

Find hidden patterns and causal relationships.

Output JSON:
{{
    "hidden_patterns": ["pattern1", "pattern2"],
    "causal_relationships": ["cause->effect"]
}}
"""

METAPROMPT_SYNTHESIS_TEMPLATE = """
Synthesize all findings into rules.

SURFACE:
{surface_analysis}

DEEP:
{deep_analysis}

Create comprehensive procedural rules.

Output JSON:
{{
    "procedural_rules": [
        {{"rule": "instruction", "confidence": 0.9}}
    ],
    "optimization_summary": "synthesis"
}}
"""

# ============== GENERIC SEMANTIC EXTRACTION ==============

SEMANTIC_EXTRACTION_PROMPT = """
Analyze this conversation and extract important facts.

Conversation: {conversation}

Extract facts in JSON format:
{{"facts": [{{"subject": "...", "predicate": "...", "object": "...", 
              "confidence": 0.0-1.0, "source": "user or assistant"}}]}}

Only extract clear, specific facts. Output valid JSON only.
"""

# ============== GENERIC PROMPTS REGISTRY ==============

GENERAL_PROMPTS = {
    "global": GLOBAL_LEARNING_PROMPT,
    "user": USER_LEARNING_PROMPT,
    "community": COMMUNITY_LEARNING_PROMPT,
    "task": TASK_LEARNING_PROMPT,
    "semantic_extraction": SEMANTIC_EXTRACTION_PROMPT
}

def get_prompts_for_domain(domain: str) -> dict:
    """Get all prompts for a specific domain - domains should override this"""
    return GENERAL_PROMPTS