# prompts.py
"""
LangMem instruction templates for procedural memory extraction.
These are passed to create_memory_manager as extraction instructions.
"""

GLOBAL_EXTRACTION_INSTRUCTIONS = """
Extract universal procedural strategies from this interaction that would help any user with similar queries.
Focus on identifying reusable patterns, step-by-step approaches, and best practices.
"""

USER_EXTRACTION_INSTRUCTIONS = """
Extract user-specific preferences and personalized approaches from this interaction.
Focus on individual communication style, risk preferences, and unique requirements.
"""

COMMUNITY_EXTRACTION_INSTRUCTIONS = """
Extract patterns that apply to users in this community segment.
Focus on shared characteristics, common needs, and group-level strategies.
"""

TASK_EXTRACTION_INSTRUCTIONS = """
Extract task-specific procedures and workflows from this interaction.
Focus on the steps, decision points, and success criteria for this type of task.
"""

SEMANTIC_EXTRACTION_PROMPT = """
Analyze this conversation and extract important facts.

Conversation: {conversation}

Extract facts in JSON format:
{{"facts": [{{"subject": "...", "predicate": "...", "object": "...", 
              "confidence": 0.0-1.0, "source": "user or assistant"}}]}}

Only extract clear, specific facts. Output valid JSON only.
"""

RESPONSE_PROMPT = """
You are a helpful assistant with access to memory systems.

{semantic_context}

{episodic_context}

{procedural_context}

Current conversation:
{messages}

Respond helpfully using your memories when relevant.
"""

PROMPTS = {
    "global": GLOBAL_EXTRACTION_INSTRUCTIONS,
    "user": USER_EXTRACTION_INSTRUCTIONS,
    "community": COMMUNITY_EXTRACTION_INSTRUCTIONS,
    "task": TASK_EXTRACTION_INSTRUCTIONS,
    "semantic": SEMANTIC_EXTRACTION_PROMPT,
    "response": RESPONSE_PROMPT
}


def get_prompts() -> dict:
    """Return all prompts"""
    return PROMPTS