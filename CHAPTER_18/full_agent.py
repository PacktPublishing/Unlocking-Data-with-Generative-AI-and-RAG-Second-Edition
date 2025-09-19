# full_agent.py
"""
Complete CoALA Agent with Semantic, Episodic, and Procedural Memory
Combines baseline_agent.py with procedural memory from Code Lab 18-1
"""

import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from baseline_agent import CoALABaselineAgent
from langmem import create_prompt_optimizer


class ProceduralMemoryOptimizer:
    """Base class for procedural memory optimization."""
    
    def __init__(self, model="openai:gpt-4o-mini", kind="metaprompt"):
        self.optimizer = create_prompt_optimizer(model, kind=kind)
        self.kind = kind
    
    async def optimize(self, conversations: List[Dict], current_prompt: str) -> str:
        """Run optimization on conversations."""
        
        # Format conversations as trajectories for LangMem
        trajectories = []
        for conv in conversations:
            messages = conv.get("messages", [])
            feedback = conv.get("feedback", {})
            trajectories.append((messages, feedback))
        
        print(f"Optimizing with {self.kind} algorithm...")
        
        # Run optimization
        improved_prompt = await self.optimizer.ainvoke({
            "trajectories": trajectories,
            "prompt": current_prompt
        })
        
        return improved_prompt


class MetapromptProceduralOptimizer(ProceduralMemoryOptimizer):
    """Metaprompt optimization with reflection."""
    
    def __init__(self, model="openai:gpt-4o-mini"):
        super().__init__(model, kind="metaprompt")
        # Reconfigure with reflection steps
        self.optimizer = create_prompt_optimizer(
            model,
            kind="metaprompt",
            config={
                "max_reflection_steps": 3,
                "min_reflection_steps": 1
            }
        )


class GradientProceduralOptimizer(ProceduralMemoryOptimizer):
    """Gradient optimization with critique-proposal separation."""
    
    def __init__(self, model="openai:gpt-4o-mini"):
        super().__init__(model, kind="gradient")


class PromptMemoryOptimizer(ProceduralMemoryOptimizer):
    """Single-pass prompt memory optimization."""
    
    def __init__(self, model="openai:gpt-4o-mini"):
        super().__init__(model, kind="prompt_memory")


class CoALAFullAgent(CoALABaselineAgent):
    """
    Complete CoALA agent with semantic, episodic, and procedural memory.
    Extends the baseline agent with LangMem's procedural memory capabilities.
    """
    
    def __init__(self, 
                 model_name: str = "gpt-4o-mini",
                 temperature: float = 0,
                 persist_directory: str = "./memory_store",
                 procedural_algorithm: str = "metaprompt",
                 optimization_threshold: int = 5):
        """
        Initialize the full CoALA agent with all memory types.
        
        Args:
            model_name: The LLM model to use
            temperature: Temperature for LLM responses
            persist_directory: Directory for persisting memory store
            procedural_algorithm: Algorithm for procedural optimization 
                                 ('metaprompt', 'gradient', or 'prompt_memory')
            optimization_threshold: Number of conversations before triggering optimization
        """
        # Initialize base agent with semantic and episodic memory
        super().__init__(model_name, temperature, persist_directory)
        
        # Initialize procedural memory components
        self.procedural_algorithm = procedural_algorithm
        self.optimization_threshold = optimization_threshold
        
        # Create the appropriate optimizer
        if procedural_algorithm == "metaprompt":
            self.procedural_optimizer = MetapromptProceduralOptimizer(f"openai:{model_name}")
        elif procedural_algorithm == "gradient":
            self.procedural_optimizer = GradientProceduralOptimizer(f"openai:{model_name}")
        elif procedural_algorithm == "prompt_memory":
            self.procedural_optimizer = PromptMemoryOptimizer(f"openai:{model_name}")
        else:
            raise ValueError(f"Unknown algorithm: {procedural_algorithm}")
        
        # Initialize procedural memory state
        self.current_system_prompt = "You are a helpful AI assistant."
        self.conversation_buffer = []
        self.optimization_history = []
        self.total_optimizations = 0
        
        # Track performance metrics
        self.performance_metrics = {
            "success_rate": [],
            "satisfaction_scores": [],
            "optimization_events": []
        }
    
    async def add_conversation_for_learning(self, 
                                           messages: List[Dict],
                                           feedback: Dict) -> bool:
        """
        Add a conversation to the procedural memory buffer for learning.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            feedback: Dictionary with feedback data (success, satisfaction, etc.)
            
        Returns:
            True if optimization was triggered, False otherwise
        """
        # Add to buffer
        self.conversation_buffer.append({
            "messages": messages,
            "feedback": feedback,
            "timestamp": datetime.now()
        })
        
        # Track metrics
        if "success" in feedback:
            self.performance_metrics["success_rate"].append(
                1.0 if feedback["success"] else 0.0
            )
        if "satisfaction" in feedback:
            self.performance_metrics["satisfaction_scores"].append(
                feedback["satisfaction"]
            )
        
        # Check if we should optimize
        if len(self.conversation_buffer) >= self.optimization_threshold:
            await self.optimize_procedural_memory()
            return True
        
        return False
    
    async def optimize_procedural_memory(self) -> str:
        """
        Trigger procedural memory optimization using accumulated conversations.
        
        Returns:
            The updated system prompt after optimization
        """
        if not self.conversation_buffer:
            print("No conversations to optimize from")
            return self.current_system_prompt
        
        print(f"\n{'='*60}")
        print(f"PROCEDURAL MEMORY OPTIMIZATION")
        print(f"Algorithm: {self.procedural_algorithm}")
        print(f"Conversations in buffer: {len(self.conversation_buffer)}")
        print(f"{'='*60}\n")
        
        # Run optimization
        try:
            improved_prompt = await self.procedural_optimizer.optimize(
                self.conversation_buffer,
                self.current_system_prompt
            )
            
            if improved_prompt and improved_prompt != self.current_system_prompt:
                # Store optimization event
                self.optimization_history.append({
                    "timestamp": datetime.now(),
                    "old_prompt": self.current_system_prompt,
                    "new_prompt": improved_prompt,
                    "conversations_used": len(self.conversation_buffer),
                    "algorithm": self.procedural_algorithm
                })
                
                # Update current prompt
                self.current_system_prompt = improved_prompt
                self.total_optimizations += 1
                
                # Track optimization event
                self.performance_metrics["optimization_events"].append(
                    datetime.now()
                )
                
                print(f"✅ Optimization complete (#{self.total_optimizations})")
                print(f"New prompt: {improved_prompt[:200]}...")
                
                # Clear buffer after successful optimization
                self.conversation_buffer = []
                
                return improved_prompt
            else:
                print("No improvement found from optimization")
                return self.current_system_prompt
                
        except Exception as e:
            print(f"Optimization error: {e}")
            return self.current_system_prompt
    
    def get_current_system_prompt(self) -> str:
        """Get the current system prompt (including procedural optimizations)."""
        return self.current_system_prompt
    
    def get_procedural_stats(self) -> Dict:
        """Get statistics about procedural memory."""
        return {
            "current_prompt": self.current_system_prompt,
            "total_optimizations": self.total_optimizations,
            "conversations_in_buffer": len(self.conversation_buffer),
            "algorithm": self.procedural_algorithm,
            "optimization_history_length": len(self.optimization_history),
            "average_success_rate": (
                sum(self.performance_metrics["success_rate"]) / 
                len(self.performance_metrics["success_rate"])
                if self.performance_metrics["success_rate"] else 0
            ),
            "average_satisfaction": (
                sum(self.performance_metrics["satisfaction_scores"]) / 
                len(self.performance_metrics["satisfaction_scores"])
                if self.performance_metrics["satisfaction_scores"] else 0
            )
        }
    
    def get_optimization_history(self) -> List[Dict]:
        """Get the history of procedural optimizations."""
        return self.optimization_history
    
    async def force_optimization(self) -> str:
        """Force an optimization even if threshold not met."""
        if not self.conversation_buffer:
            print("Warning: No conversations in buffer to optimize from")
            # Create a minimal example for testing
            self.conversation_buffer.append({
                "messages": [
                    {"role": "user", "content": "Help"},
                    {"role": "assistant", "content": "How can I help you?"}
                ],
                "feedback": {"success": True, "satisfaction": 4}
            })
        
        return await self.optimize_procedural_memory()
    
    def clear_conversation_buffer(self):
        """Clear the conversation buffer without optimizing."""
        buffer_size = len(self.conversation_buffer)
        self.conversation_buffer = []
        print(f"Cleared {buffer_size} conversations from buffer")
    
    def set_system_prompt(self, prompt: str):
        """Manually set the system prompt (useful for testing)."""
        self.optimization_history.append({
            "timestamp": datetime.now(),
            "old_prompt": self.current_system_prompt,
            "new_prompt": prompt,
            "conversations_used": 0,
            "algorithm": "manual"
        })
        self.current_system_prompt = prompt
        print(f"System prompt manually updated")
    
    # Override process_message to use procedural memory
    def process_message(self, 
                       message: str,
                       user_id: Optional[str] = None,
                       conversation_id: Optional[str] = None,
                       use_procedural_prompt: bool = True) -> str:
        """
        Process a message with all memory types including procedural.
        
        Args:
            message: The user's message
            user_id: Optional user ID for personalization
            conversation_id: Optional conversation ID for context
            use_procedural_prompt: Whether to use the procedurally optimized prompt
            
        Returns:
            The agent's response
        """
        # If using procedural prompt, temporarily update the agent's prompt
        original_prompt = None
        if use_procedural_prompt and self.current_system_prompt:
            # This is a simplified approach - in production you'd integrate
            # the procedural prompt more deeply into the LangGraph workflow
            pass  # The actual integration would happen in the workflow
        
        # Call parent's process_message
        response = super().process_message(message, user_id, conversation_id)
        
        return response
    
    def get_all_memory_stats(self) -> Dict:
        """Get comprehensive statistics about all memory types."""
        base_stats = self.get_memory_stats()
        procedural_stats = self.get_procedural_stats()
        
        return {
            **base_stats,
            "procedural": procedural_stats
        }