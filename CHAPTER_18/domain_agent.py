# domain_agent.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

class DomainProcedure(BaseModel):
    """Generic procedure structure that all domains must implement"""
    strategy_pattern: str
    steps: List[str]
    segments: List[str] = Field(default_factory=list)
    success_rate: float = 1.0
    usage_count: int = 0
    adaptations: List[Dict] = Field(default_factory=list)
    
    scope: str = "global"  
    scope_id: Optional[str] = None
    priority: int = 0
    learned_from_count: int = 1
    
    domain_metrics: Dict[str, float] = Field(default_factory=dict)

class DomainAgent(ABC):
    """Abstract base class for domain-specific agents"""
    
    @abstractmethod
    def get_procedure_class(self) -> type:
        """Return the procedure class for this domain"""
        pass
    
    @abstractmethod
    def get_community_definitions(self) -> Dict[str, Dict]:
        """Return community/segment definitions for this domain"""
        pass
    
    @abstractmethod
    def get_learning_prompts(self) -> Dict[str, str]:
        """Return domain-specific learning prompts"""
        pass
    
    @abstractmethod
    def identify_task_type(self, query: str) -> str:
        """Identify task type from query for this domain"""
        pass
    
    @abstractmethod
    def identify_community(self, user_id: str, user_profile: Optional[Dict]) -> str:
        """Determine which community a user belongs to"""
        pass
    
    @abstractmethod
    def extract_profile(self, facts: List[Dict], query: str) -> Dict:
        """Extract user profile from facts for this domain"""
        pass
    
    @abstractmethod
    def format_procedural_context(self, strategy: Dict) -> str:
        """Format procedural strategy for display in this domain"""
        pass
    
    @abstractmethod
    def calculate_success_score(self, performance_data: Dict) -> float:
        """Calculate success score from performance data for this domain"""
        pass
    
    @abstractmethod
    def update_domain_metrics(self, procedure: DomainProcedure, performance_data: Dict) -> None:
        """Update domain-specific metrics on a procedure"""
        pass
    
    @abstractmethod
    def get_response_prompt_template(self) -> str:
        """Get the domain-specific response generation prompt"""
        pass
    
    @abstractmethod
    def get_semantic_extraction_prompt(self) -> str:
        """Get the domain-specific semantic extraction prompt"""
        pass