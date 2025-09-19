# domain_investment/investment_advisor_agent.py
import os
import sys
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domain_agent import DomainAgent, DomainProcedure
from typing import Dict, List, Optional
from .investment_advisor_prompts import INVESTMENT_PROMPTS

class InvestmentAdvisorAgent(DomainAgent):
    """Investment advisor specific implementation"""
    
    def __init__(self):
        # Set domain-specific paths
        self.domain_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.domain_dir, "investment_advisor_data")
        
        # Use generic "domain_memory_store" name for reusability across domains
        self.memory_dir = os.path.join(self.domain_dir, "domain_memory_store")
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.memory_dir, exist_ok=True)
    
    def get_procedure_class(self) -> type:
        return DomainProcedure
    
    def get_community_definitions(self) -> Dict[str, Dict]:
        return {
            "conservative_retirees": {
                "age": (60, 100),
                "risk_tolerance": "conservative",
                "min_members": 2
            },
            "aggressive_millennials": {
                "age": (25, 40),
                "risk_tolerance": "aggressive",
                "min_members": 2
            },
            "moderate_professionals": {
                "age": (35, 60),
                "risk_tolerance": "moderate",
                "min_members": 2
            }
        }
    
    def get_learning_prompts(self) -> Dict[str, str]:
        """Get prompts from investment_advisor_prompts.py"""
        return INVESTMENT_PROMPTS
    
    def get_response_prompt_template(self) -> str:
        """Get response prompt from investment_advisor_prompts.py"""
        return INVESTMENT_PROMPTS.get("response", "")
    
    def get_semantic_extraction_prompt(self) -> str:
        """Get semantic extraction prompt from investment_advisor_prompts.py"""
        return INVESTMENT_PROMPTS.get("semantic", "")
    
    def identify_task_type(self, query: str) -> str:
        query_lower = query.lower()
        task_map = {
            "rebalanc": "rebalancing",
            "tax": "tax_planning", 
            "risk": "risk_assessment",
            "esg": "esg_investing",
            "sustainable": "esg_investing",
            "retirement": "retirement_planning"
        }
        for key, task in task_map.items():
            if key in query_lower:
                return task
        return "general"
    
    def identify_community(self, user_id: str, user_profile: Optional[Dict]) -> str:
        if user_profile:
            age = user_profile.get("age", 35)
            risk = user_profile.get("risk_tolerance", "moderate")
        else:
            # Use user_id to determine community (for demo)
            user_num = int(user_id) if user_id.isdigit() else hash(user_id) % 100
            if user_num % 3 == 0:
                age, risk = 30, "aggressive"
            elif user_num % 3 == 1:
                age, risk = 65, "conservative"
            else:
                age, risk = 45, "moderate"
        
        if age >= 60 and risk == "conservative":
            return "conservative_retirees"
        elif age <= 40 and risk == "aggressive":
            return "aggressive_millennials"
        else:
            return "moderate_professionals"
    
    def extract_profile(self, facts: List[Dict], query: str) -> Dict:
        profile = {}
        for fact in facts:
            if "risk tolerance" in fact.get("predicate", "").lower():
                profile["risk_tolerance"] = fact.get("object", "").lower()
            if "age" in fact.get("predicate", "").lower():
                try:
                    age = int(fact.get("object", "0"))
                    profile["age"] = age
                    if age < 30:
                        profile["age_group"] = "young_professional"
                    elif age < 50:
                        profile["age_group"] = "millennial"
                    elif age < 65:
                        profile["age_group"] = "pre_retirement"
                    else:
                        profile["age_group"] = "retirement"
                except:
                    pass
        return profile
    
    def format_procedural_context(self, strategy: Dict) -> str:
        context = f"\n📋 Investment Strategy (from {strategy.get('source', 'unknown')}):\n"
        context += f"Strategy: {strategy['strategy']}\n"
        context += f"Confidence: {strategy['confidence']:.1%}\n"
        context += f"Scope: {strategy.get('scope', 'unknown')}\n"
        
        # Add domain-specific metrics if present
        if 'domain_metrics' in strategy:
            metrics = strategy['domain_metrics']
            if 'avg_portfolio_performance' in metrics:
                context += f"Avg Portfolio Performance: {metrics['avg_portfolio_performance']:.2f}%\n"
        
        context += "Steps:\n"
        for i, step in enumerate(strategy['steps'], 1):
            context += f"  {i}. {step}\n"
        return context
    
    def calculate_success_score(self, performance_data: Dict) -> float:
        client_satisfied = performance_data.get("client_satisfaction", 5) > 7
        portfolio_performed = performance_data.get("returns", 0) > 0
        goals_met = performance_data.get("goals_achieved", False)
        
        return (
            0.3 * (1.0 if client_satisfied else 0.0) +
            0.5 * (1.0 if portfolio_performed else 0.0) +
            0.2 * (1.0 if goals_met else 0.0)
        )
    
    def update_domain_metrics(self, procedure: DomainProcedure, performance_data: Dict) -> None:
        if "returns" in performance_data:
            current = procedure.domain_metrics.get("avg_portfolio_performance", 0.0)
            procedure.domain_metrics["avg_portfolio_performance"] = (
                current * 0.9 + performance_data["returns"] * 0.1
            )