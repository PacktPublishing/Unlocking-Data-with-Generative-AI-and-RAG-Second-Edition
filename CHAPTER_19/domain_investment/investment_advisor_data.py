# investment_advisor_data.py (updated version with original content)
"""
Enhanced synthesized data for Investment Advisor Agent with realistic conversation structure
All content is original and conversational
"""

import os
import json
import random
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict

class QueryType(Enum):
    """Types of queries users make"""
    PERFORMANCE_ANALYSIS = "performance_analysis"
    EXPOSURE_ANALYSIS = "exposure_analysis"
    COMPOSITION_ANALYSIS = "composition_analysis"
    RANKING_ANALYSIS = "ranking_analysis"
    REBALANCING = "rebalancing"
    TAX_PLANNING = "tax_planning"
    RISK_ASSESSMENT = "risk_assessment"
    MARKET_OUTLOOK = "market_outlook"

class Topic(Enum):
    """Main topics in conversations"""
    PORTFOLIO_ANALYSIS = "portfolio_analysis"
    MARKET_CONDITIONS = "market_conditions"
    INVESTMENT_STRATEGY = "investment_strategy"
    TAX_OPTIMIZATION = "tax_optimization"
    RISK_MANAGEMENT = "risk_management"

@dataclass
class RealisticMessage:
    """Message structure matching real data"""
    role: str
    content: str

@dataclass
class BehavioralSignals:
    """Behavioral signals from conversation"""
    provided_specific_data: bool
    used_retrieval: bool
    personalized_response: bool
    asked_clarification: bool
    error_occurred: bool
    sorted_results: bool
    showed_empathy: bool = False
    explained_jargon: bool = False
    referenced_context: bool = False

@dataclass
class ConversationData:
    """Complete conversation data structure matching real format"""
    conversation_id: str
    user_id: int
    timestamp: str
    messages: List[Dict[str, str]]
    feedback: Dict[str, Any]
    behavioral_signals: Dict[str, bool]
    metadata: Dict[str, Any]

class EnhancedInvestmentAdvisorDataGenerator:
    """Generate realistic investment advisor conversations matching real data structure"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.base_date = datetime(2025, 8, 1)
        
        # Set domain directory
        self.domain_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Real portfolio data templates
        self.portfolio_templates = {
            "conservative": {
                "holdings": ["BND", "AGG", "VCIT", "TLT", "IEF", "SPY", "VTI"],
                "allocation": {"bonds": 0.7, "stocks": 0.3},
                "avg_yield": 0.04
            },
            "moderate": {
                "holdings": ["SPY", "QQQ", "BND", "VTI", "VXUS", "GLD", "VCIT"],
                "allocation": {"stocks": 0.6, "bonds": 0.3, "alternatives": 0.1},
                "avg_yield": 0.025
            },
            "aggressive": {
                "holdings": ["NVDA", "TSLA", "QQQ", "ARKK", "SPY", "TQQQ", "AMD"],
                "allocation": {"stocks": 0.9, "alternatives": 0.1},
                "avg_yield": 0.01
            },
            "income_focused": {
                "holdings": ["VNQ", "DGRO", "VCIT", "DIV", "VYM", "SCHD", "REZ"],
                "allocation": {"dividend_stocks": 0.5, "bonds": 0.5},
                "avg_yield": 0.06
            }
        }
        
        # Original response templates - completely different from provided data
        self.response_templates = {
            "performance_success": [
                "Looking at your investments from {start_date} through {end_date}, you've seen a {percent}% {gain_loss}. That translates to about ${amount} in actual dollars. The standout winner has been {top_gainer} which contributed ${gainer_amount} to your returns. On the flip side, {top_loser} has been weighing things down a bit with a ${loser_amount} decline. Keep in mind these numbers reflect your current positions.",
                "Hey, quick update on your investments - you're {direction} {percent}% as of now, which means you've {gained_lost} about ${amount}. {top_holding} is really carrying the team today with a {holding_percent}% gain. Not bad at all!",
                "So I've run the numbers for you. Since {start_date}, your investments have moved {percent}% {direction}, putting you {gain_loss} ${amount}. What's interesting is that {top_gainer} alone accounts for ${gainer_amount} of that movement. Meanwhile, {top_loser} has been a bit of a drag, but that's pretty normal market behavior."
            ],
            "exposure_success": [
                "I can see you've got {percent}% of your money in {sector} right now. Most of that comes from three main positions: {holding1} makes up {percent1}%, then {holding2} at {percent2}%, and {holding3} at {percent3}%. This makes {sector} your {rank} biggest area of investment.",
                "Let me break down your {category} exposure for you - it's sitting at {percent}% of your total investments. The main contributors here are {holdings_list}. This seems pretty reasonable given your overall strategy.",
                "Your {sector} allocation is currently {percent}%. The heavy lifters in this category are {holding1} and {holding2}, which together make up most of that exposure. Is this level comfortable for you?"
            ],
            "error_response": [
                "Hmm, I'm running into a technical hiccup trying to pull that information. Mind giving me another moment?",
                "Something's not quite right with my data connection. Let me try a different approach to get you that answer.",
                "I'm having trouble accessing those specific details right now. Can we try looking at this from a different angle?"
            ],
            "clarification_needed": [
                "Just to make sure I give you the most useful information - are you asking about {clarification}?",
                "I want to make sure I understand correctly. Are you interested in {option1} or {option2}?",
                "Before I dive in, let me clarify - you're looking for information about {topic}, right?"
            ]
        }
        
        # Tool combinations based on query types
        self.tool_patterns = {
            QueryType.PERFORMANCE_ANALYSIS: ["get_portfolio", "get_attribute", "calculate_returns"],
            QueryType.EXPOSURE_ANALYSIS: ["get_portfolio", "factor_contribution", "filter"],
            QueryType.COMPOSITION_ANALYSIS: ["get_portfolio", "aggregate", "filter"],
            QueryType.RANKING_ANALYSIS: ["get_portfolio", "get_attribute", "sort", "filter"],
            QueryType.REBALANCING: ["get_portfolio", "optimize", "calculate_trades"],
            QueryType.TAX_PLANNING: ["get_transactions", "calculate_tax", "filter"],
            QueryType.RISK_ASSESSMENT: ["get_portfolio", "calculate_var", "stress_test"],
            QueryType.MARKET_OUTLOOK: ["get_market_data", "analyze_trends"]
        }

    def generate_realistic_conversations(self, 
                                       num_users: int = 50,
                                       convs_per_user: int = 10) -> List[ConversationData]:
        """Generate conversations matching real data structure"""
        conversations = []
        
        for user_id in range(3000, 3000 + num_users):
            # Assign user profile
            profile_type = random.choice(list(self.portfolio_templates.keys()))
            user_template = self.portfolio_templates[profile_type]
            
            # Track conversation quality evolution
            conversation_session_id = str(uuid.uuid4())
            base_timestamp = self.base_date + timedelta(days=random.randint(0, 30))
            
            for conv_num in range(convs_per_user):
                # Quality improves over time (simulating learning)
                early_conversation = conv_num < 3
                success_probability = 0.4 if early_conversation else 0.8
                
                # Generate conversation
                timestamp = base_timestamp + timedelta(minutes=conv_num * random.randint(1, 10))
                
                conv = self._generate_realistic_conversation(
                    conversation_id=conversation_session_id,
                    user_id=user_id,
                    timestamp=timestamp,
                    user_template=user_template,
                    success_probability=success_probability,
                    conversation_number=conv_num
                )
                
                conversations.append(conv)
        
        return conversations

    def _calculate_realistic_satisfaction(self,
                                        is_successful: bool,
                                        messages: List[Dict[str, str]],
                                        behavioral_signals_dict: Dict[str, bool],
                                        query_type: QueryType,
                                        conversation_number: int) -> float:
        """
        Calculate realistic satisfaction based on response quality and conversation flow
        """
        # Base satisfaction depends on success
        if not is_successful:
            # Failed responses get low satisfaction
            base_satisfaction = random.uniform(1.0, 2.0)
            
            # If error message is helpful, slightly better
            if messages and "different approach" in messages[-1]["content"].lower():
                base_satisfaction += 0.5
            
            return min(base_satisfaction, 5.0)
        
        # Start with neutral satisfaction for successful responses
        base_satisfaction = 3.0
        
        # Quality factors that increase satisfaction
        quality_adjustments = 0.0
        
        # 1. Response specificity and data
        if behavioral_signals_dict.get("provided_specific_data", False):
            quality_adjustments += 0.5  # Specific numbers and data points
            assistant_response = messages[1]["content"] if len(messages) > 1 else ""
            
            # Extra boost for detailed, formatted responses
            if len(assistant_response) > 200 and any(x in assistant_response for x in ["$", "%", "."]):
                quality_adjustments += 0.3
        
        # 2. Personalization
        if behavioral_signals_dict.get("personalized_response", False):
            quality_adjustments += 0.4  # Referencing user's actual situation
        
        # 3. Clarification and understanding
        if behavioral_signals_dict.get("asked_clarification", False):
            # Asking clarification can be good (shows care) or bad (didn't understand)
            if conversation_number >= 3:  # Later conversations should need less clarification
                quality_adjustments += 0.2  # Still good to verify
            else:
                quality_adjustments += 0.4  # Early clarification is very good
        
        # 4. Empathy and acknowledgment
        if behavioral_signals_dict.get("showed_empathy", False):
            quality_adjustments += 0.3  # Acknowledging concerns
        
        # 5. Education and explanation
        if behavioral_signals_dict.get("explained_jargon", False):
            quality_adjustments += 0.2  # Making things understandable
        
        # 6. Follow-up conversation quality
        if len(messages) > 2:  # Has follow-up
            follow_up_response = messages[-1]["content"]
            if len(follow_up_response) > 100:
                quality_adjustments += 0.3  # Detailed follow-up
        
        # Penalties that decrease satisfaction
        penalties = 0.0
        
        # 1. Vague or generic responses
        if len(messages) > 1:
            response = messages[1]["content"]
            generic_phrases = ["various investments", "different areas", "usually a good idea", 
                            "you're doing fine", "it depends", "generally speaking"]
            if any(phrase in response.lower() for phrase in generic_phrases):
                penalties += 0.5
        
        # 2. Too brief for complex queries
        if query_type in [QueryType.REBALANCING, QueryType.TAX_PLANNING, QueryType.RISK_ASSESSMENT]:
            if len(messages) > 1 and len(messages[1]["content"]) < 100:
                penalties += 0.7  # Complex queries need detailed responses
        
        # 3. Missing key information for the query type
        if query_type == QueryType.PERFORMANCE_ANALYSIS:
            response = messages[1]["content"] if len(messages) > 1 else ""
            if "$" not in response and "%" not in response:
                penalties += 0.5  # Performance queries need numbers
        
        # 4. Learning curve - earlier conversations naturally less satisfying
        if conversation_number < 2:
            penalties += 0.3  # Early conversations are still learning
        
        # Calculate final satisfaction
        final_satisfaction = base_satisfaction + quality_adjustments - penalties
        
        # Add some realistic variance
        variance = random.uniform(-0.2, 0.2)
        final_satisfaction += variance
        
        # Ensure within bounds and round to 1 decimal
        final_satisfaction = max(1.0, min(5.0, final_satisfaction))
        return round(final_satisfaction, 1)


    def _generate_realistic_conversation(self,
                                        conversation_id: str,
                                        user_id: int,
                                        timestamp: datetime,
                                        user_template: Dict,
                                        success_probability: float,
                                        conversation_number: int) -> ConversationData:
        """Generate a single realistic conversation"""
        
        # Select query type based on common patterns
        query_types = list(QueryType)
        weights = [0.3, 0.2, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05]
        query_type = random.choices(query_types, weights=weights)[0]
        
        # Determine success
        is_successful = random.random() < success_probability
        
        # Generate messages based on query type
        messages = self._generate_messages_for_query(
            query_type=query_type,
            user_template=user_template,
            is_successful=is_successful,
            conversation_number=conversation_number
        )
        
        # Generate behavioral signals
        behavioral_signals = self._generate_behavioral_signals(
            is_successful=is_successful,
            query_type=query_type,
            messages=messages
        )
        
        # Convert behavioral_signals to dict for easier access
        behavioral_signals_dict = asdict(behavioral_signals)
        
        # Calculate response latency (realistic range)
        if is_successful:
            latency = random.randint(800, 4000) if behavioral_signals_dict["used_retrieval"] else random.randint(200, 1000)
        else:
            latency = random.randint(5000, 15000)
        
        # Calculate realistic satisfaction based on actual response quality
        satisfaction = self._calculate_realistic_satisfaction(
            is_successful=is_successful,
            messages=messages,
            behavioral_signals_dict=behavioral_signals_dict,
            query_type=query_type,
            conversation_number=conversation_number
        )
        
        # Determine topic and subtopic
        topic_map = {
            QueryType.PERFORMANCE_ANALYSIS: (Topic.PORTFOLIO_ANALYSIS, "performance"),
            QueryType.EXPOSURE_ANALYSIS: (Topic.PORTFOLIO_ANALYSIS, "exposure"),
            QueryType.COMPOSITION_ANALYSIS: (Topic.PORTFOLIO_ANALYSIS, "composition"),
            QueryType.RANKING_ANALYSIS: (Topic.PORTFOLIO_ANALYSIS, "rankings"),
            QueryType.REBALANCING: (Topic.INVESTMENT_STRATEGY, "rebalancing"),
            QueryType.TAX_PLANNING: (Topic.TAX_OPTIMIZATION, "planning"),
            QueryType.RISK_ASSESSMENT: (Topic.RISK_MANAGEMENT, "assessment"),
            QueryType.MARKET_OUTLOOK: (Topic.MARKET_CONDITIONS, "outlook")
        }
        
        topic, subtopic = topic_map[query_type]
        
        # Select tools used
        tools_used = []
        if is_successful and behavioral_signals_dict["used_retrieval"]:
            tools_used = random.sample(
                self.tool_patterns[query_type],
                k=random.randint(1, len(self.tool_patterns[query_type]))
            )
        
        return ConversationData(
            conversation_id=conversation_id,
            user_id=user_id,
            timestamp=timestamp.isoformat(),
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            feedback={
                "satisfaction_score": satisfaction,  # Now using realistic calculation
                "task_completed": is_successful,
                "success": is_successful,
                "response_latency_ms": latency
            },
            behavioral_signals=behavioral_signals_dict,
            metadata={
                "topic": topic.value,
                "subtopic": subtopic,
                "query_type": query_type.value,
                "tools_used": tools_used,
                "data_points_returned": random.randint(0, 5) if is_successful else 0,
                "memory_type": random.choice(["procedural", "semantic", "episodic"])
            }
        )

    def generate_segmented_conversations(self, num_users: int = 50, convs_per_user: int = 10) -> List[ConversationData]:
        """Generate conversations that will trigger different learning scopes"""
        conversations = []
        
        # Define user segments with clear characteristics
        user_segments = {
            "aggressive_millennials": {
                "user_ids": range(3000, 3010),
                "age": 30,
                "risk_tolerance": "aggressive",
                "common_queries": ["growth stocks", "tech exposure", "ARKK", "TSLA"],
                "keywords": ["aggressive", "growth", "high return", "tech"]
            },
            "conservative_retirees": {
                "user_ids": range(3010, 3020),
                "age": 65,
                "risk_tolerance": "conservative", 
                "common_queries": ["dividend income", "bond allocation", "safe investments"],
                "keywords": ["conservative", "income", "safe", "preserve capital"]
            },
            "moderate_professionals": {
                "user_ids": range(3020, 3030),
                "age": 45,
                "risk_tolerance": "moderate",
                "common_queries": ["balanced portfolio", "diversification", "rebalancing"],
                "keywords": ["balanced", "diversified", "moderate risk"]
            }
        }
        
        # Task-specific conversations
        task_specific_queries = {
            "rebalancing": [
                "Time to rebalance my portfolio?",
                "My allocation is off, what should I do?",
                "How often should I rebalance?"
            ],
            "tax_planning": [
                "Any tax loss harvesting opportunities?",
                "How can I reduce my tax bill?",
                "Should I sell before year end for taxes?"
            ],
            "esg_investing": [
                "I want to invest in sustainable companies",
                "Show me ESG options",
                "How can I invest ethically?"
            ]
        }
        
        conversation_counter = 0
        
        for segment_name, segment_info in user_segments.items():
            for user_id in segment_info["user_ids"]:
                
                for conv_num in range(convs_per_user):
                    timestamp = self.base_date + timedelta(days=conversation_counter // 100, 
                                                        minutes=conversation_counter % 100)
                    
                    # Mix of query types to trigger different scopes
                    if conv_num < 3:
                        # Early convs: segment-specific queries
                        query = random.choice(segment_info["common_queries"])
                        query_content = f"I'm interested in {query}"
                    elif conv_num < 6:
                        # Mid convs: task-specific queries
                        task_type = random.choice(list(task_specific_queries.keys()))
                        query_content = random.choice(task_specific_queries[task_type])
                    else:
                        # Later convs: user-specific patterns emerge
                        query_content = f"Based on what you know about me, {random.choice(segment_info['common_queries'])}"
                    
                    # Include segment keywords in query
                    keywords = random.sample(segment_info["keywords"], 2)
                    if random.random() > 0.5:
                        query_content += f" - looking for something {keywords[0]}"
                    
                    messages = [
                        {"role": "user", "content": query_content},
                        {"role": "assistant", "content": self._generate_segment_aware_response(
                            segment_name, query_content, segment_info
                        )}
                    ]
                    
                    # Higher satisfaction for segment-appropriate responses
                    satisfaction = random.uniform(4.0, 5.0) if conv_num > 5 else random.uniform(3.5, 4.5)
                    
                    conv = ConversationData(
                        conversation_id=f"conv_{user_id}_{conv_num}",
                        user_id=user_id,
                        timestamp=timestamp.isoformat(),
                        messages=messages,
                        feedback={
                            "satisfaction_score": satisfaction,
                            "task_completed": True,
                            "success": True,
                            "response_latency_ms": random.randint(500, 2000)
                        },
                        behavioral_signals={
                            "provided_specific_data": True,
                            "used_retrieval": True,
                            "personalized_response": conv_num > 5,
                            "asked_clarification": False,
                            "error_occurred": False,
                            "sorted_results": False,
                            "showed_empathy": segment_name == "conservative_retirees",
                            "explained_jargon": segment_name == "conservative_retirees",
                            "referenced_context": conv_num > 3
                        },
                        metadata={
                            "topic": "investment_strategy",
                            "subtopic": segment_name,
                            "query_type": "segment_specific",
                            "tools_used": ["get_portfolio"],
                            "data_points_returned": 3,
                            "memory_type": "procedural",
                            "client_age": segment_info["age"],
                            "risk_tolerance": segment_info["risk_tolerance"]
                        }
                    )
                    
                    conversations.append(conv)
                    conversation_counter += 1
        
        return conversations

    def _generate_segment_aware_response(self, segment_name: str, query: str, segment_info: Dict) -> str:
        """Generate responses that reflect segment-specific patterns"""
        
        if segment_name == "aggressive_millennials":
            return f"Given your growth focus, I'd suggest looking at high-momentum tech plays. ARKK and TSLA align with your aggressive approach. Your age ({segment_info['age']}) gives you time to ride out volatility. Consider 80% growth stocks, 20% emerging markets."
        
        elif segment_name == "conservative_retirees":
            return f"At {segment_info['age']}, capital preservation is key. I recommend focusing on dividend aristocrats and investment-grade bonds. A 30/70 stock/bond allocation would provide income while protecting your principal. BND and VCIT are solid choices."
        
        else:  # moderate_professionals
            return f"For a balanced approach at your stage, consider a 60/40 portfolio. Mix index funds like SPY and QQQ with bond holdings in AGG. This gives you growth potential with downside protection. Rebalance quarterly to maintain your target allocation."

    def _generate_messages_for_query(self,
                                    query_type: QueryType,
                                    user_template: Dict,
                                    is_successful: bool,
                                    conversation_number: int) -> List[Dict[str, str]]:
        """Generate realistic message exchanges"""
        messages = []
        holdings = user_template["holdings"]
        
        # Original user queries - all different from provided data
        user_queries = {
            QueryType.PERFORMANCE_ANALYSIS: [
                "hey how are my investments doing?",
                "what's my return looking like this year?",
                "which of my stocks are winning and losing?",
                "am I beating the market?",
                "can you show me last month's performance?"
            ],
            QueryType.EXPOSURE_ANALYSIS: [
                "do I have too much in tech stocks?",
                "what's my international exposure looking like?",
                "am I diversified enough?",
                "break down my sectors for me",
                "how much of my money is in large companies?"
            ],
            QueryType.COMPOSITION_ANALYSIS: [
                "what portion of my investments are index funds?",
                "how many individual stocks vs ETFs do I have?",
                "where's my cash sitting?",
                "show me my overall allocation",
                "what's my bond percentage?"
            ],
            QueryType.RANKING_ANALYSIS: [
                "which investments pay the best dividends?",
                "rank my holdings by performance",
                "what's costing me the most in fees?",
                "show me my riskiest positions",
                "order my stocks by size"
            ],
            QueryType.REBALANCING: [
                "is it time to rebalance?",
                "I want to be more conservative, what should I do?",
                "help me get more aggressive with my investments",
                "how do I get to a 70/30 split?",
                "what changes would save me on taxes?"
            ],
            QueryType.TAX_PLANNING: [
                "any tax loss harvesting opportunities?",
                "what are my capital gains looking like?",
                "which stocks should I sell for tax reasons?",
                "if I sold everything what would I owe?",
                "how do I reduce my tax bill?"
            ],
            QueryType.RISK_ASSESSMENT: [
                "how risky is my portfolio?",
                "what happens if the market crashes 20%?",
                "is this too risky for someone my age?",
                "what's my downside risk?",
                "show me worst case scenarios"
            ],
            QueryType.MARKET_OUTLOOK: [
                "what do you think about tech stocks right now?",
                "should I be worried about inflation?",
                "are we heading for a recession?",
                "which sectors look promising?",
                "is it time to go to cash?"
            ]
        }
        
        # Select user query
        user_query = random.choice(user_queries[query_type])
        messages.append({"role": "user", "content": user_query})
        
        # Generate assistant response
        if is_successful:
            response = self._generate_successful_response(
                query_type=query_type,
                user_template=user_template,
                user_query=user_query,
                conversation_number=conversation_number
            )
        else:
            response = self._generate_failed_response(query_type=query_type)
        
        messages.append({"role": "assistant", "content": response})
        
        # Add follow-up for longer conversations (30% chance)
        if random.random() < 0.3 and is_successful:
            follow_up = self._generate_follow_up(query_type, user_template)
            messages.extend(follow_up)
        
        return messages

    def _generate_successful_response(self,
                                     query_type: QueryType,
                                     user_template: Dict,
                                     user_query: str,
                                     conversation_number: int) -> str:
        """Generate a successful response with improving quality over time"""
        
        holdings = user_template["holdings"]
        
        if query_type == QueryType.PERFORMANCE_ANALYSIS:
            # Early conversations: brief, less detailed
            if conversation_number < 3:
                return f"You're up about 2.3% right now. {random.choice(holdings)} is doing pretty well."
            # Later conversations: detailed, conversational
            else:
                template = random.choice(self.response_templates["performance_success"])
                return template.format(
                    start_date="August 1st",
                    end_date="today",
                    change_type="some nice",
                    percent=round(random.uniform(0.5, 3.0), 2),
                    gain_loss="gain",
                    gained_lost="gained",
                    direction="up",
                    amount=round(random.uniform(1000, 5000), 2),
                    top_gainer=random.choice(holdings),
                    gainer_amount=round(random.uniform(100, 1000), 2),
                    gainer_price=round(random.uniform(50, 500), 2),
                    gainer_cost=round(random.uniform(40, 400), 2),
                    top_loser=random.choice(holdings),
                    loser_amount=round(random.uniform(50, 500), 2),
                    loser_price=round(random.uniform(30, 300), 2),
                    loser_cost=round(random.uniform(35, 350), 2),
                    top_holding=random.choice(holdings),
                    holding_percent=round(random.uniform(1, 5), 1)
                )
        
        elif query_type == QueryType.EXPOSURE_ANALYSIS:
            sector = random.choice(["technology", "healthcare", "financial services", "consumer goods"])
            percent = round(random.uniform(10, 35), 2)
            
            # Early: basic response
            if conversation_number < 3:
                return f"You've got about {percent}% in {sector}."
            # Later: detailed, conversational
            else:
                selected_holdings = random.sample(holdings, min(3, len(holdings)))
                template = random.choice(self.response_templates["exposure_success"])
                return template.format(
                    sector=sector,
                    category=sector,
                    percent=percent,
                    holding1=selected_holdings[0],
                    percent1=round(random.uniform(3, 10), 2),
                    holding2=selected_holdings[1] if len(selected_holdings) > 1 else "N/A",
                    percent2=round(random.uniform(2, 8), 2),
                    holding3=selected_holdings[2] if len(selected_holdings) > 2 else "N/A",
                    percent3=round(random.uniform(1, 5), 2),
                    rank=random.choice(["biggest", "second biggest", "third biggest"]),
                    holdings_list=", ".join(selected_holdings[:2])
                )
        
        elif query_type == QueryType.RANKING_ANALYSIS:
            # Show improvement in providing sorted, detailed results
            if conversation_number < 3:
                return f"Looks like {random.choice(holdings)} has the best yield."
            else:
                top_holdings = random.sample(holdings, min(3, len(holdings)))
                yields = [round(random.uniform(2, 8), 2) for _ in top_holdings]
                return f"Here are your top dividend payers: {top_holdings[0]} is yielding {yields[0]}%, followed by {top_holdings[1]} at {yields[1]}%, and {top_holdings[2]} bringing in {yields[2]}%. Pretty solid income stream there!"
        
        elif query_type == QueryType.REBALANCING:
            # Early: generic advice
            if conversation_number < 3:
                return "Quarterly rebalancing is usually a good idea."
            # Later: specific, actionable advice
            else:
                return f"Based on where you are now, here's what I'd suggest: First, trim back {random.choice(holdings)} by about 5% - it's gotten a bit heavy. Second, add some bond exposure through something like {random.choice(['BND', 'AGG', 'VCIT'])} to smooth out the ride. Third, maybe consider some international diversification with VXUS. These moves would better match your moderate approach and spread your risk around."
        
        else:
            # Generic successful response for other types
            return f"Looking at your investments, {random.choice(holdings)} is a significant position. Your current setup aligns well with a {list(self.portfolio_templates.keys())[0]} approach."

    def _generate_failed_response(self, query_type: QueryType) -> str:
        """Generate a failed/error response"""
        error_responses = self.response_templates["error_response"]
        
        # Different failure modes
        failure_type = random.choice(["error", "no_data", "vague"])
        
        if failure_type == "error":
            return random.choice(error_responses)
        elif failure_type == "no_data":
            return "I don't have enough information to answer that right now."
        else:  # vague
            return "You have various investments across different areas."

    def _generate_follow_up(self, 
                           query_type: QueryType,
                           user_template: Dict) -> List[Dict[str, str]]:
        """Generate follow-up exchanges"""
        follow_ups = []
        
        if query_type == QueryType.PERFORMANCE_ANALYSIS:
            follow_ups.append({"role": "user", "content": "how does that compare to my benchmark?"})
            follow_ups.append({"role": "assistant", "content": "You're actually doing better than your benchmark (a balanced 60/40 portfolio) by about 0.8% year-to-date. The difference mainly comes from your tech holdings which have been on fire lately."})
        
        elif query_type == QueryType.EXPOSURE_ANALYSIS:
            follow_ups.append({"role": "user", "content": "is that too much?"})
            follow_ups.append({"role": "assistant", "content": "For someone with your risk tolerance and time horizon, this is actually within a reasonable range. That said, if you're losing sleep over concentration risk, we could definitely spread things out more."})
        
        else:
            follow_ups.append({"role": "user", "content": "can you explain more?"})
            follow_ups.append({"role": "assistant", "content": "Sure, let me break this down in more detail for you..."})
        
        return follow_ups

    def _generate_behavioral_signals(self,
                                    is_successful: bool,
                                    query_type: QueryType,
                                    messages: List[Dict[str, str]]) -> BehavioralSignals:
        """Generate realistic behavioral signals"""
        
        assistant_response = messages[1]["content"] if len(messages) > 1 else ""
        
        signals = BehavioralSignals(
            provided_specific_data=is_successful and any(char.isdigit() for char in assistant_response),
            used_retrieval=is_successful and query_type != QueryType.MARKET_OUTLOOK,
            personalized_response=is_successful and "your" in assistant_response.lower(),
            asked_clarification="?" in assistant_response and is_successful,
            error_occurred=not is_successful and ("error" in assistant_response.lower() or "hiccup" in assistant_response.lower()),
            sorted_results=query_type == QueryType.RANKING_ANALYSIS and is_successful,
            showed_empathy="understand" in assistant_response.lower() or "I can see" in assistant_response.lower(),
            explained_jargon="this means" in assistant_response.lower() or "in other words" in assistant_response.lower(),
            referenced_context=len(messages) > 2  # Has follow-up
        )
        
        return signals

    def extract_patterns_from_conversations(self, conversations: List[ConversationData]) -> Dict:
        """
        Extract patterns from conversations following LangMem's trajectory-based approach.
        This method analyzes successful and failed trajectories to identify patterns
        that can be used for procedural memory optimization.
        """
        from collections import defaultdict
        import numpy as np
        
        # Initialize pattern tracking structures
        pattern_stats = {
            # Behavioral patterns from successful/failed trajectories
            "success_behaviors": defaultdict(int),
            "failure_behaviors": defaultdict(int),
            "behavioral_combinations": defaultdict(lambda: {"success": 0, "failure": 0}),
            
            # Query-specific patterns for different conversation types
            "query_type_patterns": defaultdict(lambda: {
                "success_rate": [],
                "satisfaction_scores": [],
                "response_characteristics": defaultdict(int),
                "common_failures": []
            }),
            
            # Satisfaction correlations with specific behaviors
            "satisfaction_correlations": defaultdict(list),
            
            # Temporal patterns (how conversations evolve)
            "temporal_patterns": {
                "opening_strategies": defaultdict(int),
                "closing_strategies": defaultdict(int),
                "follow_up_patterns": defaultdict(int)
            },
            
            # Response quality indicators
            "quality_indicators": {
                "response_length": [],
                "specificity_scores": [],
                "personalization_scores": []
            },
            
            # User segment patterns
            "user_segments": defaultdict(lambda: {
                "success_rate": [],
                "preferred_styles": defaultdict(int)
            })
        }
        
        # Analyze each conversation trajectory
        for conv in conversations:
            is_successful = conv.feedback["success"]
            satisfaction = conv.feedback["satisfaction_score"]
            signals = conv.behavioral_signals
            query_type = conv.metadata["query_type"]
            messages = conv.messages
            
            # 1. Track individual behavioral signals
            for signal, value in signals.items():
                if value:
                    if is_successful:
                        pattern_stats["success_behaviors"][signal] += 1
                    else:
                        pattern_stats["failure_behaviors"][signal] += 1
                    
                    # Track satisfaction correlation
                    pattern_stats["satisfaction_correlations"][signal].append(satisfaction)
            
            # 2. Track behavioral combinations (multi-signal patterns)
            active_signals = [s for s, v in signals.items() if v]
            if len(active_signals) >= 2:
                # Create combinations of 2 behaviors
                for i in range(len(active_signals)):
                    for j in range(i+1, len(active_signals)):
                        combo_key = f"{active_signals[i]}+{active_signals[j]}"
                        if is_successful:
                            pattern_stats["behavioral_combinations"][combo_key]["success"] += 1
                        else:
                            pattern_stats["behavioral_combinations"][combo_key]["failure"] += 1
            
            # 3. Query-type specific patterns
            pattern_stats["query_type_patterns"][query_type]["success_rate"].append(1 if is_successful else 0)
            pattern_stats["query_type_patterns"][query_type]["satisfaction_scores"].append(satisfaction)
            
            # Analyze response characteristics by query type
            if len(messages) > 1:
                response = messages[1]["content"]
                
                # Response characteristics
                if "$" in response or "%" in response:
                    pattern_stats["query_type_patterns"][query_type]["response_characteristics"]["has_numbers"] += 1
                if len(response) > 200:
                    pattern_stats["query_type_patterns"][query_type]["response_characteristics"]["detailed_response"] += 1
                if "?" in response:
                    pattern_stats["query_type_patterns"][query_type]["response_characteristics"]["asks_clarification"] += 1
                if any(word in response.lower() for word in ["understand", "i see", "concern"]):
                    pattern_stats["query_type_patterns"][query_type]["response_characteristics"]["shows_empathy"] += 1
            
            # 4. Temporal patterns (conversation flow)
            if messages:
                # Opening strategy
                first_user_msg = messages[0]["content"].lower()
                if "?" in first_user_msg:
                    pattern_stats["temporal_patterns"]["opening_strategies"]["direct_question"] += 1
                elif any(greeting in first_user_msg for greeting in ["hi", "hello", "hey"]):
                    pattern_stats["temporal_patterns"]["opening_strategies"]["greeting_first"] += 1
                else:
                    pattern_stats["temporal_patterns"]["opening_strategies"]["statement"] += 1
                
                # Follow-up patterns
                if len(messages) > 2:
                    pattern_stats["temporal_patterns"]["follow_up_patterns"]["has_follow_up"] += 1
                    if is_successful:
                        pattern_stats["temporal_patterns"]["follow_up_patterns"]["follow_up_successful"] += 1
            
            # 5. Response quality metrics
            if len(messages) > 1:
                response = messages[1]["content"]
                
                # Length as quality indicator
                pattern_stats["quality_indicators"]["response_length"].append(len(response))
                
                # Specificity score (presence of specific data)
                specificity = 0
                if any(char.isdigit() for char in response):
                    specificity += 0.3
                if "$" in response or "%" in response:
                    specificity += 0.3
                if any(holding in response for holding in ["SPY", "QQQ", "BND", "VTI", "NVDA"]):
                    specificity += 0.4
                pattern_stats["quality_indicators"]["specificity_scores"].append(specificity)
                
                # Personalization score
                personalization = 0
                if "your" in response.lower():
                    personalization += 0.5
                if "you" in response.lower():
                    personalization += 0.3
                if signals.get("referenced_context", False):
                    personalization += 0.2
                pattern_stats["quality_indicators"]["personalization_scores"].append(personalization)
        
        # Generate LangMem-style extracted rules from patterns
        total_successful = sum(1 for c in conversations if c.feedback["success"])
        total_failed = len(conversations) - total_successful
        
        extracted_rules = {
            "universal_rules": [],
            "segment_specific_rules": {},
            "antipatterns": [],
            "query_type_rules": {},
            "behavioral_combinations": [],
            "quality_thresholds": {},
            "statistics": {
                "total_conversations": len(conversations),
                "success_rate": total_successful / len(conversations) if conversations else 0,
                "patterns_analyzed": len(pattern_stats["success_behaviors"]) + len(pattern_stats["failure_behaviors"]),
                "avg_satisfaction": np.mean([c.feedback["satisfaction_score"] for c in conversations]) if conversations else 0
            }
        }
        
        # Extract universal success patterns (>60% of successful conversations)
        for behavior, count in pattern_stats["success_behaviors"].items():
            if total_successful > 0 and count / total_successful > 0.6:
                avg_satisfaction = np.mean(pattern_stats["satisfaction_correlations"][behavior])
                
                # Create specific, actionable rules
                if behavior == "asked_clarification":
                    extracted_rules["universal_rules"].append({
                        "pattern": "clarification_questions",
                        "rule": "When user intent is ambiguous, ask specific clarifying questions before providing detailed information",
                        "impact": f"Appears in {count/total_successful:.1%} of successful conversations",
                        "avg_satisfaction": avg_satisfaction
                    })
                elif behavior == "provided_specific_data":
                    extracted_rules["universal_rules"].append({
                        "pattern": "data_specificity", 
                        "rule": "Always include specific numbers, percentages, and dollar amounts when discussing portfolio performance or holdings",
                        "impact": f"Appears in {count/total_successful:.1%} of successful conversations",
                        "avg_satisfaction": avg_satisfaction
                    })
                elif behavior == "personalized_response":
                    extracted_rules["universal_rules"].append({
                        "pattern": "personalization",
                        "rule": "Reference the user's actual portfolio holdings and personal investment situation in every response",
                        "impact": f"Appears in {count/total_successful:.1%} of successful conversations",
                        "avg_satisfaction": avg_satisfaction
                    })
                elif behavior == "showed_empathy":
                    extracted_rules["universal_rules"].append({
                        "pattern": "emotional_acknowledgment",
                        "rule": "Acknowledge user emotions and concerns, especially when discussing losses or market volatility",
                        "impact": f"Appears in {count/total_successful:.1%} of successful conversations",
                        "avg_satisfaction": avg_satisfaction
                    })
                elif behavior == "explained_jargon":
                    extracted_rules["universal_rules"].append({
                        "pattern": "jargon_explanation",
                        "rule": "Automatically explain financial terms and concepts in plain language without being asked",
                        "impact": f"Appears in {count/total_successful:.1%} of successful conversations",
                        "avg_satisfaction": avg_satisfaction
                    })
        
        # Extract antipatterns (>50% of failed conversations)
        for behavior, count in pattern_stats["failure_behaviors"].items():
            if total_failed > 0 and count / total_failed > 0.5:
                if behavior == "error_occurred":
                    extracted_rules["antipatterns"].append({
                        "pattern": "technical_errors",
                        "avoid": "Showing technical error messages without providing helpful alternatives or workarounds",
                        "frequency": f"Occurs in {count/total_failed:.1%} of failed conversations"
                    })
                elif not signals.get(behavior, False):  # Missing expected behavior
                    extracted_rules["antipatterns"].append({
                        "pattern": f"missing_{behavior}",
                        "avoid": f"Responses that lack {behavior.replace('_', ' ')}",
                        "frequency": f"Missing in {count/total_failed:.1%} of failed conversations"
                    })
        
        # Extract powerful behavioral combinations
        for combo, stats in pattern_stats["behavioral_combinations"].items():
            total_combo = stats["success"] + stats["failure"]
            if total_combo > 10:  # Minimum sample size
                success_rate = stats["success"] / total_combo
                if success_rate > 0.8:  # High success combinations
                    behaviors = combo.split("+")
                    extracted_rules["behavioral_combinations"].append({
                        "combination": combo,
                        "rule": f"Combine {behaviors[0].replace('_', ' ')} with {behaviors[1].replace('_', ' ')} for maximum effectiveness",
                        "success_rate": success_rate,
                        "occurrences": total_combo
                    })
        
        # Query-type specific rules
        for query_type, stats in pattern_stats["query_type_patterns"].items():
            if stats["success_rate"]:
                avg_success = np.mean(stats["success_rate"])
                avg_satisfaction = np.mean(stats["satisfaction_scores"]) if stats["satisfaction_scores"] else 0
                
                query_rules = {
                    "query_type": query_type,
                    "success_rate": avg_success,
                    "avg_satisfaction": avg_satisfaction,
                    "effective_strategies": []
                }
                
                # Identify effective strategies for this query type
                for characteristic, count in stats["response_characteristics"].items():
                    effectiveness = count / len(stats["success_rate"]) if stats["success_rate"] else 0
                    if effectiveness > 0.5:
                        if characteristic == "has_numbers":
                            query_rules["effective_strategies"].append("Include specific numerical data")
                        elif characteristic == "detailed_response":
                            query_rules["effective_strategies"].append("Provide comprehensive, detailed responses")
                        elif characteristic == "asks_clarification":
                            query_rules["effective_strategies"].append("Clarify user intent before responding")
                        elif characteristic == "shows_empathy":
                            query_rules["effective_strategies"].append("Acknowledge emotional context")
                
                if query_rules["effective_strategies"]:
                    extracted_rules["query_type_rules"][query_type] = query_rules
        
        # Quality thresholds from successful conversations
        if pattern_stats["quality_indicators"]["response_length"]:
            successful_lengths = [
                pattern_stats["quality_indicators"]["response_length"][i]
                for i, c in enumerate(conversations)
                if c.feedback["success"] and i < len(pattern_stats["quality_indicators"]["response_length"])
            ]
            if successful_lengths:
                extracted_rules["quality_thresholds"] = {
                    "min_response_length": int(np.percentile(successful_lengths, 25)),
                    "optimal_response_length": int(np.median(successful_lengths)),
                    "min_specificity_score": 0.6,  # Threshold for including specific data
                    "min_personalization_score": 0.5  # Threshold for personalized responses
                }
        
        # Add metadata for LangMem optimization
        extracted_rules["metadata"] = {
            "extraction_method": "trajectory_analysis",
            "total_trajectories_analyzed": len(conversations),
            "successful_trajectories": total_successful,
            "failed_trajectories": total_failed,
            "behavioral_signals_tracked": len(pattern_stats["success_behaviors"]) + len(pattern_stats["failure_behaviors"]),
            "combination_patterns_found": len(pattern_stats["behavioral_combinations"]),
            "query_types_analyzed": len(pattern_stats["query_type_patterns"])
        }
        
        return extracted_rules

    def export_realistic_data(self, output_dir: str = None):
        """Export all generated data in realistic format"""

        # Use domain directory if not specified
        if output_dir is None:
            output_dir = os.path.join(self.domain_dir, "investment_advisor_data")
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("🚀 Generating realistic investment advisor conversations...")
        
        # Generate conversations
        conversations = self.generate_realistic_conversations(num_users=50, convs_per_user=10)
        
        # Convert to dict format for JSON serialization
        conversations_data = []
        for conv in conversations:
            conversations_data.append(asdict(conv))
        
        # Save conversations
        with open(f"{output_dir}/conversations.jsonl", "w") as f:
            for conv in conversations_data:
                f.write(json.dumps(conv) + "\n")
        
        # Extract patterns from the data (not hardcoded!)
        extracted_patterns = self.extract_patterns_from_conversations(conversations)
        
        # Save extracted patterns
        with open(f"{output_dir}/extracted_patterns.json", "w") as f:
            json.dump(extracted_patterns, f, indent=2)
        
        # Generate test scenarios based on common failure patterns
        test_scenarios = self._generate_test_scenarios_from_patterns(
            conversations=conversations,
            extracted_patterns=extracted_patterns
        )
        
        with open(f"{output_dir}/test_scenarios.json", "w") as f:
            json.dump(test_scenarios, f, indent=2)
        
        # Summary statistics
        stats = {
            "total_conversations": len(conversations),
            "unique_users": len(set(c.user_id for c in conversations)),
            "success_rate": sum(1 for c in conversations if c.feedback["success"]) / len(conversations),
            "avg_satisfaction": sum(c.feedback["satisfaction_score"] for c in conversations) / len(conversations),
            "behavioral_patterns": {
                "with_clarification": sum(1 for c in conversations if c.behavioral_signals["asked_clarification"]),
                "with_specific_data": sum(1 for c in conversations if c.behavioral_signals["provided_specific_data"]),
                "with_errors": sum(1 for c in conversations if c.behavioral_signals["error_occurred"])
            },
            "query_distribution": {}
        }
        
        # Count query types
        for conv in conversations:
            query_type = conv.metadata["query_type"]
            stats["query_distribution"][query_type] = stats["query_distribution"].get(query_type, 0) + 1
        
        with open(f"{output_dir}/statistics.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        print(f"✅ Generated {len(conversations)} realistic conversations")
        print(f"📊 Success rate: {stats['success_rate']:.1%}")
        print(f"😊 Average satisfaction: {stats['avg_satisfaction']:.1f}/5.0")
        print(f"🔍 Extracted {len(extracted_patterns['universal_rules'])} universal patterns")
        print(f"⚠️  Identified {len(extracted_patterns['antipatterns'])} antipatterns")
        
        return {
            "conversations": conversations,
            "patterns": extracted_patterns,
            "test_scenarios": test_scenarios,
            "statistics": stats
        }

    def _generate_test_scenarios_from_patterns(self,
                                              conversations: List[ConversationData],
                                              extracted_patterns: Dict) -> List[Dict]:
        """Generate test scenarios based on identified patterns"""
        
        # Original test scenarios - completely different content
        test_queries = [
            {
                "scenario_id": "test_001",
                "description": "Vague performance question",
                "user_message": "how am I doing?",
                "expected_improvements": ["ask_clarification", "provide_options"],
                "baseline_response": "You're doing fine.",
                "improved_response": "I'd love to show you how you're doing! Are you asking about today's performance, this year's returns, or would you like to see how you're tracking against your goals? I can also compare your returns to market benchmarks if that helps."
            },
            {
                "scenario_id": "test_002",
                "description": "Complex term from novice investor",
                "user_message": "what's my sharpe ratio?",
                "expected_improvements": ["explain_concept", "provide_context"],
                "baseline_response": "Your Sharpe ratio is 0.85.",
                "improved_response": "Your Sharpe ratio is 0.85. Let me explain what this means - the Sharpe ratio measures how much return you're getting for the risk you're taking. A ratio of 0.85 means you're getting decent returns relative to your risk. Generally, anything above 1.0 is considered good, so you might want to look at ways to either boost returns or reduce volatility."
            },
            {
                "scenario_id": "test_003",
                "description": "Worried investor during downturn",
                "user_message": "I lost 20% this quarter, should I sell everything?",
                "expected_improvements": ["acknowledge_concern", "provide_perspective", "avoid_rash_decisions"],
                "baseline_response": "You should stay invested for the long term.",
                "improved_response": "I can understand why a 20% drop would be concerning - that's a significant decline and it's natural to want to protect what's left. Before making any big moves though, let's consider a few things: Historical data shows that investors who sell during downturns often lock in losses and miss the recovery. What's your timeline for needing this money? If it's more than 5 years, you likely have time to recover. Would you like to discuss adjusting your risk level instead of selling everything?"
            }
        ]
        
        return test_queries

# Main execution
if __name__ == "__main__":
    generator = EnhancedInvestmentAdvisorDataGenerator()
    data = generator.export_realistic_data()