# coala_agent.py
"""
Consolidated CoALA Agent with Episodic, Semantic, and Procedural Memory
Domain-agnostic implementation that works with any DomainAgent
"""

import json
from datetime import datetime
from typing import List, Dict, Optional, TypedDict, Annotated, Sequence
from collections import defaultdict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from domain_agent import DomainAgent, DomainProcedure
from procedural_memory import ProceduralMemory


class SemanticFact(BaseModel):
    """Structure for semantic memory facts"""
    subject: str = Field(description="Entity or topic")
    predicate: str = Field(description="Relationship or property")
    object: str = Field(description="Value or related entity")
    confidence: float = Field(description="Confidence score 0-1")
    source: str = Field(description="Source: user or assistant")


class AgentState(TypedDict):
    """State structure for the agent workflow"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    working_memory: dict
    episodic_recall: list
    semantic_facts: dict
    procedural_strategy: dict
    user_id: str
    conversation_id: str


class CoALAAgent:
    """
    Full CoALA agent with episodic, semantic, and procedural memory.
    Works with any domain by accepting a DomainAgent.
    """
    
    def __init__(self, 
                domain_agent: DomainAgent,
                model_name: str = "gpt-4.1-mini",  # Changed from gpt-4.1-mini
                temperature: float = 0,
                persist_directory: str = None):
        
        self.domain_agent = domain_agent
        
        # Use domain-specific directory if not provided
        if persist_directory is None and hasattr(domain_agent, 'memory_dir'):
            persist_directory = domain_agent.memory_dir
        elif persist_directory is None:
            persist_directory = "./memory_store"
        
        # Initialize LLM and embeddings
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.embeddings = OpenAIEmbeddings()
        self.output_parser = StrOutputParser()
        
        # Initialize vector store for episodic and semantic memories
        self.vector_store = Chroma(
            collection_name="agent_memory",
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        
        # Initialize procedural memory with domain agent
        self.procedural_memory = ProceduralMemory(
            llm=self.llm, 
            domain_agent=domain_agent
        )
        
        # Build the workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
        
        # Track current state
        self.current_user_id = "default"
        self.current_conversation_id = None
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for the agent"""
        workflow = StateGraph(AgentState)
        workflow.add_node("memory_agent", self._unified_memory_agent)
        workflow.set_entry_point("memory_agent")
        workflow.add_edge("memory_agent", END)
        return workflow
    
    # ============== Episodic Memory Methods ==============
    
    def store_episodic_memory(self, 
                             conversation_id: str, 
                             messages: List,
                             summary: Optional[str] = None) -> str:
        """Store a conversation in episodic memory"""
        if not summary and messages:
            first_msg = messages[0]
            if isinstance(first_msg, tuple):
                summary = f"Discussion: {first_msg[1][:100]}..."
            else:
                summary = f"Discussion: {first_msg.content[:100]}..."
        
        metadata = {
            "type": "episodic",
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "message_count": len(messages),
            "user_id": self.current_user_id
        }
        
        conversation_text = self._format_messages(messages)
        self.vector_store.add_documents([
            Document(page_content=conversation_text, metadata=metadata)
        ])
        return conversation_id
    
    def retrieve_episodic_memories(self, query: str, k: int = 3) -> List[Document]:
        """Retrieve relevant episodic memories"""
        return self.vector_store.similarity_search(
            query=query, 
            k=k, 
            filter={"type": {"$eq": "episodic"}}
        )
    
    # ============== Semantic Memory Methods ==============
    
    def extract_semantic_facts(self, messages: List) -> List[SemanticFact]:
        """Extract semantic facts from a conversation"""
        # Use domain-specific extraction prompt
        extraction_prompt_template = self.domain_agent.get_semantic_extraction_prompt()
        extraction_prompt = PromptTemplate.from_template(extraction_prompt_template)
        
        conversation_text = self._format_messages(messages)
        
        try:
            chain = extraction_prompt | self.llm | JsonOutputParser()
            result = chain.invoke({"conversation": conversation_text})
            
            facts = []
            for fact_dict in result.get("facts", []):
                # Fix the source field if it's a boolean
                if "source" in fact_dict:
                    if isinstance(fact_dict["source"], bool):
                        # Convert boolean to appropriate string
                        fact_dict["source"] = "assistant" if fact_dict["source"] else "user"
                    elif fact_dict["source"] not in ["user", "assistant"]:
                        # Default to "assistant" for any other invalid values
                        fact_dict["source"] = "assistant"
                
                # Ensure all required fields are strings
                for field in ["subject", "predicate", "object"]:
                    if field in fact_dict and not isinstance(fact_dict[field], str):
                        fact_dict[field] = str(fact_dict[field])
                
                try:
                    facts.append(SemanticFact(**fact_dict))
                except Exception as e:
                    # Skip invalid facts
                    print(f"Skipping invalid fact: {e}")
                    continue
                    
            return facts
        except Exception as e:
            print(f"Fact extraction error: {e}")
            return []
    
    def store_semantic_facts(self, 
                            facts: List[SemanticFact],
                            user_id: Optional[str] = None) -> int:
        """Store semantic facts in memory"""
        if user_id is None:
            user_id = self.current_user_id
            
        documents = []
        for fact in facts:
            documents.append(Document(
                page_content=f"{fact.subject} {fact.predicate} {fact.object}",
                metadata={
                    "type": "semantic",
                    "user_id": user_id,
                    "subject": fact.subject,
                    "predicate": fact.predicate,
                    "object": fact.object,
                    "confidence": fact.confidence,
                    "timestamp": datetime.now().isoformat()
                }
            ))
        
        if documents:
            self.vector_store.add_documents(documents)
        return len(documents)
    
    def retrieve_semantic_facts(self, 
                               query: str,
                               user_id: Optional[str] = None,
                               k: int = 5) -> List[Dict]:
        """Retrieve relevant semantic facts"""
        if user_id is None:
            user_id = self.current_user_id
            
        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter={
                "$and": [
                    {"type": {"$eq": "semantic"}},
                    {"user_id": {"$eq": user_id}}
                ]
            }
        )
        
        return [{
            "subject": doc.metadata.get("subject"),
            "predicate": doc.metadata.get("predicate"),
            "object": doc.metadata.get("object"),
            "confidence": doc.metadata.get("confidence", 1.0)
        } for doc in results]
    
    # ============== Helper Methods ==============
    
    def _format_messages(self, messages: List) -> str:
        """Format messages into a readable string"""
        conversation_text = ""
        for msg in messages:
            if isinstance(msg, tuple):
                conversation_text += f"{msg[0]}: {msg[1]}\n"
            elif isinstance(msg, BaseMessage):
                conversation_text += f"{msg.type}: {msg.content}\n"
            else:
                conversation_text += str(msg) + "\n"
        return conversation_text
    
    def _format_semantic_context(self, facts: List[Dict]) -> str:
        """Format semantic facts into context string"""
        if not facts:
            return "No relevant facts found."
        
        context = "Known information:\n"
        for fact in facts:
            if fact.get('confidence', 1.0) > 0.7:
                context += f"- {fact['subject']} {fact['predicate']} {fact['object']}\n"
        return context
    
    def _unified_memory_agent(self, state: AgentState) -> dict:
        """Main agent logic combining all memory types"""
        current_messages = state.get("messages", [])
        user_id = state.get("user_id", self.current_user_id)
        conversation_id = state.get("conversation_id", 
                                   f"conv_{datetime.now().timestamp()}")
        
        episodic_context = ""
        semantic_context = ""
        procedural_context = ""
        
        if current_messages:
            # Get latest query
            latest_msg = current_messages[-1]
            if isinstance(latest_msg, tuple):
                latest_query = latest_msg[1]
            elif isinstance(latest_msg, BaseMessage):
                latest_query = latest_msg.content
            else:
                latest_query = str(latest_msg)
            
            # Retrieve episodic memories
            past_episodes = self.retrieve_episodic_memories(latest_query, k=2)
            if past_episodes:
                episodic_context = "Relevant past discussions:\n"
                for episode in past_episodes:
                    timestamp = episode.metadata.get('timestamp', 'Unknown')
                    episodic_context += f"[{timestamp}]:\n{episode.page_content[:200]}...\n\n"
            
            # Retrieve semantic facts
            facts = self.retrieve_semantic_facts(latest_query, user_id=user_id, k=5)
            semantic_context = self._format_semantic_context(facts)
            
            # Extract profile and retrieve procedural strategy
            profile = self.domain_agent.extract_profile(facts, latest_query)
            strategy = self.procedural_memory.get_investment_strategy(
                latest_query, 
                profile,
                user_id=user_id
            )
            
            if strategy:
                procedural_context = self.domain_agent.format_procedural_context(strategy)
        
        # Generate response with all memory contexts
        response_prompt_template = self.domain_agent.get_response_prompt_template()
        response_prompt = PromptTemplate.from_template(response_prompt_template)
        
        formatted_messages = self._format_messages(
            current_messages[-5:] if current_messages else []
        )
        
        chain = response_prompt | self.llm | self.output_parser
        response = chain.invoke({
            "semantic_context": semantic_context,
            "episodic_context": episodic_context,
            "procedural_context": procedural_context,
            "messages": formatted_messages
        })
        
        # Store memories after generating response
        if len(current_messages) >= 2:
            self.store_episodic_memory(conversation_id, current_messages)
        
        # Extract and store semantic facts
        if current_messages:
            messages_with_response = current_messages + [("assistant", response)]
            new_facts = self.extract_semantic_facts(messages_with_response[-3:])
            if new_facts:
                stored = self.store_semantic_facts(new_facts, user_id)
                state["semantic_facts"] = {"extracted": stored}
        
        # Learn from this interaction (procedural memory)
        if current_messages and latest_query:
            interaction_data = {
                "messages": [str(m) for m in messages_with_response[-3:]],
                "success": True,
                "client_satisfaction": 8,
                "profile": profile
            }
            
            learning_result = self.procedural_memory.learn_from_interaction(
                latest_query,
                interaction_data,
                user_id=user_id,
                user_profile=profile
            )
        
        return {
            "messages": [AIMessage(content=response)],
            "episodic_recall": past_episodes if past_episodes else [],
            "semantic_facts": state.get("semantic_facts", {}),
            "procedural_strategy": strategy if strategy else {}
        }
    
    # ============== Public Interface ==============
    
    def process_message(self, 
                       message: str,
                       user_id: Optional[str] = None,
                       conversation_id: Optional[str] = None) -> str:
        """Process a single message and return the response"""
        if user_id:
            self.current_user_id = user_id
        if conversation_id:
            self.current_conversation_id = conversation_id
        else:
            self.current_conversation_id = f"conv_{datetime.now().timestamp()}"
        
        state = {
            "messages": [HumanMessage(content=message)],
            "user_id": self.current_user_id,
            "conversation_id": self.current_conversation_id,
            "working_memory": {},
            "episodic_recall": [],
            "semantic_facts": {},
            "procedural_strategy": {}
        }
        
        result = self.app.invoke(state)
        
        if result and "messages" in result and result["messages"]:
            return result["messages"][-1].content
        return "I apologize, but I couldn't process that message."
    
    def get_memory_stats(self) -> Dict:
        """Get comprehensive statistics about all memory systems"""
        procedural_stats = self.procedural_memory.get_stats()
        
        return {
            "episodic_semantic": {
                "total_documents": len(self.vector_store.get()["ids"]) if hasattr(self.vector_store, 'get') else "N/A",
                "current_user": self.current_user_id,
                "current_conversation": self.current_conversation_id
            },
            "procedural": procedural_stats
        }