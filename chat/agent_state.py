"""
Agent State Definition for LangGraph
"""
from typing import TypedDict, List, Dict, Optional, Any, Annotated

class AgentState(TypedDict):
    """State that flows through the LangGraph nodes"""
    
    # Input
    question: str
    
    # Analysis results
    intent: str
    entities: List[str]
    search_terms: List[str]
    dietary_restrictions: List[str]
    
    # Retrieved data
    retrieved_products: List[Dict]
    product_count: int
    
    # Reasoning and answer
    reasoning: str
    answer: str
    
    # Quality and metadata
    quality_score: int
    needs_clarification: bool
    clarification_question: str
    
    # For UI
    reasoning_steps: List[str]
    suggested_products: List[str]
    
    # Error handling
    error: Optional[str]
    retry_count: int

def create_initial_state(question: str) -> AgentState:
    """Create initial state for the agent"""
    return {
        "question": question,
        "intent": "",
        "entities": [],
        "search_terms": [],
        "dietary_restrictions": [],
        "retrieved_products": [],
        "product_count": 0,
        "reasoning": "",
        "answer": "",
        "quality_score": 0,
        "needs_clarification": False,
        "clarification_question": "",
        "reasoning_steps": [],
        "suggested_products": [],
        "error": None,
        "retry_count": 0
    }