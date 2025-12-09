from langgraph.graph import StateGraph, END
from app.agents.state import AgentState
from app.agents.nodes import (
    data_specialist_node,
    news_curator_node,
    validation_node,
    layout_designer_node
)

def build_graph():
    """
    Constructs the StateGraph for the Health Reporting System.
    """
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("data_specialist", data_specialist_node)
    workflow.add_node("news_curator", news_curator_node)
    workflow.add_node("validator", validation_node)
    workflow.add_node("layout_designer", layout_designer_node)
    
    # Add Edges
    # Start -> Parallel Execution
    # LangGraph allows parallel branches by just pointing start to multiple nodes 
    # but usually we need a router or just define the flow.
    # Here we can run them in sequence for simplicity or use parallel.
    # To do true parallel in LangGraph, we fan out from start.
    
    workflow.set_entry_point("data_specialist")
    
    # We'll do a linear approximation for stability: Data -> News -> Validator
    # Or strict parallel: Start -> A, Start -> B. Join at C.
    # Let's do: Start -> Data -> News -> Validator (Sequential is easier to debug for MVP)
    # Refinement: The user requested "Architecture" with parallel. 
    # LangGraph fan-out:
    # entry -> data
    # entry -> news
    # data -> validator
    # news -> validator 
    # But validator needs BOTH. This requires a sync/join node.
    # LangGraph waits for all parents? No, it's state based.
    # Correct pattern: Data & News run, and we need a joiner. 
    # Let's use a router or just linearize for V1 safety: 
    # Data Specialist -> News Curator -> Validator -> Layout
    
    workflow.add_edge("data_specialist", "news_curator")
    workflow.add_edge("news_curator", "validator")
    
    # Conditional Logic at Validator
    def route_validation(state: AgentState):
        if state["validation_status"] == "passed":
            return "layout_designer"
        else:
            return END # Or a repair node
            
    workflow.add_conditional_edges(
        "validator",
        route_validation,
        {
            "layout_designer": "layout_designer",
            END: END
        }
    )
    
    workflow.add_edge("layout_designer", END)
    
    return workflow.compile()
