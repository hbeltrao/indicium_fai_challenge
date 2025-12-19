"""
Main Workflow Module.

This module defines the LangGraph workflow that orchestrates:
1. Data Pipeline: Download -> Validate -> Clean SRAG data
2. News Pipeline: Search -> Scrape -> Curate news articles
3. Report Pipeline: Calculate metrics -> Render HTML report

The workflow uses parallel execution for data and news pipelines,
then joins at the report generation step.
"""
from langgraph.graph import StateGraph, END

from app.agents import data_specialist, news_curator, report_designer
from app.agents.states import WorkflowState
from app.config.settings import settings
from app.utils.logging import get_logger

logger = get_logger("workflow")


def _start_step(state: WorkflowState) -> dict:
    """
    Initialize workflow state with defaults.
    
    This is the entry point that sets up initial state values
    before the parallel pipelines begin.
    """
    logger.info("=" * 60)
    logger.info("WORKFLOW START")
    logger.info("=" * 60)
    
    return {
        "errors": state.get("errors", []),
        "topic": state.get("topic", settings.default_topic),
        "metadata_valid": False,
        "news_articles": [],
        "metrics": {},
    }


def _join_step(state: WorkflowState) -> dict:
    """
    Synchronization point after parallel pipelines complete.
    
    This step logs the results of both pipelines before proceeding
    to report generation. It doesn't modify state, just passes through.
    """
    logger.info("-" * 40)
    logger.info("PIPELINES JOINED")
    logger.info("-" * 40)
    logger.info(f"  Data: {state.get('refined_dataset_path', 'N/A')}")
    logger.info(f"  News: {len(state.get('news_articles', []))} articles")
    logger.info(f"  Errors: {len(state.get('errors', []))}")
    
    # Pass through - no state changes
    return {}


def _end_step(state: WorkflowState) -> dict:
    """
    Final step to log workflow completion.
    """
    logger.info("=" * 60)
    logger.info("WORKFLOW COMPLETE")
    logger.info(f"Report: {state.get('final_report_path', 'Not generated')}")
    if state.get("errors"):
        logger.warning(f"Errors encountered: {state['errors']}")
    logger.info("=" * 60)
    
    return {}


def _build_workflow() -> StateGraph:
    """
    Build and configure the workflow graph.
    
    Graph Structure:
        start_node
           ├──> download_node ──> process_data_node ──┐
           └──> fetch_news_node ─────────────────────>├──> join_node ──> report_node ──> end_node
           
    Returns:
        Configured StateGraph ready for compilation
    """
    workflow = StateGraph(WorkflowState)
    
    # === Add Nodes ===
    
    # Control flow nodes
    workflow.add_node("start_node", _start_step)
    workflow.add_node("join_node", _join_step)
    workflow.add_node("end_node", _end_step)
    
    # Data pipeline nodes
    workflow.add_node("download_node", data_specialist.download_step)
    workflow.add_node("process_data_node", data_specialist.processing_step)
    
    # News pipeline node
    workflow.add_node("fetch_news_node", news_curator.curation_step)
    
    # Report node
    workflow.add_node("report_node", report_designer.creation_step)
    
    # === Define Edges ===
    
    # Entry point
    workflow.set_entry_point("start_node")
    
    # Fork: start splits into parallel data and news pipelines
    workflow.add_edge("start_node", "download_node")
    workflow.add_edge("start_node", "fetch_news_node")
    
    # Data pipeline: download -> process -> join
    workflow.add_edge("download_node", "process_data_node")
    workflow.add_edge("process_data_node", "join_node")
    
    # News pipeline: fetch -> join
    workflow.add_edge("fetch_news_node", "join_node")
    
    # Join -> Report -> End
    workflow.add_edge("join_node", "report_node")
    workflow.add_edge("report_node", "end_node")
    workflow.add_edge("end_node", END)
    
    return workflow


# Build and compile the workflow
_workflow = _build_workflow()
app = _workflow.compile()

# For debugging: generate graph visualization
def get_mermaid_diagram() -> str:
    """Generate Mermaid diagram of the workflow."""
    return app.get_graph().draw_mermaid()


def run_workflow(initial_state: dict = None) -> dict:
    """
    Run the workflow with optional initial state.
    
    Args:
        initial_state: Optional dict with initial values (e.g., {"topic": "Dengue"})
        
    Returns:
        Final workflow state dictionary
    """
    if initial_state is None:
        initial_state = {"errors": []}
    
    return app.invoke(initial_state)