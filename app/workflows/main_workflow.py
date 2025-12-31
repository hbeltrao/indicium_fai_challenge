"""
Main Workflow Module.

This module defines the LangGraph workflow that orchestrates:
1. Data Pipeline: Download -> Validate -> Clean SRAG data
2. News Pipeline: Search -> Scrape -> Curate news articles
3. Report Pipeline: Calculate metrics -> Render HTML report

The workflow uses parallel execution for data and news pipelines,
then joins at the report generation step.
"""
from langgraph.graph import StateGraph, END, START

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
    Synchronization barrier after parallel pipelines.
    
    This step ensures both pipelines have completed by tracking their status.
    Only proceeds to report generation when both data and news are ready.
    
    Since this node receives edges from both pipelines, it will be called
    twice. We use state flags to detect when both have completed.
    """
    # Check what just completed
    has_data = bool(state.get('refined_dataset_path'))
    has_news = len(state.get('news_articles', [])) > 0
    
    # Check what was already marked complete
    already_data_complete = state.get('data_complete', False)
    already_news_complete = state.get('news_complete', False)
    
    # Mark what just completed
    updates = {}
    if has_data:
        updates['data_complete'] = True
    if has_news:
        updates['news_complete'] = True
    
    # Determine if BOTH are complete (accounting for this update)
    data_will_be_complete = already_data_complete or has_data
    news_will_be_complete = already_news_complete or has_news
    both_complete = data_will_be_complete and news_will_be_complete
    
    if not both_complete:
        logger.info("-" * 40)
        logger.info("PIPELINE SYNCHRONIZATION")
        logger.info("-" * 40)
        logger.info(f"  Data Ready: {data_will_be_complete}")
        logger.info(f"  News Ready: {news_will_be_complete}")
        logger.info(f"  Waiting for both to complete...")
    else:
        logger.info("-" * 40)
        logger.info("PIPELINES JOINED - Both Complete")
        logger.info("-" * 40)
        logger.info(f"  Data: {state.get('refined_dataset_path', 'N/A')}")
        logger.info(f"  News: {len(state.get('news_articles', []))} articles")
        logger.info(f"  Errors: {len(state.get('errors', []))}")
        # Mark as ready to proceed
        updates['ready_for_report'] = True
    
    return updates


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
        START
           ├──> download_node ──> process_data_node ──┐
           └──> fetch_news_node ─────────────────────┴──> report_node ──> END
           
    LangGraph automatically synchronizes the parallel branches at report_node
    using state merging, ensuring it executes only once after both branches complete.
           
    Returns:
        Configured StateGraph ready for compilation
    """
    workflow = StateGraph(WorkflowState)
    
    # === Add Nodes ===
    
    # Join/barrier node
    workflow.add_node("join_node", _join_step)
    
    # Data pipeline nodes
    workflow.add_node("download_node", data_specialist.download_step)
    workflow.add_node("process_data_node", data_specialist.processing_step)
    
    # News pipeline node
    workflow.add_node("fetch_news_node", news_curator.curation_step)
    
    # Report node
    workflow.add_node("report_node", report_designer.creation_step)
    
    # === Define Edges ===
    
    # Fork: START splits into parallel data and news pipelines
    workflow.add_edge(START, "download_node")
    workflow.add_edge(START, "fetch_news_node")
    
    # Data pipeline: download -> process -> join
    workflow.add_edge("download_node", "process_data_node")
    workflow.add_edge("process_data_node", "join_node")
    
    # News pipeline: fetch -> join
    workflow.add_edge("fetch_news_node", "join_node")
    
    # Conditional edge from join: only proceed to report when BOTH are complete
    def should_generate_report(state: WorkflowState) -> str:
        """Router: check if both pipelines completed before generating report."""
        has_data = bool(state.get('refined_dataset_path'))
        has_news = len(state.get('news_articles', [])) > 0  
        both_ready = has_data and has_news
        
        logger.debug(f"Router check: data={has_data}, news={has_news}, proceeding={both_ready}")
        return "report_node" if both_ready else END
    
    workflow.add_conditional_edges(
        "join_node",
        should_generate_report
    )
    
    # Report -> End
    workflow.add_edge("report_node", END)
    
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