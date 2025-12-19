"""
Report Designer Agent Module.

This agent is responsible for:
- Calculating metrics from the refined dataset
- Gathering news articles from state
- Rendering the final HTML report
"""
from typing import Dict, Any

import pandas as pd

from app.agents.states import WorkflowState
from app.config.settings import settings
from app.tools.report_tools import render_report, cleanup_old_reports
from app.utils.logging import get_logger

logger = get_logger("agents.report_designer")


def _calculate_metrics(dataset_path: str) -> Dict[str, Any]:
    """
    Calculate metrics from the refined dataset.
    
    Args:
        dataset_path: Path to the refined CSV file
        
    Returns:
        Dictionary of calculated metrics
    """
    metrics: Dict[str, Any] = {
        "total_cases": 0,
        "error": None,
    }
    
    try:
        df = pd.read_csv(dataset_path)
        metrics["total_cases"] = len(df)
        
        # Add more metrics as needed
        if "SG_UF_NOT" in df.columns:
            metrics["states_affected"] = df["SG_UF_NOT"].nunique()
        
        if "HOSPITAL" in df.columns:
            hospitalized = df["HOSPITAL"].apply(
                lambda x: str(x).strip().upper() in ["1", "SIM", "S"]
            ).sum()
            metrics["hospitalized_count"] = int(hospitalized)
        
        if "UTI" in df.columns:
            uti = df["UTI"].apply(
                lambda x: str(x).strip().upper() in ["1", "SIM", "S"]
            ).sum()
            metrics["uti_count"] = int(uti)
        
        if "EVOLUCAO" in df.columns:
            # EVOLUCAO: 1=Cura, 2=Óbito, 3=Óbito por outras causas, 9=Ignorado
            deaths = df["EVOLUCAO"].apply(
                lambda x: str(x).strip() in ["2", "3"]
            ).sum()
            metrics["death_count"] = int(deaths)
        
        logger.debug(f"Calculated metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Failed to calculate metrics: {e}")
        metrics["error"] = str(e)
    
    return metrics


def creation_step(state: WorkflowState) -> Dict[str, Any]:
    """
    Create the final HTML report.
    
    Gathers data from state, calculates metrics, and renders the report.
    
    Args:
        state: Current workflow state
        
    Returns:
        State updates with final_report_path
    """
    logger.info("=== Report Designer: Creation Step ===")
    
    updates: Dict[str, Any] = {}
    
    refined_path = state.get("refined_dataset_path")
    news_articles = state.get("news_articles", [])
    topic = state.get("topic") or settings.default_topic
    errors = state.get("errors", [])
    
    logger.info(f"Dataset: {refined_path or 'N/A'}")
    logger.info(f"News articles: {len(news_articles)}")
    
    # 1. Calculate metrics
    metrics = {"total_cases": 0}
    
    if refined_path:
        metrics = _calculate_metrics(refined_path)
        logger.info(f"Metrics: total_cases={metrics.get('total_cases', 0)}")
    else:
        logger.warning("No refined dataset available for metrics calculation")
    
    # 2. Prepare report data
    report_data = {
        "refined_dataset_path": refined_path or "N/A",
        "metrics": metrics,
        "news_articles": news_articles,
        "topic": topic,
        "errors": errors,
    }
    
    # 3. Render report
    logger.info("Rendering HTML report...")
    
    try:
        report_path = render_report.invoke({"data": report_data})
        
        if report_path:
            updates["final_report_path"] = report_path
            logger.info(f"=== Report created: {report_path} ===")
            
            # Cleanup old reports
            cleanup_old_reports(max_reports=3)
        else:
            error_msg = "Report rendering failed - no path returned"
            logger.error(error_msg)
            updates["errors"] = [error_msg]
            
    except Exception as e:
        error_msg = f"Report creation failed: {e}"
        logger.error(error_msg)
        updates["errors"] = [error_msg]
    
    return updates
