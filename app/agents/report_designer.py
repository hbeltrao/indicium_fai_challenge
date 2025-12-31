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
    
    Metrics:
    - Summary Cards: Filtered to current month (Dec 2025)
    - Monthly history: Last 12 months for bar graph
    
    Args:
        dataset_path: Path to the refined CSV file
        
    Returns:
        Dictionary of calculated metrics
    """
    metrics: Dict[str, Any] = {
        "total_notified": 0,
        "confirmed_cases": 0,
        "notifications_increase_rate": 0.0,
        "cases_increase_rate": 0.0,
        "mortality_rate": 0.0,
        "hospitalization_rate": 0.0,
        "icu_rate": 0.0,
        "states": [],
        "monthly_history": [], # For bar graph: [{"label": "Jan/25", "value": 100}, ...]
        "daily_history": [], # For line graph: [{"label": "01/12", "value": 10}, ...]
        "datasource_url": "https://opendatasus.saude.gov.br/dataset/srag-2024-e-2025",
        "error": None,
    }
    
    try:
        df = pd.read_csv(dataset_path)
        
        # Parse dates
        if "DT_NOTIFIC" in df.columns:
            df["DT_NOTIFIC"] = pd.to_datetime(df["DT_NOTIFIC"], errors='coerce')
        
        if len(df) == 0:
            return metrics

        # Define time window for cards (Current Month)
        now = pd.Timestamp.now()
        current_month_start = now.replace(day=1, hour=0, minute=0, second=0)
        current_month_df = df[df["DT_NOTIFIC"] >= current_month_start]
        
        # Total notifications = row count for current month
        total_notifications = len(current_month_df)
        metrics["total_notified"] = total_notifications
        
        # --- Card Metrics (Current Month Only) ---
        
        # 1. Confirmed cases = row count where CLASSI_FIN is not null
        confirmed_cases = 0
        if "CLASSI_FIN" in current_month_df.columns:
            confirmed_cases = current_month_df["CLASSI_FIN"].notna().sum()
        metrics["confirmed_cases"] = confirmed_cases
        
        # 2. Mortality rate = deaths (EVOLUCAO=2 AND CLASSI_FIN not null) / confirmed_cases
        if confirmed_cases > 0 and "EVOLUCAO" in current_month_df.columns and "CLASSI_FIN" in current_month_df.columns:
            deaths_df = current_month_df[
                (current_month_df["EVOLUCAO"] == 2) &
                (current_month_df["CLASSI_FIN"].notna())
            ]
            deaths = len(deaths_df)
            metrics["mortality_rate"] = (deaths / confirmed_cases) * 100
            
        # 3. Hospitalization rate = (HOSPITAL=1) / total_notifications
        if total_notifications > 0 and "HOSPITAL" in current_month_df.columns:
            hospitalized = (current_month_df["HOSPITAL"] == 1).sum()
            metrics["hospitalization_rate"] = (hospitalized / total_notifications) * 100
            
        # 4. ICU rate = (UTI=1) / total_notifications
        if total_notifications > 0 and "UTI" in current_month_df.columns:
            icu = (current_month_df["UTI"] == 1).sum()
            metrics["icu_rate"] = (icu / total_notifications) * 100

        # --- Monthly Variations ---
        
        if "DT_NOTIFIC" in df.columns:
            prev_month_start = (current_month_start - pd.DateOffset(months=1))
            prev_month_df = df[(df["DT_NOTIFIC"] >= prev_month_start) & (df["DT_NOTIFIC"] < current_month_start)]
            
            # 5. Notifications increase rate = (current_notifications / prev_notifications) - 1
            prev_notifications = len(prev_month_df)
            if prev_notifications > 0:
                metrics["notifications_increase_rate"] = ((total_notifications / prev_notifications) - 1) * 100
            else:
                metrics["notifications_increase_rate"] = 100.0 if total_notifications > 0 else 0.0
            
            # 6. Confirmed cases increase rate = (current_cases / prev_cases) - 1
            prev_confirmed_cases = 0
            if "CLASSI_FIN" in prev_month_df.columns:
                prev_confirmed_cases = prev_month_df["CLASSI_FIN"].notna().sum()
            
            if prev_confirmed_cases > 0:
                metrics["cases_increase_rate"] = ((confirmed_cases / prev_confirmed_cases) - 1) * 100
            else:
                metrics["cases_increase_rate"] = 100.0 if confirmed_cases > 0 else 0.0

            # 6. Bar Graph History (Last 12 months - total row counts)
            history = []
            for i in range(11, -1, -1):
                m_start = current_month_start - pd.DateOffset(months=i)
                m_end = m_start + pd.DateOffset(months=1)
                month_df = df[(df["DT_NOTIFIC"] >= m_start) & (df["DT_NOTIFIC"] < m_end)]
                val = len(month_df)  # Total rows for month
                    
                history.append({
                    "label": m_start.strftime("%b/%y"),
                    "value": val
                })
            metrics["monthly_history"] = history
            
            # 7. Daily History for Line Graph (Current Month Only)
            daily_history = []
            if len(current_month_df) > 0:
                # Group by day and count (make copy to avoid SettingWithCopyWarning)
                daily_df = current_month_df.copy()
                daily_df["day"] = daily_df["DT_NOTIFIC"].dt.date
                daily_counts = daily_df.groupby("day").size().reset_index(name="count")
                daily_counts = daily_counts.sort_values("day")
                
                for _, row in daily_counts.iterrows():
                    daily_history.append({
                        "label": pd.to_datetime(row["day"]).strftime("%d/%m"),
                        "value": int(row["count"])
                    })
            metrics["daily_history"] = daily_history

        # 8. List of states for filter (from all data)
        if "SG_UF_NOT" in df.columns:
            metrics["states"] = sorted(df["SG_UF_NOT"].dropna().unique().tolist())
        
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
