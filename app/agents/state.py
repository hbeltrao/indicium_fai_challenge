from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    """
    Represents the state of the public health report generation pipeline.
    """
    # Inputs
    start_date: str
    end_date: str
    news_urls: List[str]

    # Data Branch State
    raw_srag_data: Optional[List[Dict[str, Any]]]
    raw_metrics_data: Optional[Dict[str, Any]] # Consolidated data (mortality, icu)
    calculated_metrics: Optional[Dict[str, float]]

    # News Branch State
    raw_news_items: List[Dict[str, str]]
    curated_news: List[Dict[str, str]]

    # Output
    report_rel_path: Optional[str]
    
    # Validation/Errors
    errors: List[str]
    validation_status: str # 'pending', 'passed', 'failed'
