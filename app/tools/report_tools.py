"""
Report Tools Module.

This module provides tools for:
- Rendering HTML reports using Jinja2 templates
- Managing report output files

All tools include proper error handling and logging.
"""
import datetime
import os
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from langchain_core.tools import tool

from app.config.settings import settings
from app.utils.logging import get_logger

logger = get_logger("tools.report")


def _get_template_env() -> Environment:
    """
    Get configured Jinja2 environment.
    
    Returns:
        Jinja2 Environment with template directory configured
    """
    template_dir = os.path.join(os.getcwd(), 'app', 'templates')
    
    if not os.path.exists(template_dir):
        logger.warning(f"Template directory not found: {template_dir}")
        os.makedirs(template_dir, exist_ok=True)
    
    return Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=True,  # Security: auto-escape HTML
    )


@tool
def render_report(
    data: Dict[str, Any],
    template_name: str = "report_template.html",
    output_filename: Optional[str] = None
) -> str:
    """
    Render an HTML report using a Jinja2 template.
    
    Takes the provided data dictionary and renders it into an HTML report
    using the specified template. The report is saved to the output directory.
    
    Args:
        data: Dictionary containing report data (metrics, news_articles, etc.)
        template_name: Name of the Jinja2 template file
        output_filename: Optional custom output filename (auto-generated if None)
        
    Returns:
        Absolute path to the generated HTML file, or empty string on failure
        
    Expected data structure:
        {
            "refined_dataset_path": str,
            "metrics": {"total_cases": int, ...},
            "news_articles": [NewsArticle, ...],
            "topic": str,
            "errors": [str, ...]
        }
    """
    logger.info(f"Generating report using template: {template_name}")
    
    # 1. Validate data
    if not isinstance(data, dict):
        logger.error(f"Invalid data type: expected dict, got {type(data)}")
        return ""
    
    # 2. Setup Jinja2 environment
    env = _get_template_env()
    
    try:
        template = env.get_template(template_name)
    except TemplateNotFound:
        logger.error(f"Template not found: {template_name}")
        return ""
    except Exception as e:
        logger.error(f"Error loading template: {e}")
        return ""
    
    # 3. Prepare data
    report_data = dict(data)  # Copy to avoid mutation
    report_data['generation_date'] = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    
    # Ensure required fields have defaults
    report_data.setdefault('refined_dataset_path', 'N/A')
    report_data.setdefault('metrics', {'total_cases': 0})
    report_data.setdefault('news_articles', [])
    report_data.setdefault('topic', settings.default_topic)
    report_data.setdefault('errors', [])
    
    # Convert NewsArticle objects to dicts for template compatibility
    if report_data['news_articles']:
        articles = []
        for article in report_data['news_articles']:
            if hasattr(article, 'model_dump'):
                articles.append(article.model_dump())
            elif hasattr(article, 'dict'):
                articles.append(article.dict())
            elif isinstance(article, dict):
                articles.append(article)
            else:
                logger.warning(f"Unknown article type: {type(article)}")
        report_data['news_articles'] = articles
    
    # 4. Render template
    try:
        html_content = template.render(**report_data)
    except Exception as e:
        logger.error(f"Failed to render template: {e}")
        return ""
    
    # 5. Save output
    output_dir = settings.output_path
    os.makedirs(output_dir, exist_ok=True)
    
    if output_filename:
        filename = output_filename
    else:
        filename = f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    
    output_path = os.path.join(output_dir, filename)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"Report saved: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to save report: {e}")
        return ""


def cleanup_old_reports(max_reports: int = 3) -> int:
    """
    Clean up old report files, keeping only the most recent ones.
    
    Args:
        max_reports: Maximum number of reports to keep
        
    Returns:
        Number of files deleted
    """
    output_dir = settings.output_path
    
    if not os.path.exists(output_dir):
        return 0
    
    # Find all report HTML files
    report_files = [
        f for f in os.listdir(output_dir)
        if f.startswith('report_') and f.endswith('.html')
    ]
    
    if len(report_files) <= max_reports:
        return 0
    
    # Sort by modification time (oldest first)
    report_files_with_time = [
        (f, os.path.getmtime(os.path.join(output_dir, f)))
        for f in report_files
    ]
    report_files_with_time.sort(key=lambda x: x[1])
    
    # Delete oldest files
    files_to_delete = report_files_with_time[:-max_reports]
    deleted = 0
    
    for filename, _ in files_to_delete:
        try:
            os.remove(os.path.join(output_dir, filename))
            deleted += 1
            logger.debug(f"Deleted old report: {filename}")
        except Exception as e:
            logger.warning(f"Failed to delete {filename}: {e}")
    
    if deleted:
        logger.info(f"Cleaned up {deleted} old report files")
    
    return deleted
