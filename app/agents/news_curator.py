"""
News Curator Agent Module.

This agent is responsible for:
- Searching for news articles related to a health topic
- Scraping and validating article content
- Using LLM to curate relevant articles and generate summaries
"""
from typing import Dict, Any, List

from app.agents.states import WorkflowState, NewsArticle
from app.config.settings import settings
from app.tools.news_tools import search_news, process_news_article
from app.utils.logging import get_logger

logger = get_logger("agents.news_curator")


def curation_step(state: WorkflowState) -> Dict[str, Any]:
    """
    Orchestrate the news curation process.
    
    Performs: Search -> Scrape -> Curate -> Summarize for each article.
    
    Args:
        state: Current workflow state (may have 'topic' set)
        
    Returns:
        State updates with news_articles list
    """
    logger.info("=== News Curator: Curation Step ===")
    
    topic = state.get("topic") or settings.default_topic
    max_results = settings.max_news_results
    
    logger.info(f"Topic: {topic} | Max Results: {max_results}")
    
    updates: Dict[str, Any] = {
        "topic": topic,
        "news_articles": [],
    }
    
    # 1. Search for news
    logger.info("Searching for news articles...")
    
    try:
        raw_results = search_news.invoke({
            "topic": topic,
            "max_results": max_results
        })
        
        if not raw_results:
            logger.warning("No news results found")
            return updates
            
        logger.info(f"Found {len(raw_results)} raw search results")
        
    except Exception as e:
        logger.error(f"News search failed: {e}")
        updates["errors"] = [f"News search failed: {e}"]
        return updates
    
    # 2. Process each result
    curated_articles: List[NewsArticle] = []
    
    for idx, item in enumerate(raw_results, 1):
        url = item.get("link") or item.get("url")
        
        if not url:
            logger.debug(f"Result {idx}: No URL found, skipping")
            continue
        
        logger.debug(f"Processing result {idx}/{len(raw_results)}: {url}")
        
        try:
            article = process_news_article.invoke({
                "url": url,
                "topic": topic
            })
            
            if article:
                curated_articles.append(article)
                logger.info(f"✓ Added: {article.title[:50]}...")
            else:
                logger.debug(f"✗ Rejected: not relevant")
                
        except Exception as e:
            logger.warning(f"Failed to process {url}: {e}")
            continue
    
    # 3. Update state
    updates["news_articles"] = curated_articles
    
    logger.info(
        f"=== News Curation Complete: {len(curated_articles)}/{len(raw_results)} articles ==="
    )
    
    return updates
