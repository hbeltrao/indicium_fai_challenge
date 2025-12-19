"""
News Tools Module.

This module provides tools for:
- Searching for news articles via DuckDuckGo
- Scraping and parsing article content
- Curating articles for relevance using LLM

All tools include proper error handling, retry logic, and rate limiting.
"""
import datetime
import time
from typing import Dict, List, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.agents.states import NewsArticle
from app.config.settings import settings
from app.utils.logging import get_logger

logger = get_logger("tools.news")

# Rate limiting state
_last_api_call = 0.0
_min_delay_seconds = 60.0 / settings.api_calls_per_minute


def _rate_limit():
    """Apply rate limiting between API calls."""
    global _last_api_call
    now = time.time()
    elapsed = now - _last_api_call
    if elapsed < _min_delay_seconds:
        sleep_time = _min_delay_seconds - elapsed
        logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
        time.sleep(sleep_time)
    _last_api_call = time.time()


# === News Search Tool ===

@tool
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
)
def search_news(
    topic: str,
    max_results: int = 5,
    region: Optional[str] = None
) -> List[Dict]:
    """
    Search for news articles related to a topic using DuckDuckGo.
    
    Args:
        topic: Search topic/keywords
        max_results: Maximum number of results (default: 5)
        region: Region code for search (default: from settings, e.g., 'br-pt')
        
    Returns:
        List of dictionaries with 'title', 'link', 'date', 'source'
    """
    logger.info(f"Searching news for topic: '{topic}' (max_results={max_results})")
    
    _region = region or settings.news_region
    results = []
    
    try:
        # Use the new ddgs package
        from ddgs import DDGS
        
        with DDGS() as ddgs:
            # Search news specific to region - ddgs v9+ uses 'query' parameter
            news_results = ddgs.news(
                query=topic,
                region=_region,
                safesearch="off",
                max_results=max_results
            )
            
            for item in news_results:
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("url") or item.get("link", ""),
                    "date": item.get("date", ""),
                    "source": item.get("source", ""),
                    "body": item.get("body", ""),
                })
                
    except ImportError:
        # Fallback to old package name if ddgs not available
        logger.warning("ddgs package not found, trying duckduckgo_search")
        try:
            from duckduckgo_search import DDGS as OldDDGS
            
            with OldDDGS() as ddgs:
                news_results = ddgs.news(
                    keywords=topic,
                    region=_region,
                    safesearch="off",
                    max_results=max_results
                )
                for item in news_results:
                    results.append({
                        "title": item.get("title", ""),
                        "link": item.get("url") or item.get("link", ""),
                        "date": item.get("date", ""),
                        "source": item.get("source", ""),
                        "body": item.get("body", ""),
                    })
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            
    except Exception as e:
        logger.error(f"News search failed: {e}")
    
    logger.info(f"Found {len(results)} raw results")
    return results


# === Article Scraping ===

def _scrape_article(url: str) -> dict:
    """
    Scrape article content from a URL.
    
    Args:
        url: Article URL to scrape
        
    Returns:
        Dictionary with 'content', 'title', 'publish_date'
        
    Raises:
        Exception: If scraping fails
    """
    try:
        # Try newspaper4k first
        from newspaper import Article
    except ImportError:
        # Fallback to newspaper3k
        from newspaper import Article
    
    article = Article(url)
    article.download()
    article.parse()
    
    return {
        "content": article.text,
        "title": article.title,
        "publish_date": article.publish_date,
    }


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
)
def _scrape_article_with_retry(url: str) -> dict:
    """Scrape article with retry logic."""
    return _scrape_article(url)


# === Article Processing Tool ===

@tool
def process_news_article(url: str, topic: str) -> Optional[NewsArticle]:
    """
    Scrape, curate, and summarize a news article using LLM.
    
    Downloads the article content, uses an LLM to determine relevance
    to the topic and generate a summary if relevant.
    
    Args:
        url: URL of the article to process
        topic: Topic to check relevance against
        
    Returns:
        NewsArticle if relevant, None if irrelevant or failed
    """
    logger.info(f"Processing article: {url}")
    
    # Apply rate limiting for external calls
    _rate_limit()
    
    # 1. Scrape the article
    try:
        scraped = _scrape_article_with_retry(url)
        content = scraped["content"]
        title = scraped["title"]
        publish_date = scraped["publish_date"]
        
        if not content or len(content) < 100:
            logger.debug(f"Article content too short ({len(content) if content else 0} chars)")
            return None
            
    except Exception as e:
        logger.warning(f"Failed to scrape {url}: {e}")
        return None
    
    # 2. Apply rate limiting before LLM call
    _rate_limit()
    
    # 3. Curate and summarize with LLM
    try:
        from app.models.llms import llm
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a news curator for a health monitoring system. "
             "Analyze articles and determine if they are relevant to the given health topic. "
             "Be strict: only approve articles that are directly about the topic."),
            ("human", """
Topic: {topic}

Article Title: {title}

Article Content (truncated): 
{content}

Tasks:
1. Determine if this article is directly relevant to "{topic}" (must discuss the health topic specifically).
2. If relevant, write a 2-3 sentence summary in Portuguese.
3. Return JSON format:
   - If relevant: {{ "relevant": true, "summary": "...", "title": "..." }}
   - If not relevant: {{ "relevant": false, "reason": "..." }}
""")
        ])
        
        chain = prompt | llm.fast | JsonOutputParser()
        
        result = chain.invoke({
            "topic": topic,
            "title": title,
            "content": content[:5000]  # Limit context
        })
        
        if result.get("relevant"):
            date_str = (
                publish_date.strftime("%Y-%m-%d") 
                if publish_date 
                else datetime.datetime.now().strftime("%Y-%m-%d")
            )
            
            article = NewsArticle(
                title=result.get("title", title),
                summary=result.get("summary", "Resumo indisponível"),
                original_link=url,
                date=date_str
            )
            
            logger.info(f"Article approved: {article.title}")
            return article
        else:
            reason = result.get("reason", "Unknown")
            logger.debug(f"Article rejected: {reason}")
            return None
            
    except Exception as e:
        logger.error(f"LLM curation failed: {e}")
        return None


# === Batch Processing ===

def process_news_batch(
    search_results: List[Dict],
    topic: str,
    max_articles: Optional[int] = None
) -> List[NewsArticle]:
    """
    Process a batch of search results and return curated articles.
    
    Args:
        search_results: List of search result dictionaries
        topic: Topic for relevance checking
        max_articles: Maximum articles to return (None for all relevant)
        
    Returns:
        List of curated NewsArticle objects
    """
    curated = []
    
    for item in search_results:
        if max_articles and len(curated) >= max_articles:
            break
            
        url = item.get("link") or item.get("url")
        if not url:
            continue
        
        try:
            article = process_news_article.invoke({"url": url, "topic": topic})
            if article:
                curated.append(article)
        except Exception as e:
            logger.warning(f"Failed to process {url}: {e}")
            continue
    
    logger.info(f"Curated {len(curated)} articles from {len(search_results)} results")
    return curated
