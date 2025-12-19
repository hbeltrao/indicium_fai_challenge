#!/usr/bin/env python3
"""
Test script for news tools.
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.tools.news_tools import search_news, process_news_article
from app.utils.logging import get_logger

logger = get_logger("test.news")


def test_news_tools():
    """Test news search and processing."""
    topic = "Dengue no Brasil 2024"
    
    print(f"Testing search for: {topic}")
    print("-" * 40)
    
    # 1. Test search
    results = search_news.invoke({"topic": topic, "max_results": 3})
    
    print(f"Found {len(results)} results.")
    
    if not results:
        print("✗ Search returned no results.")
        return False
    
    # Print search results
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r.get('title', 'No title')[:60]}...")
    
    # 2. Test article processing
    first_url = results[0].get("link") or results[0].get("url")
    
    if not first_url:
        print("✗ No URL in first result.")
        return False
    
    print(f"\nTesting article processing for: {first_url}")
    print("-" * 40)
    
    article = process_news_article.invoke({"url": first_url, "topic": topic})
    
    if article:
        print("✓ Article processed successfully!")
        print(f"  Title: {article.title}")
        print(f"  Summary: {article.summary[:100]}...")
        print(f"  Date: {article.date}")
        print(f"  Link: {article.original_link}")
        return True
    else:
        print("✗ Article processing returned None")
        print("  (Could be scraping error or irrelevant content)")
        return False


if __name__ == "__main__":
    success = test_news_tools()
    sys.exit(0 if success else 1)
