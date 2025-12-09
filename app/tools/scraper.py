from typing import List, Dict, Any
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_community.document_loaders import WebBaseLoader
import requests
from bs4 import BeautifulSoup

class NewsScraperInput(BaseModel):
    urls: List[str] = Field(description="List of URLs to scrape health news from")

class NewsScraperTool(BaseTool):
    name: str = "scrape_health_news"
    description: str = "Scrapes content from provided health news URLs."
    args_schema: type[BaseModel] = NewsScraperInput

    def _run(self, urls: List[str]) -> List[Dict[str, str]]:
        print(f"  [SCRAPER] Starting extraction for {len(urls)} URLs...")
        results = []
        # Fallback headers to look like a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        for url in urls:
            print(f"    -> Scraping: {url}")
            try:
                # Basic mock check for implementation plan phase if internet not available
                # or strictly enforce scraping
                
                # Using simple requests + bs4 for finer control than WebBaseLoader sometimes
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract Title
                    title = soup.title.string if soup.title else "No Title"
                    
                    # Extract Main Content (heuristic search for <article> or <main> or common classes)
                    # This is a general scraper. Specific sites might need custom parsing.
                    article_body = soup.find('article') or soup.find('main') or soup.body
                    text_content = article_body.get_text(separator=' ', strip=True) if article_body else ""
                    
                    # Trim to reasonable length to avoid context overflow
                    text_content = text_content[:5000]

                    print(f"    [OK] Captured '{title[:30]}...' ({len(text_content)} chars)")
                    results.append({
                        "url": url,
                        "title": title,
                        "content": text_content,
                        "status": "success"
                    })
                else:
                    print(f"    [FAIL] HTTP {response.status_code}")
                    results.append({
                        "url": url,
                        "title": "Error",
                        "content": "",
                        "status": f"http_error_{response.status_code}"
                    })
            except Exception as e:
                print(f"    [ERROR] {e}")
                results.append({
                    "url": url,
                    "title": "Error",
                    "content": str(e),
                    "status": "exception"
                })
        
        print(f"  [SCRAPER] Finished. Success: {len([r for r in results if r['status'] == 'success'])}")
        return results

    def _arun(self, urls: List[str]):
        raise NotImplementedError("Async not implemented")
