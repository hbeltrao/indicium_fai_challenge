# Start of Imports
from datetime import datetime
from langchain_openai import ChatOpenAI
try:
    from langchain_google_vertexai import ChatVertexAI
except ImportError:
    ChatVertexAI = None # prevent crash if dependency missing in dev environment

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.agents.state import AgentState
from app.tools.datasus import DatasusTool
from app.tools.metrics import MetricsTool
from app.tools.scraper import NewsScraperTool
from app.tools.report import ReportGeneratorTool
from app.core.config import settings

# Initialize Tools
datasus_tool = DatasusTool()
metrics_tool = MetricsTool()
scraper_tool = NewsScraperTool()
report_tool = ReportGeneratorTool()

# Initialize LLM Factory
def get_llm():
    provider = settings.LLM_PROVIDER.lower()
    print(f"--- [INIT] Initializing LLM Provider: {provider} ---")
    
    if provider == "vertex":
        if not ChatVertexAI:
            raise ImportError("langchain-google-vertexai not installed.")
        
        # Vertex Config
        return ChatVertexAI(
            model_name="gemini-2.5-flash", 
            project=settings.GOOGLE_PROJECT_ID,
            location=settings.GOOGLE_LOCATION,
            temperature=0,
            convert_system_message_to_human=True # often needed for Vertex
        )
    else:
        # Default to OpenAI
        if not settings.OPENAI_API_KEY:
            print("  [WARN] OPENAI_API_KEY not set. Expect failure if using OpenAI.")
        return ChatOpenAI(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY, temperature=0)

llm_curator = get_llm()

def data_specialist_node(state: AgentState):
    """
    Fetches raw data and calculates metrics.
    """
    print("\n--- [NODE] Data Specialist Working ---")
    start = state["start_date"]
    end = state["end_date"]
    
    try:
        # 1. Fetch
        raw_data = datasus_tool.run({"start_date": start, "end_date": end, "region": "BR"})
        
        # 2. Calculate
        metrics = metrics_tool.run({"raw_data": raw_data})
        
        return {
            "raw_srag_data": raw_data.get("srag_cases_daily", []),
            "raw_metrics_data": raw_data,
            "calculated_metrics": metrics
        }
    except Exception as e:
        print(f"!!! [DATA SPECIALIST ERROR] {e}")
        return {
            "errors": [f"Data Specialist Failed: {str(e)}"]
        }

def news_curator_node(state: AgentState):
    """
    Scrapes, filters, and summarizes news.
    """
    print("\n--- [NODE] News Curator Working ---")
    urls = state["news_urls"]
    curated = []
    
    try:
        # 1. Scrape
        raw_articles = scraper_tool.run({"urls": urls})
        
        # 2. Filter & Summarize
        print(f"  [LLM] Filtering and Summarizing {len(raw_articles)} articles...")
        
        for article in raw_articles:
            if article["status"] != "success":
                continue
                
            content = article["content"]
            
            prompt = PromptTemplate.from_template(
                """
                Analyze the following health news article text.
                
                1. Safety Check: If the content is about 'celebrity gossip', 'sports matches' (not injury stats), 'politics' (purely electioneering), 'adult content', or 'crime', output "SKIP".
                2. Relevance: It MUST be about public health, vaccines, diseases, or medical breakthroughs.
                3. Summary: If safe and relevant, provide a 2-sentence summary in Portuguese.
                
                Article Title: {title}
                Article Content: {content}
                
                Output format: "SUMMARY: <summary>" or "SKIP"
                """
            )
            chain = prompt | llm_curator | StrOutputParser()
            
            try:
                # truncate content to avoid token limits
                result = chain.invoke({"title": article["title"], "content": content[:2000]})
                
                if "SKIP" in result:
                    print(f"    [SKIP] Irrelevant/Unsafe: {article['title'][:30]}...")
                    continue
                
                summary = result.replace("SUMMARY:", "").strip()
                
                print(f"    [ADD] {article['title'][:30]}...")
                curated.append({
                    "title": article["title"],
                    "url": article["url"],
                    "content": summary
                })
                
                if len(curated) >= 5:
                    break
                    
            except Exception as inner_e:
                print(f"    [LLM FAIL] Error on article {article['title']}: {inner_e}")
                continue
    
    except Exception as e:
        print(f"!!! [NEWS CURATOR ERROR] {e}")
        # Return what we have
    
    return {
        "raw_news_items": raw_articles if 'raw_articles' in locals() else [],
        "curated_news": curated
    }

def validation_node(state: AgentState):
    """
    Checks if we have enough data to generate the report.
    """
    print("\n--- [NODE] Validating State ---")
    errors = state.get("errors", [])
    
    # Check Metrics
    if not state.get("calculated_metrics"):
        errors.append("Missing metrics")
    
    # Check News
    news = state.get("curated_news", [])
    if len(news) < 3: # Soft requirement
        msg = f"Insufficient news items: found {len(news)}, need at least 3"
        print(f"  [WARN] {msg}")
        errors.append(msg)
    else:
        print(f"  [CHECK] News count passes ({len(news)})")
        
    status = "failed" if errors else "passed"
    print(f"  [RESULT] Validation Status: {status}")
    if errors:
        print(f"  [ERRORS] {errors}")
    
    return {
        "errors": errors,
        "validation_status": status
    }

def layout_designer_node(state: AgentState):
    """
    Generates the PDF.
    """
    print("\n--- [NODE] Generating Output ---")
    
    output_path = f"output/Relatorio_Saude_{datetime.now().strftime('%Y%m%d')}.pdf"
    
    try:
        report_tool.run({
            "metrics": state["calculated_metrics"],
            "srag_data": state["raw_srag_data"],
            "news_items": state["curated_news"],
            "output_path": output_path
        })
        return {"report_rel_path": output_path}
    except Exception as e:
        print(f"!!! [LAYOUT ERROR] {e}")
        return {"errors": [f"Layout Generation Failed: {str(e)}"]}
