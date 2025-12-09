import os
import argparse
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load env before imports
load_dotenv()

from app.agents.graph import build_graph

def main():
    parser = argparse.ArgumentParser(
        description="""
Indicium HealthCare Report Generator
====================================
Generates a one-page PDF public health report using AI to gather data and curate news.

Features:
- Fetches epidemiological data (Simulated DATASUS).
- Scrapes and summarizes health news (AI-powered).
- Generates trends/graphics (Matplotlib).
- Outputs a professional PDF report.

Usage:
  python main.py --days 30
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--days", type=int, default=30, help="Number of past days to analyze (default: 30)")
    args = parser.parse_args()

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        
        # Curated List of Sources
        urls = [
            "https://www.gov.br/saude/pt-br",
            "https://www.cnnbrasil.com.br/saude/", 
            "https://g1.globo.com/saude/",
            "https://agenciabrasil.ebc.com.br/saude",
            "https://www.bbc.com/portuguese/topics/c4794229c22t"
        ]
        
        print(f"Starting Report Generation for period: {start_date.date()} to {end_date.date()}")
        print("-" * 50)
        
        app = build_graph()
        
        input_state = {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "news_urls": urls,
            "raw_srag_data": [],
            "curated_news": []
        }
        
        result = app.invoke(input_state)
        
        if result.get("validation_status") == "passed":
            print("-" * 50)
            print(f"SUCCESS! Report generated at: {result.get('report_rel_path')}")
            print("-" * 50)
        else:
            print("-" * 50)
            print(f"FAILED. Validation errors: {result.get('errors')}")
            print("-" * 50)

    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
    except Exception as e:
        print(f"\n\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
