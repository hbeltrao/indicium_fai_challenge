#!/usr/bin/env python3
"""
Indicium FAI Challenge - Main Entry Point.

This script runs the AI-powered health data analysis workflow:
1. Downloads SRAG data from DATASUS OpenDataSUS
2. Validates and cleans the dataset
3. Searches and curates relevant news articles
4. Generates an HTML report with metrics and news
"""
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config.settings import settings, LLMProvider
from app.utils.logging import get_logger, setup_logging
from app.workflows.main_workflow import run_workflow

# Initialize logging first
logger = get_logger("main")


def print_banner():
    """Print startup banner with configuration info."""
    print("\n" + "=" * 60)
    print("   INDICIUM FAI CHALLENGE - Health Data Agent")
    print("=" * 60)


def print_config():
    """Print current configuration."""
    logger.info("Configuration:")
    logger.info(f"  LLM Provider: {settings.llm_provider.value}")
    logger.info(f"  LLM Model: {settings.llm_model_name}")
    logger.info(f"  GCP Project: {settings.google_cloud_project or 'N/A'}")
    logger.info(f"  LangSmith: {'Enabled' if settings.langchain_tracing_v2 else 'Disabled'}")
    logger.info(f"  Data Dir: {settings.data_path}")
    logger.info(f"  Output Dir: {settings.output_path}")
    logger.info(f"  Default Topic: {settings.default_topic}")


def main():
    """Main entry point."""
    print_banner()
    print_config()
    
    print("\n" + "-" * 60)
    logger.info("Starting workflow execution...")
    print("-" * 60 + "\n")
    
    try:
        # Run the workflow
        final_state = run_workflow({"errors": []})
        
        # Print results
        print("\n" + "=" * 60)
        print("   EXECUTION COMPLETE")
        print("=" * 60)
        
        if final_state.get("final_report_path"):
            logger.info(f"Report generated: {final_state['final_report_path']}")
            print(f"\n📄 Final Report: {final_state['final_report_path']}")
        else:
            logger.warning("No report was generated")
            print("\n⚠️  No report was generated")
        
        if final_state.get("errors"):
            logger.warning(f"Errors: {final_state['errors']}")
            print(f"\n⚠️  Errors encountered: {len(final_state['errors'])}")
            for error in final_state['errors']:
                print(f"   - {error}")
        
        # Return success/failure
        return 0 if final_state.get("final_report_path") else 1
        
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        print("\n\n⚠️  Interrupted by user")
        return 130
        
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        print(f"\n❌ Fatal Error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
