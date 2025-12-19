#!/usr/bin/env python3
"""
Test script for the complete workflow execution.
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.workflows.main_workflow import run_workflow
from app.utils.logging import get_logger

logger = get_logger("test.workflow")


def test_workflow():
    """Test the complete workflow execution."""
    print("=" * 60)
    print("WORKFLOW EXECUTION TEST")
    print("=" * 60)
    
    # Empty initial state
    initial_state = {"errors": []}
    
    try:
        # Run the graph
        result = run_workflow(initial_state)
        
        print("\n" + "=" * 60)
        print("WORKFLOW FINISHED")
        print("=" * 60)
        
        print("\nFinal State Keys:", list(result.keys()))
        
        # Check outputs
        success = True
        
        if "refined_dataset_path" in result and result["refined_dataset_path"]:
            print(f"✓ Refined Dataset: {result['refined_dataset_path']}")
        else:
            print("✗ Refined Dataset: NOT FOUND")
            success = False
        
        if "news_articles" in result:
            print(f"✓ News Articles: {len(result['news_articles'])} found")
        else:
            print("✗ News Articles: NOT FOUND")
        
        if "final_report_path" in result and result["final_report_path"]:
            print(f"✓ Final Report: {result['final_report_path']}")
        else:
            print("✗ Final Report: NOT GENERATED")
            success = False
        
        if result.get("errors"):
            print(f"\n⚠️  Errors encountered ({len(result['errors'])}):")
            for error in result["errors"]:
                print(f"   - {error}")
        else:
            print("\n✓ No errors encountered")
        
        return success
        
    except Exception as e:
        print(f"\n✗ Workflow Execution Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_workflow()
    sys.exit(0 if success else 1)
