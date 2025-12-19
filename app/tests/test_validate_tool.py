#!/usr/bin/env python3
"""
Test script for the validate_columns tool.
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.tools.data_tools import validate_columns, find_latest_dataset
from app.utils.logging import get_logger

logger = get_logger("test.validate")


def test_validate_columns():
    """Test the validate_columns tool."""
    # Find files
    csv_file = find_latest_dataset()
    pdf_file = "data/dicionario-de-dados-2019-a-2025.pdf"
    
    if not csv_file:
        print("Error: No INFLUD CSV file found in data/")
        print("Run the workflow first to download the dataset.")
        return False
    
    print(f"CSV file: {csv_file}")
    print(f"PDF file: {pdf_file}")
    
    if not os.path.exists(pdf_file):
        print(f"Warning: PDF dictionary not found at {pdf_file}")
        print("Column mapping will use fallback logic.")
    
    print("\nRunning validate_columns tool...")
    print("-" * 40)
    
    selected_columns = validate_columns.invoke({
        "raw_file_path": csv_file,
        "dictionary_pdf_path": pdf_file
    })
    
    print("\n--- Test Results ---")
    
    if selected_columns:
        print(f"✓ Successfully selected {len(selected_columns)} columns:")
        for col in selected_columns:
            print(f"  - {col}")
        return True
    else:
        print("✗ Column validation returned empty list.")
        return False


if __name__ == "__main__":
    success = test_validate_columns()
    sys.exit(0 if success else 1)
