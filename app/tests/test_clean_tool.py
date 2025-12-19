#!/usr/bin/env python3
"""
Test script for the clean_dataset tool.
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from app.tools.data_tools import clean_dataset, find_latest_dataset
from app.utils.logging import get_logger

logger = get_logger("test.clean")


def test_clean_dataset():
    """Test the clean_dataset tool."""
    # Find a CSV file
    csv_file = find_latest_dataset()
    
    if not csv_file:
        print("Error: No INFLUD CSV file found in data/")
        print("Run the workflow first to download the dataset.")
        return False
    
    print(f"Using file: {csv_file}")
    
    # Define columns to select (based on schema)
    selected_columns = [
        'NU_NOTIFIC', 'DT_NOTIFIC', 'SG_UF_NOT', 'ID_MUNICIP',
        'VACINA', 'HOSPITAL', 'DT_INTERNA', 'UTI',
        'CLASSI_FIN', 'EVOLUCAO', 'DT_ENCERRA', 'VACINA_COV', 'SURTO_SG'
    ]
    
    # Define a wide date range to ensure data exists
    date_range = ('2024-01-01', '2025-12-31')
    
    print("Running clean_dataset tool...")
    
    output_path = clean_dataset.invoke({
        "raw_file_path": csv_file,
        "selected_columns": selected_columns,
        "date_range": date_range
    })
    
    print("\n--- Test Results ---")
    
    if output_path and os.path.exists(output_path):
        print(f"✓ Success! File created at: {output_path}")
        
        df_out = pd.read_csv(output_path)
        print(f"  Output rows: {len(df_out)}")
        print(f"  Output columns: {list(df_out.columns)}")
        
        if 'DT_NOTIFIC' in df_out.columns:
            dates = pd.to_datetime(df_out['DT_NOTIFIC'], errors='coerce')
            print(f"  Min Date: {dates.min()}")
            print(f"  Max Date: {dates.max()}")
        
        return True
    else:
        print("✗ Failure: Output file not created.")
        return False


if __name__ == "__main__":
    success = test_clean_dataset()
    sys.exit(0 if success else 1)
