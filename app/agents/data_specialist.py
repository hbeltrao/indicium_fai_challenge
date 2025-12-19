"""
Data Specialist Agent Module.

This agent is responsible for:
- Downloading SRAG datasets from OpenDataSUS
- Validating dataset columns against the required schema
- Cleaning and filtering the dataset
"""
import os
from typing import Dict, Any

from app.agents.states import WorkflowState
from app.config.settings import settings
from app.tools.data_tools import (
    download_dataset,
    validate_columns,
    clean_dataset,
    find_latest_dataset,
)
from app.utils.logging import get_logger

logger = get_logger("agents.data_specialist")


def download_step(state: WorkflowState) -> Dict[str, Any]:
    """
    Download or locate the dataset.
    
    First checks if a dataset already exists locally. If found, uses it
    to avoid unnecessary downloads. Otherwise, downloads from OpenDataSUS.
    
    Args:
        state: Current workflow state
        
    Returns:
        State updates with raw_dataset_path or errors
    """
    logger.info("=== Data Specialist: Download Step ===")
    
    updates: Dict[str, Any] = {}
    
    # 1. Check for existing dataset
    existing_file = find_latest_dataset()
    
    if existing_file:
        logger.info(f"Using existing dataset: {existing_file}")
        updates["raw_dataset_path"] = existing_file
        return updates
    
    # 2. Download new dataset
    logger.info("No existing dataset found. Downloading...")
    
    try:
        raw_path = download_dataset.invoke({})
        
        if not raw_path:
            error_msg = "Dataset download failed - no path returned"
            logger.error(error_msg)
            updates["errors"] = [error_msg]
        else:
            logger.info(f"Download complete: {raw_path}")
            updates["raw_dataset_path"] = raw_path
            
    except Exception as e:
        error_msg = f"Dataset download exception: {e}"
        logger.error(error_msg)
        updates["errors"] = [error_msg]
    
    return updates


def processing_step(state: WorkflowState) -> Dict[str, Any]:
    """
    Validate and clean the dataset.
    
    Uses LLM to map columns, then filters and cleans the data
    based on the required schema.
    
    Args:
        state: Current workflow state (must have raw_dataset_path)
        
    Returns:
        State updates with refined_dataset_path, metadata_valid, or errors
    """
    logger.info("=== Data Specialist: Processing Step ===")
    
    updates: Dict[str, Any] = {}
    raw_path = state.get("raw_dataset_path")
    
    # 1. Validate input
    if not raw_path:
        error_msg = "No raw dataset path available for processing"
        logger.error(error_msg)
        updates["errors"] = [error_msg]
        return updates
    
    if not os.path.exists(raw_path):
        error_msg = f"Raw dataset file not found: {raw_path}"
        logger.error(error_msg)
        updates["errors"] = [error_msg]
        return updates
    
    # 2. Validate and select columns
    logger.info("Validating columns...")
    
    try:
        selected_columns = validate_columns.invoke({
            "raw_file_path": raw_path,
        })
        
        if not selected_columns:
            error_msg = "Column validation returned empty list"
            logger.warning(error_msg)
            # Continue with all columns as fallback
            selected_columns = []
        else:
            logger.info(f"Selected {len(selected_columns)} columns for refinement")
            
    except Exception as e:
        error_msg = f"Column validation failed: {e}"
        logger.error(error_msg)
        selected_columns = []  # Continue with all columns
    
    # 3. Clean and filter dataset
    logger.info("Cleaning dataset...")
    
    try:
        refined_path = clean_dataset.invoke({
            "raw_file_path": raw_path,
            "selected_columns": selected_columns,
        })
        
        if not refined_path:
            error_msg = "Data cleaning returned no file path"
            logger.error(error_msg)
            updates["errors"] = [error_msg]
            return updates
        
        updates["refined_dataset_path"] = refined_path
        updates["metadata_valid"] = True
        logger.info(f"Data processing complete: {refined_path}")
        
    except Exception as e:
        error_msg = f"Data cleaning failed: {e}"
        logger.error(error_msg)
        updates["errors"] = [error_msg]
    
    return updates
