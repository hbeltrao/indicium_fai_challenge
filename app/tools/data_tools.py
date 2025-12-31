"""
Data Tools Module.

This module provides tools for:
- Downloading SRAG datasets from OpenDataSUS
- Validating dataset columns against a schema
- Cleaning and filtering datasets

All tools use proper error handling, logging, and retry logic.
"""
import datetime
import os
import re
from typing import List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pypdf import PdfReader
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from tqdm import tqdm


from app.config.settings import settings
from app.utils.logging import get_logger

logger = get_logger("tools.data")


# === Retry Configuration ===

def _log_retry(retry_state):
    """Log retry attempts."""
    logger.warning(
        f"Retrying {retry_state.fn.__name__} "
        f"(attempt {retry_state.attempt_number}): {retry_state.outcome.exception()}"
    )


# === Download Tool ===

@tool
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((requests.RequestException, ConnectionError)),
    before_sleep=_log_retry,
)
def download_dataset(
    url: str = "https://dadosabertos.saude.gov.br/dataset/srag-2021-a-2024"
) -> str:
    """
    Download the most recent SRAG CSV file from OpenDataSUS.
    
    Accesses the OpenDataSUS page, identifies the most recent CSV resource
    (checking subpages if necessary), and downloads it.
    
    Args:
        url: OpenDataSUS dataset page URL
        
    Returns:
        Absolute path to the downloaded file, or empty string on failure
    """
    logger.info(f"Accessing OpenDataSUS to fetch most recent CSV: {url}")
    
    try:
        session = requests.Session()
        # 1. Get main page content
        response = session.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 2. Start Strategy: Find resource items
        # Structure identified: 
        # a[href*='/resource/'] -> parent div -> parent div -> parent div.card-content
        # Title is in div.text-weight-bold sibling
        resources = []
        seen_links = set()
        
        # Find all distinct resource links
        all_links = soup.find_all('a', href=True)
        resource_candidates = [
            l for l in all_links 
            if '/dataset/srag-2021-a-2024/resource/' in l['href']
        ]
        
        for link in resource_candidates:
            href = link['href']
            if href in seen_links:
                continue
            seen_links.add(href)
            
            # Traverse up to find card-content
            title = "Unknown"
            curr = link
            for _ in range(5): # Max 5 levels up
                if not curr.parent:
                    break
                curr = curr.parent
                if curr.name == 'div' and 'card-content' in curr.get('class', []):
                    # Found card, look for title
                    title_div = curr.find('div', class_='text-weight-bold')
                    if title_div:
                        title = title_div.get_text(strip=True)
                    break
            
            resources.append({
                'title': title,
                'link': href
            })
        
        if not resources:
            logger.error("No resource links found on the main page.")
            return ""

        logger.info(f"Found {len(resources)} resources. Analyzing for most recent year...")

        # 3. Find the most recent resource by year in title
        def score_resource(res: dict) -> Tuple[int, int, int]:
            title = res['title'].upper()
            
            # Penalties/Bonuses
            score = 0
            if 'CSV' in title:
                score += 100
            if 'BANCO' in title or 'DADOS' in title:
                score += 50
            if 'FICHA' in title or 'PDF' in title:
                score -= 100
                
            # Extract Year
            year = 0
            # Try to find year 2020-2030
            years = re.findall(r'202[0-9]', title)
            if years:
                year = int(max(years))
            else:
                # Fallback INFLUDxx
                match = re.search(r'INFLUD(\d{2})', title)
                if match:
                    year = 2000 + int(match.group(1))
            
            # Extract Month/Day if possible (DD/MM/YYYY)
            date_score = 0
            date_match = re.search(r'(\d{2})/(\d{2})/(\d{4})', title)
            if date_match:
                d, m, y = map(int, date_match.groups())
                if y == year:
                    date_score = m * 31 + d
            
            return (year, score, date_score)
        
        # Sort by (Year, Score, Date) descending
        resources.sort(key=score_resource, reverse=True)
        
        best_resource = resources[0]
        # Recalculate year for logging
        resource_year = score_resource(best_resource)[0]
        logger.info(f"Selected resource: '{best_resource['title']}' (Year: {resource_year})")
        
        # Construct full resource URL if relative
        resource_url = best_resource['link']
        if resource_url.startswith('/'):
            # extract base url from input url
            base_match = re.search(r'(https?://[^/]+)', url)
            if base_match:
                resource_url = base_match.group(1) + resource_url

        # 4. Fetch the resource subpage
        logger.info(f"Fetching resource page: {resource_url}")
        res_page = session.get(resource_url, timeout=30)
        res_page.raise_for_status()
        res_soup = BeautifulSoup(res_page.text, 'html.parser')
        
        # 5. Find the actual download link (AWS S3 or .csv)
        # Usually in a 'resource-url-analytics' class or similar, or just text 'Baixar'
        download_link = None
        
        # Strategy A: Look for class 'resource-url-analytics' (common in CKAN)
        dl_tag = res_soup.find('a', class_='resource-url-analytics')
        if dl_tag and dl_tag.get('href'):
            download_link = dl_tag['href']
            
        # Strategy B: Look for any link ending in .csv (case insensitive)
        if not download_link:
            csv_candidates = res_soup.find_all('a', href=True)
            for cand in csv_candidates:
                if cand['href'].lower().endswith('.csv'):
                    download_link = cand['href']
                    break
        
        # Strategy C: Look for link with text "Baixar" or "Download"
        if not download_link:
            for cand in res_soup.find_all('a', href=True):
                text = cand.get_text(strip=True).lower()
                if 'baixar' in text or 'download' in text:
                    # check if it looks like a file url
                    if 'amazonaws' in cand['href'] or '.csv' in cand['href'].lower():
                        download_link = cand['href']
                        break

        if not download_link:
            logger.error("Could not find CSV download link on resource page.")
            return ""

        # Extract filename from URL
        filename = download_link.split('/')[-1]
        # Safety: ensure it ends with csv
        if not filename.lower().endswith('.csv'):
            filename += ".csv"
            
        logger.info(f"Found download link: {download_link}")
        logger.info(f"Target filename: {filename}")
        
        # 6. Setup output directory
        output_dir = settings.data_path
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)
        
        # Check if file already exists
        if os.path.exists(save_path):
            logger.info(f"File already exists: {save_path}")
            return save_path
        
        # 7. Download with progress bar
        logger.info(f"Downloading {filename}...")
        csv_response = session.get(download_link, stream=True, timeout=300)
        csv_response.raise_for_status()
        
        total_size = int(csv_response.headers.get('content-length', 0))
        block_size = 8192  # 8 KB
        
        with open(save_path, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in csv_response.iter_content(block_size):
                size = f.write(chunk)
                progress_bar.update(size)
        
        logger.info(f"Successfully saved to: {save_path}")
        return save_path
        
    except requests.RequestException as e:
        logger.error(f"Network error downloading CSV: {e}")
        raise  # Will trigger retry
    except Exception as e:
        logger.error(f"Unexpected error downloading CSV: {e}")
        return ""


# === Column Validation Tool ===

@tool
def validate_columns(
    raw_file_path: str,
    dictionary_pdf_path: str = "data/dicionario-de-dados-2019-a-2025.pdf"
) -> List[str]:
    """
    Validate and map raw dataset columns to the required schema using an LLM.
    
    Uses an LLM agent to intelligently map the raw CSV column names to the 
    target schema fields based on the data dictionary.
    
    Args:
        raw_file_path: Path to the raw CSV file
        dictionary_pdf_path: Path to the data dictionary PDF
        
    Returns:
        List of raw column names that should be selected
    """
    logger.info(f"Validating columns for: {raw_file_path}")
    
    # 1. Validate input file exists
    if not os.path.exists(raw_file_path):
        logger.error(f"Raw file not found: {raw_file_path}")
        return []
    
    # 2. Read raw columns (header only)
    try:
        df_head = pd.read_csv(raw_file_path, sep=';', encoding='latin1', nrows=0)
        raw_columns = list(df_head.columns)
        logger.debug(f"Found {len(raw_columns)} columns in raw file")
    except Exception as e:
        logger.error(f"Could not read raw file header: {e}")
        return []
    
    # 3. Read PDF dictionary context
    pdf_text = ""
    if os.path.exists(dictionary_pdf_path):
        try:
            reader = PdfReader(dictionary_pdf_path)
            # Limit to first 15 pages (usually contains variable list)
            for i, page in enumerate(reader.pages[:15]):
                pdf_text += page.extract_text() or ""
            logger.debug(f"Read {len(pdf_text)} chars from PDF dictionary")
        except Exception as e:
            logger.warning(f"Could not read PDF dictionary: {e}")
    else:
        logger.warning(f"PDF dictionary not found: {dictionary_pdf_path}")
    
    # 4. Get target schema
    from app.agents.states import RefinedSragCasesDataset
    target_properties = RefinedSragCasesDataset.model_json_schema()['properties']
    target_descriptions = {k: v.get('description', k) for k, v in target_properties.items()}
    
    # 5. Use LLM to map columns
    try:
        from app.models.llms import llm
        from app.utils.prompts import get_chat_prompt_content
        
        system_prompt, human_prompt = get_chat_prompt_content("column_mapping")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
        
        chain = prompt | llm.fast | JsonOutputParser()
        
        mapping = chain.invoke({
            "target_descriptors": str(target_descriptions),
            "raw_columns": str(raw_columns),
            "pdf_text": pdf_text[:8000]  # Context limit
        })
        
        logger.info(f"Column mapping: {mapping}")
        
        # Extract valid column names
        selected_raw_columns = [v for v in mapping.values() if v in raw_columns]
        selected_raw_columns = list(set(selected_raw_columns))  # Remove duplicates
        
        logger.info(f"Selected {len(selected_raw_columns)} columns for refinement")
        return selected_raw_columns
        
    except Exception as e:
        logger.error(f"LLM mapping failed: {e}. Using fallback exact match.")
        # Fallback: simple case-insensitive match
        fallback = [c for c in raw_columns if c.lower() in target_properties.keys()]
        logger.info(f"Fallback selected {len(fallback)} columns")
        return fallback


# === Data Cleaning Tool ===

@tool
def clean_dataset(
    raw_file_path: str,
    selected_columns: List[str],
    date_range: Optional[Tuple[str, str]] = None
) -> str:
    """
    Clean and filter the raw dataset.
    
    Reads the raw dataset, selects specified columns, applies date filtering,
    and saves the refined data to a new CSV file.
    
    Args:
        raw_file_path: Path to the raw CSV file
        selected_columns: List of column names to keep
        date_range: Optional tuple of (start_date, end_date) in 'YYYY-MM-DD' format.
                   If None, defaults to the previous 11 full months plus current month.
                   
    Returns:
        Path to the refined dataset file, or empty string on failure
    """
    logger.info(f"Cleaning dataset: {raw_file_path}")
    
    # 1. Validate inputs
    if not os.path.exists(raw_file_path):
        logger.error(f"Raw file not found: {raw_file_path}")
        return ""
    
    if not selected_columns:
        logger.warning("No columns specified. Reading all columns.")
    
    try:
        # 2. Read data with selected columns
        if selected_columns:
            df = pd.read_csv(
                raw_file_path,
                sep=';',
                encoding='latin1',
                usecols=lambda c: c in selected_columns,
                low_memory=False
            )
            logger.debug(f"Read {len(df)} rows with {len(df.columns)} columns")
        else:
            df = pd.read_csv(
                raw_file_path,
                sep=';',
                encoding='latin1',
                low_memory=False
            )
        
        # 3. Find and process date column
        date_candidates = [c for c in df.columns if 'DT_NOTI' in c.upper()]
        date_col = date_candidates[0] if date_candidates else None
        
        if date_col:
            logger.info(f"Using '{date_col}' for date filtering")
            
            # Keep original for reparse attempts
            original_dates = df[date_col].copy()
            
            # Parse dates - try multiple formats
            # DATASUS uses different formats: yyyy-mm-dd (most common), dd/mm/yyyy
            for fmt in ['%Y-%m-%d', '%d/%m/%Y', None]:
                try:
                    if fmt:
                        parsed = pd.to_datetime(original_dates, format=fmt, errors='coerce')
                    else:
                        # Fallback: let pandas infer with dayfirst=True for Brazilian dates
                        parsed = pd.to_datetime(original_dates, dayfirst=True, errors='coerce')
                    
                    # Check if parsing worked (more than 50% valid)
                    valid_count = parsed.notna().sum()
                    if valid_count > len(df) * 0.5:
                        df[date_col] = parsed
                        logger.debug(f"Date format detected: {fmt or 'auto-inferred'} ({valid_count}/{len(df)} valid)")
                        break
                except Exception as e:
                    logger.debug(f"Date format {fmt} failed: {e}")
                    continue
            
            # Determine date range
            if date_range:
                start_date = pd.to_datetime(date_range[0])
                end_date = pd.to_datetime(date_range[1])
            else:
                # Default: previous 11 months full + current month
                now = datetime.datetime.now()
                
                # Calculate the 1st day of the month 11 months ago
                year = now.year
                month = now.month - 11
                while month <= 0:
                    month += 12
                    year -= 1
                
                start_date = datetime.datetime(year, month, 1)
                # Ensure end_date covers all entries in the current month (and a bit of buffer)
                end_date = now + datetime.timedelta(days=7)
            
            # Apply filter
            mask = (df[date_col] >= start_date) & (df[date_col] <= end_date)
            original_count = len(df)
            df = df[mask]
            
            logger.info(
                f"Filtered records: {start_date.strftime('%Y-%m-%d')} to "
                f"{end_date.strftime('%Y-%m-%d')} ({len(df)}/{original_count} rows)"
            )
        else:
            logger.warning("No date column found. Skipping date filter.")
        
        # 4. Save refined dataset
        output_dir = os.path.dirname(raw_file_path)
        output_path = os.path.join(output_dir, "refined_dataset.csv")
        
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Refined dataset saved: {output_path} ({len(df)} rows)")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to clean dataset: {e}")
        return ""


# === Utility Functions ===

def find_latest_dataset(data_dir: Optional[str] = None) -> Optional[str]:
    """
    Find the most recent INFLUD CSV file in the data directory.
    
    Args:
        data_dir: Directory to search (defaults to settings.data_dir)
        
    Returns:
        Path to the most recent file, or None if not found
    """
    search_dir = data_dir or settings.data_path
    
    if not os.path.exists(search_dir):
        logger.warning(f"Data directory not found: {search_dir}")
        return None
    
    csv_files = [
        f for f in os.listdir(search_dir)
        if f.startswith('INFLUD') and f.endswith('.csv')
    ]
    
    if not csv_files:
        logger.debug(f"No INFLUD CSV files found in {search_dir}")
        return None
    
    # Sort by date in filename (INFLUD25-DD-MM-YYYY.csv)
    def extract_date(filename: str) -> datetime.datetime:
        try:
            # Extract date part: INFLUD25-DD-MM-YYYY.csv
            match = re.search(r'INFLUD\d{2}-(\d{2})-(\d{2})-(\d{4})', filename)
            if match:
                day, month, year = match.groups()
                return datetime.datetime(int(year), int(month), int(day))
        except Exception:
            pass
        return datetime.datetime.min
    
    sorted_files = sorted(csv_files, key=extract_date, reverse=True)
    latest = os.path.join(search_dir, sorted_files[0])
    
    logger.debug(f"Found latest dataset: {latest}")
    return latest
