"""
Workflow State Definitions.

This module defines:
- Data models for structured data (NewsArticle, RefinedSragCasesDataset)
- Workflow state type definition for LangGraph
"""
import operator
from typing import List, Dict, Optional, Any, Annotated

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class NewsArticle(BaseModel):
    """
    Curated news article model.
    
    Represents a news article that has been scraped, validated for relevance,
    and summarized by the News Curator agent.
    
    Attributes:
        title: Article headline
        summary: LLM-generated summary in Portuguese (2-3 sentences)
        original_link: URL to the original article
        date: Publication date in YYYY-MM-DD format
    """
    title: str = Field(..., description="Article headline")
    summary: str = Field(..., description="LLM-generated summary in Portuguese")
    original_link: str = Field(..., description="URL to the original article")
    date: str = Field(..., description="Publication date (YYYY-MM-DD)")


class RefinedSragCasesDataset(BaseModel):
    """
    Schema definition for refined SRAG dataset.
    
    Defines the target schema that raw SRAG data should be mapped to.
    Field names are lowercase and match the target column names after refinement.
    
    This model is used by the Data Specialist agent to:
    1. Identify which columns to select from the raw data
    2. Validate that required columns are present
    3. Provide context to the LLM for column mapping
    
    Reference: DATASUS SRAG Data Dictionary (dicionario-de-dados-2019-a-2025.pdf)
    """
    nu_notific: str = Field(
        ..., 
        title="Notification ID",
        description="ID do registro da notificação (Notification record ID)"
    )
    dt_notific: str = Field(
        ..., 
        title="Notification Date",
        description="Data do preenchimento da ficha de notificação (Notification form fill date)"
    )
    sg_uf_not: str = Field(
        ..., 
        title="State Code",
        description="Unidade Federativa onde está localizada a Unidade que realizou a notificação (Federal Unit of notification)"
    )
    id_municip: str = Field(
        ..., 
        title="Municipality IBGE Code",
        description="Município onde está localizada a Unidade que realizou a notificação (IBGE municipality code)"
    )
    vacina_cov: str = Field(
        ..., 
        title="COVID-19 Vaccination Status",
        description="Informar se o paciente recebeu vacina COVID-19 (COVID-19 vaccination status)"
    )
    vacina: str = Field(
        ..., 
        title="Influenza Vaccination Status",
        description="Informar se o paciente foi vacinado contra gripe na última campanha (Influenza vaccination status)"
    )
    hospital: str = Field(
        ..., 
        title="Hospitalization Status",
        description="O paciente foi internado? (Was the patient hospitalized?)"
    )
    dt_interna: str = Field(
        ..., 
        title="Hospitalization Date",
        description="Data em que o paciente foi hospitalizado (Date of hospitalization)"
    )
    uti: str = Field(
        ..., 
        title="ICU Admission",
        description="O paciente foi internado em UTI? (Was the patient admitted to ICU?)"
    )
    surto_sg: str = Field(
        ..., 
        title="Outbreak Status",
        description="O caso faz parte de uma cadeia de surto de SRAG? (Is case part of an SRAG outbreak?)"
    )
    classi_fin: str = Field(
        ..., 
        title="Final Classification",
        description="Diagnóstico final do caso (Final case diagnosis/classification)"
    )
    evolucao: str = Field(
        ..., 
        title="Case Outcome",
        description="Evolução do caso: 1=Cura, 2=Óbito, 3=Óbito outras causas, 9=Ignorado"
    )
    dt_encerra: str = Field(
        ..., 
        title="Closure Date",
        description="Data do encerramento do caso (Case closure date)"
    )


class WorkflowState(TypedDict, total=False):
    """
    LangGraph Workflow State Definition.
    
    Defines all state variables that flow through the workflow graph.
    Uses TypedDict for compatibility with LangGraph's state management.
    
    The workflow has three main pipelines:
    1. Data Pipeline: Downloads, validates, and cleans SRAG data
    2. News Pipeline: Searches, scrapes, and curates news articles
    3. Report Pipeline: Combines data and generates HTML report
    
    Attributes:
        raw_dataset_path: Path to downloaded raw SRAG CSV file
        refined_dataset_path: Path to cleaned/filtered dataset
        refined_dataset: Structured dataset (currently unused)
        metadata_valid: Flag indicating successful data validation
        
        metrics: Dictionary of calculated metrics (total_cases, etc.)
        
        news_articles: List of curated NewsArticle objects
        topic: Topic for news search (default: "SRAG")
        
        final_report_path: Path to generated HTML report
        errors: Accumulated error messages (uses reducer for append)
    """
    # === Data Pipeline State ===
    raw_dataset_path: Optional[str]
    refined_dataset_path: Optional[str]
    refined_dataset: Optional[RefinedSragCasesDataset]  # For future type validation
    metadata_valid: bool
    
    # === Metrics State ===
    metrics: Dict[str, Any]
    
    # === News Pipeline State ===
    news_articles: List[NewsArticle]
    topic: str
    
    # === Report State ===
    final_report_path: Optional[str]
    
    # === Error Accumulation ===
    # Uses operator.add reducer to accumulate errors from all nodes
    errors: Annotated[List[str], operator.add]
