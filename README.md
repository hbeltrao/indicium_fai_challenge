# Indicium HealthCare Report Generator

An AI-powered system that generates comprehensive public health reports by combining statistical analysis (DATASUS simulation) with AI-curated news summaries.

## Features

- **Automated Data Retrieval**: Fetches and aggregates epidemiological data (SRAG, Mortality, ICU, Vaccination).
- **Intelligent News Curation**: Scrapes health news sites, filters irrelevant content, and generates concise summaries using OpenAI GPT models.
- **Visual Analytics**: Generates trend lines and volume charts.
- **PDF Generation**: Compilation of all insights into a professional, printable one-page PDF.
- **Robust Architecture**: Built on LangChain and LangGraph for reliable agentic orchestration.

## Architecture

The system follows a multi-agent graph architecture:
1.  **Data Specialist Agent**: Fetches hard statistics and calculates key metrics.
2.  **News Curator Agent**: Scrapes web sources and uses LLMs to filter/summarize content.
3.  **Validation Node**: Ensures data integrity and news sufficiency before generation.
4.  **Layout Designer Agent**: Renders the final PDF artifact.

## Prerequisites

- Python 3.10+
- OpenAI API Key (for news summarization)

## Installation

1.  **Clone the repository** (if applicable)
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    # OR if using poetry
    poetry install
    ```
    *Note: If `requirements.txt` is missing, use `pip install langchain langgraph langchain-openai pandas requests beautifulsoup4 weasyprint matplotlib jinja2 pydantic-settings python-dotenv`*

3.  **Configuration**:
    Copy `.env.template` to `.env` and configure your keys:
    ```bash
    cp .env.template .env
    ```
    
    ### LLM Providers
    You can switch between OpenAI and Vertex AI (Gemini) by setting `LLM_PROVIDER`:
    
    **Option A: OpenAI (Default)**
    ```ini
    LLM_PROVIDER=openai
    OPENAI_API_KEY=sk-...
    ```

    **Option B: Google Vertex AI**
    ```ini
    LLM_PROVIDER=vertex
    GOOGLE_PROJECT_ID=my-project-id
    GOOGLE_LOCATION=us-central1
    # Ensure 'gcloud auth application-default login' is run or GOOGLE_APPLICATION_CREDENTIALS is set
    ```

    ### Monitoring (LangSmith)
    To enable deep tracing of the agent pipeline:
    ```ini
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=lsv2_...
    LANGCHAIN_PROJECT=Indicium HealthCare Report
    ```

## Usage

Run the main script to generate a report:

```bash
python main.py --days 30
```

### Options

- `--days <int>`: Number of past days to analyze (default: 30).
- `--help`: Show help message.

## Output

Reports are generated in the `output/` directory with the naming convention `Relatorio_Saude_YYYYMMDD.pdf`.

## Troubleshooting

- **Missing Template**: Ensure `app/templates/report_template.html` exists.
- **API Errors**: Check your `.env` file credentials.
- **Empty News**: If news scraping fails, check your internet connection or update the URL list in `main.py`.

## License

CONFIDENTIAL - Indicium HealthCare Inc.
# indicium_fai_challenge
