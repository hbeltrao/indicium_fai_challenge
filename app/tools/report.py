import base64
import io
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, List
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
import os

class ReportInput(BaseModel):
    metrics: Dict[str, float] = Field(description="Calculated key metrics")
    srag_data: List[Dict[str, Any]] = Field(description="Raw SRAG daily data for line chart")
    news_items: List[Dict[str, str]] = Field(description="List of curated news items")
    output_path: str = Field(default="output/Relatorio_Saude.pdf", description="Path to save the PDF")

class ReportGeneratorTool(BaseTool):
    name: str = "generate_pdf_report"
    description: str = "Generates a PDF report with charts and news."
    args_schema: type[BaseModel] = ReportInput

    def _generate_charts(self, srag_data: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generates base64 encoded strings for charts."""
        charts = {}
        
        if not srag_data:
            return charts

        df = pd.DataFrame(srag_data)
        df['date'] = pd.to_datetime(df['date'])
        
        # 1. Line Chart: Daily Cases
        plt.figure(figsize=(10, 4))
        plt.plot(df['date'], df['cases'], marker='o', linestyle='-', color='teal')
        plt.title('Daily SRAG Cases')
        plt.xlabel('Date')
        plt.ylabel('Cases')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        charts['line_chart'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        # 2. Bar Chart: Monthly Volume (Aggregating daily data)
        # For simulation, we might only have 30 days, so it might be 1-2 bars.
        df['month'] = df['date'].dt.to_period('M')
        monthly = df.groupby('month')['cases'].sum()
        
        plt.figure(figsize=(10, 4))
        monthly.plot(kind='bar', color='coral')
        plt.title('Monthly Case Volume')
        plt.xlabel('Month')
        plt.ylabel('Total Cases')
        plt.xticks(rotation=0)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        charts['bar_chart'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        return charts

    def _run(self, metrics: Dict[str, float], srag_data: List[Dict[str, Any]], news_items: List[Dict[str, str]], output_path: str = "output/Report.pdf") -> str:
        print(f"  [REPORT] Generating PDF report at {output_path}...")
        
        try:
            # 1. Generate Charts
            print("    -> Generating charts...")
            charts = self._generate_charts(srag_data)
            
            # 2. Prepare Template Context
            context = {
                "metrics": metrics,
                "charts": charts,
                "news": news_items,
                "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
            }

            # 3. Render HTML (Assuming template exists in app/templates)
            template_dir = os.path.join(os.getcwd(), "app/templates")
            print(f"    -> Loading template from {template_dir}")
            
            # Create a simple default template if file logic fails or just for safety
            # Ideally we load from file, but for the tool I'll inline a fallback or expect file.
            # I will implement the HTML template file in the next step, so here I assume it exists.
            
            try:
                env = Environment(loader=FileSystemLoader(template_dir))
                template = env.get_template("report_template.html")
                html_content = template.render(context)
            except Exception as e_tpl:
                print(f"    [WARN] Template load failed ({e_tpl}). Using specific fallback.")
                # Fallback simple HTML if template missing
                html_content = f"<h1>Health Report</h1><p>Metrics: {metrics}</p><p>Error loading template.</p>"

            # 4. Generate PDF
            # Ensure output dir exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            HTML(string=html_content).write_pdf(output_path)
            
            print("  [REPORT] PDF generated successfully.")
            return f"Report generated successfully at {output_path}"
        except Exception as e:
            print(f"  [REPORT ERROR] Failed to generate PDF: {e}")
            return f"Error generating report: {str(e)}"

    def _arun(self, metrics: Dict, srag_data: List, news_items: List, output_path: str):
        raise NotImplementedError("Async not implemented")
