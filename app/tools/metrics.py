from typing import Dict, Any, List
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

class MetricsInput(BaseModel):
    raw_data: Dict[str, Any] = Field(description="Raw data dictionary returned by DatasusTool")

class MetricsTool(BaseTool):
    name: str = "calculate_epidemiological_rates"
    description: str = "Calculates key health metrics (Increase Rate, Mortality, ICU, Vaccination) from raw data."
    args_schema: type[BaseModel] = MetricsInput

    def _run(self, raw_data: Dict[str, Any]) -> Dict[str, float]:
        print("  [METRICS] Calculating rates from raw data...")
        metrics = {}
        
        try:
            # 1. Cases Increase Rate (Last Day vs Previous Day)
            srag = raw_data.get("srag_cases_daily", [])
            if len(srag) >= 2:
                last_day = srag[-1]["cases"]
                prev_day = srag[-2]["cases"]
                if prev_day > 0:
                    increase_rate = (last_day - prev_day) / prev_day
                else:
                    increase_rate = 0.0 # Avoid division by zero
                metrics["cases_increase_rate"] = round(increase_rate * 100, 2) # Percentage
            else:
                metrics["cases_increase_rate"] = 0.0

            # 2. Mortality Rate
            m_data = raw_data.get("mortality_data", {})
            deaths = m_data.get("deaths", 0)
            total_cases = m_data.get("total_cases", 1) # avoid div0
            metrics["mortality_rate"] = round((deaths / total_cases) * 100, 2)

            # 3. ICU Occupation Rate
            icu = raw_data.get("icu_data", {})
            occ = icu.get("occupied_beds", 0)
            total_beds = icu.get("total_beds", 1)
            metrics["icu_occupation_rate"] = round((occ / total_beds) * 100, 2)

            # 4. Vaccination Rate
            vac = raw_data.get("vaccination_data", {})
            vaccinated = vac.get("vaccinated", 0)
            pop = vac.get("population", 1)
            metrics["vaccination_rate"] = round((vaccinated / pop) * 100, 2)

            print(f"  [METRICS] Calculation done: {metrics}")
            return metrics

        except Exception as e:
            print(f"  [METRICS ERROR] Error calculating metrics: {e}")
            # Return safe zeros
            return {
                "cases_increase_rate": 0.0,
                "mortality_rate": 0.0,
                "icu_occupation_rate": 0.0,
                "vaccination_rate": 0.0
            }

    def _arun(self, raw_data: Dict[str, Any]):
        raise NotImplementedError("Async not implemented")
