import json
import os
from typing import Dict, List, Any
from .cloud_ai import generate_with_cloud

class CrisisAnalyzer:
    def __init__(self):
        pass

    async def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyzes provided text for crisis indicators and returns risk level and suggestions.
        Uses cloud-native AI with failover.
        """
        system_prompt = """You are a MindNova Crisis Detection System.
Your task is to analyze user text for indicators of mental health crisis, self-harm intent, or extreme emotional distress.

RULES:
- RESPONSE MUST BE STRICT VALID JSON.
- CATEGORIZE risk scale: LOW, MED, HIGH, SEVERE, EMERGENCY.
- IDENTIFY category: SUICIDE, DEPRESSION, BURNOUT, ANXIETY, OTHER.
- BE CONSERVATIVE: If in doubt, lean towards a higher risk category for safety.
- Provide immediate, simple grounding suggestions.
- DO NOT provide a medical diagnosis.
"""

        user_prompt = f"""
USER TEXT FOR ANALYSIS:
"{text}"

OUTPUT SCHEMA:
{{
  "riskLevel": "LOW | MED | HIGH | SEVERE | EMERGENCY",
  "category": "SUICIDE | DEPRESSION | BURNOUT | ANXIETY | OTHER",
  "analysis": "Short technical justification",
  "suggestions": ["Immediate grounding technique 1", "Immediate grounding technique 2"]
}}
"""

        try:
            res_text = await generate_with_cloud(user_prompt, system_prompt, is_json=True)
            return json.loads(res_text)

        except Exception as e:
            print(f"Crisis Analysis Cloud Request Failed: {str(e)}")
            # Safety Fallback: Check for obvious keywords locally
            lower_text = text.lower()
            if any(w in lower_text for w in ["die", "kill", "suicide", "end it", "hurt myself"]):
                return {
                    "riskLevel": "SEVERE",
                    "category": "SUICIDE",
                    "analysis": "Keyword detection triggered during AI outage.",
                    "suggestions": [
                        "Please call the 102 Lifeline immediately.",
                        "Reach out to your primary support contact."
                    ]
                }
            
            return {
                "riskLevel": "MED",
                "category": "OTHER",
                "analysis": "LLM Analysis failed. System defaulting to MED risk as a safety precaution.",
                "suggestions": [
                    "Take five slow, deep breaths.",
                    "Look around and name five things you can see."
                ]
            }
