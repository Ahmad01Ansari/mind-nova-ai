import json
import os
from typing import Dict, List, Any
from .cloud_ai import generate_with_cloud

class InsightGenerator:
    def __init__(self):
        pass

    async def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a structured AI insight using cloud providers.
        """
        prediction_type = payload.get("predictionType", "STRESS")
        model_data = payload.get("modelData", {})
        context = payload.get("context", {})
        
        # 1. Determine Tone and System Rules
        risk_level = model_data.get("riskLevel", "MINIMAL").upper()
        tone = self._get_tone(risk_level)
        
        system_prompt = f"""You are a Clinical AI Product Architect for MindNova.
Your task is to generate a {tone} and actionable insight report based on physiological and behavioral prediction data.

STRICT TERMINOLOGY GUARD:
- You MUST focus the report EXCLUSIVELY on the predicted condition: {prediction_type.upper()}.
- If the type is ANXIETY, use terms like "anxiety patterns", "worry", or "nervousness".
- If the type is BURNOUT, use terms like "exhaustion", "burnout", or "depersonalization".
- If the type is STRESS, use "stress signals" or "tension".
- NEVER substitute the prediction type with a different one (e.g., do NOT say "stress patterns" in an Anxiety report).

CRITICAL CLINICAL GUARDRAILS:
- NEVER provide a medical diagnosis. 
- NEVER say "You have depression", "You are mentally ill", or "You are diagnosed".
- INSTEAD say "Your responses may indicate elevated risk", "Current signals suggest {prediction_type.lower()} patterns", or "Consider professional help if persistent".
- NEVER contradict the risk level. If risk is HIGH or SEVERE, do NOT say "You are doing great" or "LOW".
- Keep the tone warm, intelligent, and supportive.
- RESPONSE MUST BE STRICT VALID JSON.

OUTPUT SCHEMA:
{{
  "title": "Supportive and accurate headline (e.g., 'Elevated Stress Signals Detected')",
  "summary": "Clear, non-diagnostic interpretation of the score (max 2 sentences).",
  "why": "Explanation of why this result happened based on the contributing factors.",
  "actions": ["Actionable step 1", "Actionable step 2", "Actionable step 3"],
  "encouragement": "Short, warm closing encouragement.",
  "safetyNote": "Include a safety note urging professional support ONLY IF risk is HIGH or SEVERE. Otherwise, output null."
}}
"""

        user_prompt = f"""
INPUT DATA:
- Prediction Type: {prediction_type}
- Model Score (0-100): {model_data.get('score')}
- Risk Level: {risk_level}
- Contributing Factors: {", ".join(model_data.get('contributors', []))}
- User Context: {json.dumps(context)}

Generate the JSON response matching the schema.
"""

        try:
            res_text = await generate_with_cloud(user_prompt, system_prompt, is_json=True)
            parsed = json.loads(res_text)
            print(f"✨ AI Insight Generated: {parsed.get('title')}")
            return parsed

        except Exception as e:
            print(f"Insight Generation Failed: {str(e)}")
            return self._get_fallback(prediction_type, model_data)

    def _get_tone(self, risk_level: str) -> str:
        tones = {
            "MINIMAL": "encouraging and positive",
            "MILD": "supportive and calm",
            "MODERATE": "focused and supportive",
            "HIGH": "serious and grounded",
            "SEVERE": "serious, crisis-focused, and urgent"
        }
        return tones.get(risk_level, "supportive")

    def _get_fallback(self, p_type: str, data: Dict) -> Dict:
        """
        Robust dynamic fallback mimicking the exact LLM JSON output.
        """
        risk = data.get("riskLevel", "MODERATE").upper()
        score = data.get("score", 50)
        contributors = data.get("contributors", ["recent patterns"])
        
        title = f"{p_type.capitalize()} Check-in"
        summary = f"Your current score of {score} indicates a {risk.lower()} level of {p_type.lower()}."
        why = f"This result is primarily influenced by: {', '.join(contributors)}."
        
        actions = [
            "Take a short break to reset your nervous system.",
            "Review your daily routines to identify stressors.",
            "Prioritize consistent, high-quality sleep."
        ]
        
        if risk in ["HIGH", "SEVERE"]:
            title = f"Elevated {p_type.capitalize()} Signals Detected"
            actions = [
                "Consider stepping away from current stressors immediately.",
                "Reach out to a trusted friend or support system.",
                "Prioritize absolute rest and recovery tonight."
            ]
            
        return {
            "title": title,
            "summary": summary,
            "why": why,
            "actions": actions,
            "encouragement": "We are here to support your journey. Take it one step at a time.",
            "safetyNote": "Consider professional support if these feelings persist." if risk in ["HIGH", "SEVERE"] else None
        }

    async def generate_journal_insight(self, content: str, mood_state: str = None) -> Dict[str, Any]:
        """
        Analyzes a journal entry for emotional tone, triggers, and reflection.
        """
        system_prompt = """You are Nova, an empathetic AI Journal Guide. 
Analyze the user's journal entry and provide emotional insights.
STRICT RULES:
- Output MUST be valid JSON.
- Be compassionate but objective.
- Identify potential emotional triggers (people, places, topics).
- Give an emotional score (1-10) where 1 is deep distress and 10 is peak joy.

OUTPUT SCHEMA:
{
  "tone": "One word (e.g. Reflective, Anxious, Joyful, Somber)",
  "emotionalScore": float,
  "summary": "A 1-sentence summary of the entry's emotional core.",
  "suggestedAction": "One small wellness action based on the content.",
  "detectedTriggers": ["trigger1", "trigger2"]
}
"""
        user_prompt = f"JOURNAL ENTRY: {content}\nMOOD AT TIME: {mood_state}\nGenerate insights JSON."

        try:
            res_text = await generate_with_cloud(user_prompt, system_prompt, is_json=True)
            return json.loads(res_text)
        except Exception as e:
            print(f"Journal Insight Generation Failed: {str(e)}")
            return {
                "tone": "Reflective",
                "emotionalScore": 5.0,
                "summary": "You've shared your thoughts with Nova.",
                "suggestedAction": "Take a deep breath and acknowledge your feelings.",
                "detectedTriggers": []
            }
