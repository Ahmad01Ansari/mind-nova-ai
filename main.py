import os
import json
import re
import httpx
import datetime
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

# Custom Model Inference
from src.inference_utils import ModelManager
from src.insight_generator import InsightGenerator
from src.crisis_analyzer import CrisisAnalyzer

load_dotenv()

app = FastAPI(title="Nova AI Prediction API")

BRIDGE_SECRET = os.getenv("FASTAPI_BRIDGE_SECRET", "mock_secret")

# Lazy loading of hubs to speed up startup and prevent Render timeouts
_model_hub = None
_insight_hub = None
_crisis_hub = None

def get_model_hub():
    global _model_hub
    if _model_hub is None:
        _model_hub = ModelManager()
    return _model_hub

def get_insight_hub():
    global _insight_hub
    if _insight_hub is None:
        _insight_hub = InsightGenerator()
    return _insight_hub

def get_crisis_hub():
    global _crisis_hub
    if _crisis_hub is None:
        _crisis_hub = CrisisAnalyzer()
    return _crisis_hub

async def verify_bridge_token(x_bridge_secret: str = Header(...)):
    if x_bridge_secret != BRIDGE_SECRET:
        raise HTTPException(status_code=403, detail="Invalid Bridge Secret")

class InferenceBaseRequest(BaseModel):
    userId: str

class ChatRequest(BaseModel):
    prompt: str

class InsightRequest(BaseModel):
    userId: str
    predictionType: str
    modelData: dict
    context: dict
    promptVersion: str = "3.0"

class CrisisRequest(BaseModel):
    text: str

class ToneRequest(BaseModel):
    text: str
    context: Optional[str] = "GROUP_CHAT"

class AnxietyDepressionRequest(InferenceBaseRequest):
    sleep_hours: float = 7.0
    gad2_score: float = 2.0
    phq2_score: float = 2.0
    exercise_freq: float = 3.0
    social_activity: float = 5.0
    online_stress: float = 3.0
    academic_performance: float = 3.0
    family_support: float = 7.0
    screen_time: float = 4.0
    academic_stress: float = 5.0
    financial_stress: float = 5.0
    sleep_quality: float = 5.0
    self_efficacy: float = 5.0
    peer_relationship: float = 5.0
    diet_quality: float = 5.0

class BurnoutRequest(InferenceBaseRequest):
    work_hours: float = 8.0
    sleep_hours: float = 7.0
    screen_time: float = 5.0
    break_frequency: float = 3.0
    stress_level: float = 5.0
    experience_years: float = 2.0
    job_satisfaction: float = 5.0
    social_support: float = 5.0

class StressRequest(InferenceBaseRequest):
    mood_current: float = 5.0
    sleep_hours: float = 7.0
    workload_level: float = 5.0
    job_satisfaction: float = 5.0
    work_hours: float = 8.0
    screen_time: float = 5.0
    age: float = 25.0
    experience_years: float = 2.0
    academic_stress: float = 5.0
    financial_stress: float = 5.0
    social_support: float = 5.0

class DeteriorationLogEntry(BaseModel):
    day: int
    mood: float
    sleep: float
    workload: float

class DeteriorationRequest(InferenceBaseRequest):
    history: List[DeteriorationLogEntry]

# ══════════════════════════════════════════════════════════════════════
#  CLOUD AI INFERENCE HELPER (Replacing Ollama for Server Deployment)
# ══════════════════════════════════════════════════════════════════════

async def generate_with_cloud(prompt: str, system_prompt: str = None, is_json: bool = False):
    """
    Unified cloud inference helper using Groq as primary and NVIDIA as fallback.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "temperature": 0.5,
    }
    
    if is_json:
        payload["response_format"] = {"type": "json_object"}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload
            )
            
            if response.status_code != 200:
                print(f"Groq Error: {response.text}")
                return await _try_nvidia_fallback(messages, is_json)
                
            data = response.json()
            return data["choices"][0]["message"]["content"]
            
    except Exception as e:
        print(f"Cloud AI Generation Failed: {str(e)}")
        # Try fallback anyway
        try:
            return await _try_nvidia_fallback(messages, is_json)
        except:
            raise HTTPException(status_code=500, detail=f"AI Generation Failed: {str(e)}")

async def _try_nvidia_fallback(messages: list, is_json: bool):
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise Exception("NVIDIA Fallback unavailable")
    
    payload = {
        "model": "meta/llama-3.1-70b-instruct",
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 1024
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload
        )
        if response.status_code != 200:
            raise Exception(f"NVIDIA Error: {response.text}")
        
        data = response.json()
        return data["choices"][0]["message"]["content"]

# ══════════════════════════════════════════════════════════════════════
#  PREDICTION PIPELINE
# ══════════════════════════════════════════════════════════════════════

async def process_prediction(request_dict: dict, prediction_type: str, user_id: str):
    model_hub = get_model_hub()
    insight_hub = get_insight_hub()
    
    if prediction_type == "anxiety":
        res = model_hub.predict_anxiety(request_dict)
    elif prediction_type == "depression":
        res = model_hub.predict_depression(request_dict)
    elif prediction_type == "burnout":
        res = model_hub.predict_burnout(request_dict)
    elif prediction_type == "stress":
        res = model_hub.predict_stress(request_dict)
    elif prediction_type == "deterioration":
        res = model_hub.predict_deterioration(request_dict)
    else:
        raise ValueError(f"Unknown prediction type {prediction_type}")

    if "error" in res or "status" in res and res.get("status") == "insufficient_data":
        return {"success": False, "message": res.get("message", res.get("error"))}

    insight_payload = {
        "predictionType": prediction_type,
        "modelData": res,
        "context": request_dict
    }
    insight_res = await insight_hub.generate(insight_payload)
    
    ai_available = insight_res.get("reasonCodes") != ["SYSTEM_FALLBACK"] and "SYSTEM_FALLBACK" not in str(insight_res)

    return {
        "success": True,
        "predictionType": prediction_type,
        "score": res.get("score"),
        "riskLevel": res.get("riskLevel"),
        "confidence": res.get("confidence", "Medium"),
        "inputCompleteness": res.get("inputCompleteness", 92),
        "contributors": res.get("contributors", []),
        "title": insight_res.get("title", f"Your {prediction_type} Check-in"),
        "summary": insight_res.get("summary", ""),
        "why": insight_res.get("why", ""),
        "actions": insight_res.get("actions", []),
        "encouragement": insight_res.get("encouragement", ""),
        "safetyNote": insight_res.get("safetyNote", None),
        "aiAvailable": ai_available,
        "generatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "modelVersion": res.get("modelVersion", "unknown"),
        "pipelineVersion": res.get("pipelineVersion", "phase6.2")
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "mind_nova_ai"}

@app.post("/predict/anxiety", dependencies=[Depends(verify_bridge_token)])
async def analyze_anxiety(request: AnxietyDepressionRequest):
    return await process_prediction(request.model_dump(), "anxiety", request.userId)

@app.post("/predict/depression", dependencies=[Depends(verify_bridge_token)])
async def analyze_depression(request: AnxietyDepressionRequest):
    return await process_prediction(request.model_dump(), "depression", request.userId)

@app.post("/predict/burnout", dependencies=[Depends(verify_bridge_token)])
async def analyze_burnout(request: BurnoutRequest):
    return await process_prediction(request.model_dump(), "burnout", request.userId)

@app.post("/predict/stress", dependencies=[Depends(verify_bridge_token)])
async def analyze_stress(request: StressRequest):
    return await process_prediction(request.model_dump(), "stress", request.userId)

@app.post("/predict/deterioration", dependencies=[Depends(verify_bridge_token)])
async def analyze_deterioration(request: DeteriorationRequest):
    model_hub = get_model_hub()
    insight_hub = get_insight_hub()
    history_list = [h.model_dump() for h in request.history]
    res = model_hub.predict_deterioration(history_list)
    insight_payload = {"predictionType": "deterioration", "modelData": res, "context": {"history": history_list}}
    insight_res = await insight_hub.generate(insight_payload)
    return {
        "success": True,
        "predictionType": "deterioration",
        "score": res.get("score"),
        "riskLevel": res.get("riskLevel"),
        "title": insight_res.get("title", "Deterioration Alert"),
        "summary": insight_res.get("summary", ""),
        "generatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

@app.post("/chat/generate", dependencies=[Depends(verify_bridge_token)])
async def generate_chat(request: ChatRequest):
    reply = await generate_with_cloud(request.prompt, "You are Nova, a compassionate mental health assistant.")
    return {"reply": reply}

@app.post("/analyze/crisis", dependencies=[Depends(verify_bridge_token)])
async def analyze_crisis(request: CrisisRequest):
    crisis_hub = get_crisis_hub()
    return await crisis_hub.analyze(request.text)

@app.post("/analyze/tone", dependencies=[Depends(verify_bridge_token)])
async def analyze_tone(request: ToneRequest):
    system_prompt = """You are Nova's Safety Engine. Analyze for Toxicity, Harassment, Self-harm.
    OUTPUT SCHEMA: {"safe": bool, "label": "SAFE"|"DISTRESS"|"TOXIC", "action": "ALLOW"|"FLAG", "reason": "string"}"""
    try:
        res_text = await generate_with_cloud(f"Analyze: '{request.text}'", system_prompt, is_json=True)
        return json.loads(res_text)
    except:
        return {"safe": True, "label": "SAFE", "action": "ALLOW", "reason": "Fallback"}

# ══════════════════════════════════════════════════════════════════════
#  WEEKLY REPORT AI SUMMARY
# ══════════════════════════════════════════════════════════════════════

class WeeklySummaryRequest(BaseModel):
    userId: str
    metrics: dict

@app.post("/reports/weekly/summarize", dependencies=[Depends(verify_bridge_token)])
async def generate_weekly_summary(request: WeeklySummaryRequest):
    m = request.metrics
    system_prompt = """You are Nova. Write a warm, personalized weekly mental health summary. 
    OUTPUT SCHEMA: {"title": "str", "summary": "str", "whatHelped": "str", "challenges": "str", "recommendations": [], "encouragement": "str"}"""
    
    user_prompt = f"WEEKLY DATA: {json.dumps(m)}. Generate summary JSON."
    
    try:
        res_text = await generate_with_cloud(user_prompt, system_prompt, is_json=True)
        parsed = json.loads(res_text)
        return {
            "title": parsed.get("title", "Your Weekly Insight"),
            "summary": parsed.get("summary", ""),
            "whatHelped": parsed.get("whatHelped", ""),
            "challenges": parsed.get("challenges", ""),
            "recommendations": parsed.get("recommendations", [])[:3],
            "encouragement": parsed.get("encouragement", "Keep going! 🌟")
        }
    except:
        return _build_weekly_fallback(m)

def _build_weekly_fallback(m: dict) -> dict:
    return {
        "title": "Your Weekly Check-in",
        "summary": "We've analyzed your data for the week.",
        "whatHelped": "Consistent check-ins helped track your patterns.",
        "challenges": "Stay mindful of your stress levels.",
        "recommendations": ["Log your mood daily", "Try a meditation"],
        "encouragement": "You're doing great! 🌟"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
