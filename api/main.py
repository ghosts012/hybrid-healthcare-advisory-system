import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort

# Your custom modules
from api.utils import process_xray, simple_xray_validator
from api.regression import calculate_severity
from api.retriever import PneumoniaRetriever
from api.generator import MedicalReportGenerator

# Load environment variables from .env
load_dotenv()

app = FastAPI(title="Healthcare Advisory Vision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://hybrid-healthcare-advisory-system.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize Models and RAG once at startup ---
try:
    # 1. Vision Model
    ort_session = ort.InferenceSession("models/vision_production_v1.onnx")
    
    # 2. RAG Components
    # Assuming index.faiss and index.pkl are in the 'models' folder
    retriever = PneumoniaRetriever(index_path="models")
    
    # 3. Report Generator (Gemini)
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API_KEY not found in .env file")
    generator = MedicalReportGenerator(api_key=api_key)
    
    print("All Systems (ONNX, FAISS, Gemini) Initialized Successfully")
except Exception as e:
    print(f"Error during initialization: {e}")

@app.post("/predict/severity")
async def predict_severity(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # 1. Read and Preprocess X-ray
    content = await file.read()
    if not simple_xray_validator(content):
        raise HTTPException(
            status_code=400,
            detail="Uploaded image failed chest X-ray validation. Please upload a frontal chest X-ray image (PNG/JPG).",
        )
    input_tensor = process_xray(content)

    # 2. Run Vision Inference (ONNX)
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
    logits = ort_session.run(None, ort_inputs)[0] 
    
    # 3. Clinical Severity Logic
    severity_score = calculate_severity(logits)

    # 4. Categorization
    status = "Normal"
    if severity_score > 70:
        status = "High Risk / Critical"
    elif severity_score > 30:
        status = "Moderate Risk / Review Required"

    # 5. Hybrid RAG Logic: Retrieve clinical context and generate report
    print(f"[DEBUG] Fetching RAG advisory for score: {severity_score}%")
    context = retriever.get_advisory(severity_score, status)
    
    # Generate the professional medical advisory using Gemini
    ai_advisory = generator.generate_final_report(severity_score, status, context)

    return {
        "filename": file.filename,
        "clinical_metrics": {
            "pneumonia_probability": f"{severity_score}%",
            "severity_index": severity_score,
            "status": status
        },
        "rag_advisory": ai_advisory,  # This is the grounded Gemini output
        "disclaimer": "This is an AI-generated advisory based on clinical guidelines. Consult a radiologist."
    }