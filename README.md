# Hybrid Healthcare Advisory System
A distributed AI pipeline for medical diagnostics, combining cloud-accelerated Deep Learning with local inference and Retrieval-Augmented Generation (RAG).

## Distributed Environment Setup
Unlike traditional local setups, this system utilizes a Hybrid Development Architecture:

* Compute Engine: Google Colab T4 GPU (Remote Kernel).

* IDE Interface: VS Code via the Jupyter & Google Colab Extensions.

* Synchronization: Leverages a custom binary stream (Base64) and remote file system mapping to synchronize model artifacts between the cloud runtime and the local project repository.

## Vision Intelligence Module
The system's core is a ResNet-18 architecture fine-tuned for the PneumoniaMNIST (MedMNIST) dataset.

### Technical Challenges & Niche Solutions
During implementation, several non-trivial environment constraints were addressed:

* **Remote-to-Local Serialization**: Encountered "ephemeral storage" limitations where standard files.download() triggers were intercepted by the VS Code sandbox. Resolved by implementing a Base64 binary data stream to force-download serialized weights.

## Roadmap (Completed)
1. **Model Serialization & Optimization (ONNX)** — Production ONNX export (Opset 18) for lightweight local CPU inference via ONNX Runtime.
2. **Clinical Risk Analytics (Regression Layer)** — Softmax-based Clinical Severity Index (0–100%).
3. **Medical RAG Module** — FAISS + Gemini synthesis of grounded clinical advisories.
4. **Clinical Dashboard** — React UI for upload, severity visualization, and Markdown advisory rendering (Phase 4).
5. **API Containerization (CPU Torch)** — FastAPI image with a CPU-only PyTorch wheel so serving does not ship NVIDIA CUDA libraries.

## Repository Structure (For Now)

```text
Healthcare_Advisory_System/
├── api/
├── models/
│   ├── cnn_vision_v1.pth   
│   
├── src/
│   └── vision_module/    # Research & Training notebooks
|         |--train_cnn.ipynb
└── requirements.txt      
```

## 🛠️ Update: Phase 2 - Model Optimization & API Architecture
**Commit Date:** March 11, 2026
**Focus:** Reproducibility, Model Persistence, and Hybrid Inference Layer

### 🚀 Key Technical Milestones

#### 1. Transition to Hybrid Inference Architecture
I have successfully implemented the first two layers of our three-tier integration:
* **CNN Layer:** Feature extraction using ResNet-18 (MedMNIST).
* **Regression Layer:** A stochastic probability mapping that converts raw CNN logits into a **Clinical Severity Index (0-100%)**.
* **Next:** This index will serve as the primary trigger for the RAG-based Advisory System.

#### 2. Validation-Driven Checkpointing
To resolve issues with training variance, I moved away from static epoch-saving to a **Best-Model Checkpoint** strategy. 
* **The Problem:** Standard seeding was inconsistent due to non-deterministic CUDA atomic operations on the T4 GPU.
* **The Solution:** Implemented a stratified validation split (80/20) and a copy-on-improvement protocol. The model now only saves weights when a new global maximum in validation accuracy is achieved.

#### 3. Optimized ONNX Serialization
Successfully resolved the "Hollow Export" issue (0.08 MB files) by bypassing the TorchDynamo engine in favor of a **Legacy Trace Export**.
* **Optimization:** The final production model (`vision_production_v1.onnx`) is now **44.7 MB**, ensuring all weights are baked into the computational graph.
* **Opset Alignment:** Standardized on Opset 18 to prevent version converter crashes.

### 4. Backend Implementation: FastAPI Service Layer
I transitioned from standalone script execution to a dedicated Backend Service architecture.

* Framework: FastAPI (Asynchronous Python Framework).

* Inference Engine: ONNX Runtime (CPU-optimized for low-latency clinical response).

* Workflow: The backend accepts an X-ray image via multipart/form-data, performs image normalization via PIL, executes the ONNX graph, and applies a logistic regression layer to calculate risk.

5. Sample API Output (The Hybrid Result)
The following JSON payload demonstrates the integration of the CNN Features and the Regression Mapping. This response serves as the input for the upcoming RAG module.

```json
{
    "filename": "pneumonia_yes.jpeg",
    "clinical_metrics": {
        "pneumonia_probability": "79.46176147460938%",
        "severity_index": 79.46176147460938,
        "status": "High Risk / Critical"
    },
    "recommendation": "Consult Radiologist immediately"
}
```

---

### 📂 Updated Project Structure
```text
Healthcare_Advisory_System/
├── venv/                 # Virtual environment (Root)
├── api/
│   ├── main.py           # FastAPI entry point
│   ├── utils.py          # Image preprocessing utilities
│   └── regression.py     # Severity Index math logic
├── models/
│   ├── cnn_vision_v1.pth          # Best found PyTorch weights
│   └── vision_production_v1.onnx  # 44MB Production-ready model
├── src/
│   └── vision_module/    # Research & Training notebooks
|        |--train_cnn.ipynb
└── requirements.txt      # Updated with FastAPI & ONNX Runtime
```

## Update: Phase 3 - RAG Integration & Data Engineering
**Commit Date**: March 16, 2026

Focus: Retrieval-Augmented Generation, Bias Mitigation, and LLM Orchestration

### Key Technical Milestones
* Clinical RAG Architecture (The "Advisory" Layer)
I have successfully implemented the final tier of the hybrid system, transforming raw numerical data into structured clinical reports.

* Vector Knowledge Base: Developed a local knowledge store using FAISS (Facebook AI Similarity Search). The system indexes clinical guidelines from specialized PDF literature to ensure all advice is grounded in peer-reviewed protocols.

* Semantic Retrieval: Implemented HuggingFaceEmbeddings (all-MiniLM-L6-v2) to map the model's severity index to the most relevant medical text chunks.

* LLM Orchestration: Integrated the Google Gen AI SDK (`google-genai`) to synthesize vision model output and retrieved clinical context into a professional Markdown report. Current model ID in `api/generator.py`: `gemini-3-flash-preview`.

* Bias Mitigation & Data Engineering
During testing, the model exhibited a high False Positive rate (95% severity for healthy lungs). I resolved this through a two-pronged strategy:

* Balanced Undersampling: Identified a class imbalance in the primary dataset (3:1 ratio). I re-engineered the training pipeline using a Balanced Subset strategy, achieving a 50/50 split between "Normal" and "Pneumonia" cases.

* Input Distribution Calibration: Discovered a "Distribution Shift" between the training environment and the API. I synchronized the ImageNet Normalization (Mean/Std) constants across both `train_cnn.ipynb` and `api/utils.py`.

**Result: False Positive scores on healthy images dropped from 95% to <18%, significantly increasing diagnostic specificity.**

* Optimized API Response (The "Complete" Hybrid Result)
The API now returns a fully synthesized response. The logic flow is:

Vision (ONNX) → Severity Index → FAISS Retrieval → Gemini Synthesis → Final Report.

```json
{
  "filename": "xray.png",
  "clinical_metrics": {
    "pneumonia_probability": "79.4%",
    "severity_index": 79.4,
    "status": "High Risk / Critical"
  },
  "rag_advisory": "## Clinical Advisory\n...",
  "disclaimer": "This is an AI-generated advisory based on clinical guidelines. Consult a radiologist."
}
```

### Updated Project Structure (Phase 3)

```text
Healthcare_Advisory_System/
├── .env                  # API_KEY storage (gitignored)
├── api/
│   ├── generator.py      # Gemini wrapper (google-genai)
│   ├── main.py           # FastAPI entry point (Integrated RAG flow)
│   ├── utils.py          # Calibrated ImageNet Normalization
│   ├── regression.py     # Severity Index math logic
│   └── retriever.py      # FAISS & LangChain retrieval logic
├── dashboard/            # Planned in Phase 3; implemented in Phase 4
├── data/                 # WHO clinical PDF used for RAG (gitignored)
├── models/
│   ├── index.faiss       # FAISS vector index
│   ├── index.pkl         # Metadata storage
│   ├── vision_production_v1.onnx    # Re-balanced production model
│   └── cnn_vision_v1.pth
├── src/
│   ├── rag_module/           # RAG indexing pipeline
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── main.py
│   │   └── pdf_loader.py
│   └── vision_module/
│       └── train_cnn.ipynb
└── requirements.txt      # Added google-genai, faiss-cpu, langchain*
```

---

## Update: Phase 4 - Clinical Dashboard, Input Guardrails & Attribution
**Commit Date:** August 2, 2026  
**Focus:** Clinician-facing UI, API hardening, Markdown/math rendering, and WHO licence compliance

### Key Technical Milestones

#### 1. React Clinical Dashboard (`dashboard/`)
Shipped a Vite + React + TypeScript + Tailwind UI that talks to the FastAPI backend.

* **Upload flow:** Drag/drop or browse PNG/JPG chest X-rays → `POST /predict/severity`
* **Clinical view:** Severity Index gauge (0–100) with risk coloring, status badge, and live X-ray preview
* **Advisory rendering:** `react-markdown` + `remark-gfm` for GFM; `remark-math` + `rehype-katex` + KaTeX so clinical thresholds/symbols render correctly
* **Config:** single root `.env` (see `.env.example`) — FastAPI reads `API_KEY`; Vite loads the same file via `envDir` for `VITE_API_BASE_URL` (default `http://127.0.0.1:8000`)

#### 2. API Hardening for Frontend Integration
* **CORS:** Allowed `http://localhost:5173` and `http://127.0.0.1:5173` for local dashboard development (`api/main.py`)
* **Chest X-ray gate:** `simple_xray_validator` in `api/utils.py` rejects non-X-ray / overly colorful images before ONNX inference (channel saturation + intensity/ROI heuristics)
* **Response contract** used by the UI matches the Phase 3 payload (`rag_advisory` + `disclaimer`)

#### 3. WHO Guideline Attribution (CC BY-NC-SA 3.0 IGO)
RAG grounding uses WHO *Guideline on management of pneumonia and diarrhoea in children up to 10 years of age* (ISBN 9789240103412). Local artifact path: `data/medical_reference_who.pdf` (gitignored; not redistributed by the app).

* UI shows © World Health Organization 2025, title, licence link (**CC BY-NC-SA 3.0 IGO**), source URL, and **accessed 15 March 2026**
* Source: https://www.who.int/publications/i/item/9789240103412
* Licence: https://creativecommons.org/licenses/by-nc-sa/3.0/igo/

#### 4. LLM Note
Report synthesis uses the Google Gen AI SDK with model `gemini-3-flash-preview` (`api/generator.py`).

### How to Run (Local Hybrid Stack)

```bash
# Backend (repo root, venv active; requires .env with API_KEY and local models/)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Frontend
cd dashboard
npm install
npm run dev
```

* Dashboard: http://127.0.0.1:5173  
* API docs: http://127.0.0.1:8000/docs  

### Updated Project Structure (Phase 4)

```text
Hybrid-Healthcare-Advisory-System/
├── .env                         # API_KEY (gitignored)
├── api/
│   ├── main.py                  # FastAPI + CORS + X-ray validation + RAG flow
│   ├── utils.py                 # ImageNet norm + simple_xray_validator
│   ├── regression.py
│   ├── retriever.py             # FAISS retrieval
│   └── generator.py             # Gemini report synthesis
├── dashboard/                   # React clinical UI
│   ├── src/App.tsx              # Upload, gauge, Markdown+KaTeX advisory, WHO citation
│   ├── src/main.tsx
│   └── package.json
├── .env.example                 # API_KEY + VITE_API_BASE_URL (copy to .env)
├── data/                        # WHO PDF (gitignored)
├── models/                      # ONNX + FAISS artifacts (gitignored binaries)
├── src/
│   ├── rag_module/
│   └── vision_module/
└── requirements.txt
```

---

## Update: Phase 5 - API Containerization & CPU Torch Image
**Commit Date:** October 9, 2026  
**Focus:** Reproducible FastAPI deployment, Docker image size, and a serving-only dependency set

### Key Technical Milestones

#### 1. Why a default `pip install torch` blew up the image
The production vision path already uses **ONNX Runtime** (`vision_production_v1.onnx`). PyTorch is only required at serve time because `api/retriever.py` loads **HuggingFaceEmbeddings (`all-MiniLM-L6-v2`)** via sentence-transformers.

Pinning `torch==2.10.0` from PyPI on Linux pulls the **CUDA** wheel plus NVIDIA libraries (cuDNN, cuBLAS, CUDA runtime, Triton, …). That is what pushed a naive image toward **~13 GB**. Training on Colab (T4, CUDA Torch) is unchanged; that stack stays in the original `requirements.txt` / notebook environment.

#### 2. Serving vs training requirements
* **`requirements.txt`** — full local/Colab freeze, including CUDA-capable `torch==2.10.0` as before.
* **`requirements.api.txt`** — same API/RAG pins **without** a Torch line, so Docker cannot accidentally reinstall the CUDA wheel after the CPU install.

Inference behaviour is unchanged: same ONNX graph, same FAISS `index.faiss` / `index.pkl`, same MiniLM embeddings, same Gemini synthesis. No index rebuild.

#### 3. Dockerfile contract
* Base: `python:3.11-slim`
* Install **CPU Torch first**: `torch==2.10.0` from `https://download.pytorch.org/whl/cpu`
* Then `pip install -r requirements.api.txt` (`--no-cache-dir`)
* Build fails unless `torch.__version__` contains `+cpu`
* Copies `api/`, `src/`, `models/`, `data/`; serves with `uvicorn api.main:app --host 0.0.0.0 --port 8000`
* `.dockerignore` keeps `.env`, git metadata, venvs, notebooks, and dashboard `node_modules` out of the build context

**Measured image (Docker Desktop):** disk usage **~2.8 GB**, content size **~664 MB** (previously ~13 GB).

#### 4. Dashboard ↔ container
The React dashboard is **not** inside the image. Local Vite (`http://127.0.0.1:5173`) still calls **`http://127.0.0.1:8000/predict/severity`** (`VITE_API_BASE_URL` optional).

Traffic follows the **published host port**, not the container name. `--name healthcare-api` is only for `docker logs` / `docker stop`. Map **`-p 8000:8000`**. CORS already allows the Vite origins.

Run **detached** (`-d`) so the API stays up when the terminal closes. Foreground `docker run` (especially with `--rm`) stops when that session ends.

### How to Run (Docker API + local dashboard)

Requires a local `.env` with `API_KEY` and the gitignored artifacts under `models/` (ONNX + FAISS) so they are copied into the image.

```bash
# Build (repo root)
docker build -t healthcare-api .

# Confirm CPU wheel
docker run --rm healthcare-api python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available())"
# Expect: 2.10.0+cpu  and  cuda: False

# Serve in the background
docker run -d --name healthcare-api -p 8000:8000 --env-file .env healthcare-api

# Dashboard (separate terminal)
cd dashboard
npm install
npm run dev
```

* Dashboard: http://127.0.0.1:5173  
* API docs: http://127.0.0.1:8000/docs  
* Logs: `docker logs -f healthcare-api`  
* Stop: `docker stop healthcare-api`

The first container start may download **all-MiniLM-L6-v2** from Hugging Face if it is not already cached in the image (needs network). If port 8000 is already taken by local `uvicorn`, stop that process so the dashboard hits the container.

Local non-Docker backend remains: `uvicorn api.main:app --reload --host 0.0.0.0 --port 8000` (Phase 4).

### Updated Project Structure (Phase 5)

```text
Hybrid-Healthcare-Advisory-System/
├── .env                         # API_KEY (gitignored; passed via --env-file)
├── .dockerignore
├── Dockerfile                   # CPU Torch + requirements.api.txt
├── requirements.txt             # Local / Colab freeze (includes torch)
├── requirements.api.txt         # API image deps (Torch installed separately, CPU)
├── api/
├── dashboard/                   # Still run on the host
├── data/                        # Copied into image if present (PDF gitignored)
├── models/                      # ONNX + FAISS copied into image (binaries gitignored)
└── src/
```

