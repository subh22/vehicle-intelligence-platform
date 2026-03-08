# Vehicle Intelligence Platform

A **multi-modal AI system** that automatically creates vehicle service records by combining computer vision, LLM reasoning, and structured data engineering.

---

## System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                  Vehicle Intelligence API                  │
│                       (FastAPI)                            │
└──────────┬────────────────┬───────────────┬───────────────┘
           │                │               │
   ┌───────▼──────┐ ┌───────▼──────┐ ┌─────▼────────────┐
   │   Vehicle    │ │   Damage     │ │  Customer Intent  │
   │  Classifier  │ │  Detector    │ │    Extractor      │
   │ (EfficientNet│ │(EfficientNet │ │  (Claude claude-haiku-4-5-20251001) │
   │   -B0)       │ │ -B0 / Claude)│ │                   │
   └───────┬──────┘ └───────┬──────┘ └─────┬────────────┘
           └────────────────┼───────────────┘
                            │
                 ┌──────────▼──────────┐
                 │  Metadata Processor  │
                 │   (CarDekho CSV)     │
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │  Multi-Modal Fusion  │
                 │     Pipeline         │
                 └──────────┬──────────┘
                            │
              ┌─────────────▼──────────────┐
              │       Service Record        │
              │  {vehicle_type, damages,    │
              │   customer_intent, priority}│
              └────────────────────────────┘
```

---

## Datasets Used

| Dataset | Source | Purpose |
|---------|--------|---------|
| Vehicle Classification (~5600 imgs, 7 classes) | Kaggle: `marquis03/vehicle-classification` | Train vehicle type classifier |
| Car Damage Detection (stage1 + stage2) | Kaggle: `eashankaushik/car-damage-detection` | Train damage classifier |
| CarDekho Vehicle Metadata CSV | Kaggle: `nehalbirla/vehicle-dataset-from-cardekho` | Structured metadata fusion |
| Customer Support on Twitter | Kaggle: `thoughtvector/customer-support-on-twitter` | Intent classification reference |

---

## Tech Stack

- **CV**: PyTorch + EfficientNet-B0 (torchvision) + Albumentations
- **LLM**: Anthropic Claude claude-haiku-4-5-20251001 (intent extraction + CV fallback)
- **API**: FastAPI + Uvicorn
- **Data**: Pandas, scikit-learn
- **Container**: Docker + docker-compose

---

## Quick Start

### 1. Clone & install dependencies

```bash
git clone <repo-url>
cd vehicle-intelligence-platform
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY
```

### 3. Download datasets

```bash
# Install Kaggle CLI and authenticate first
pip install kaggle
# Place kaggle.json in ~/.kaggle/

python scripts/download_datasets.py --all
```

### 4. Train models

```bash
# Vehicle type classifier
python -m training.train_classifier --epochs 20 --batch-size 32

# Damage detector (stage 2 — multi-class)
python -m training.train_damage_detector --stage 2 --epochs 20
```

> **Skip training**: Without trained models the system falls back to Claude Vision API automatically.

### 5. Start the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI: **http://localhost:8000/docs**

---

## API Reference

### `POST /api/v1/analyze` — Full Multi-Modal Analysis

**Request** (multipart/form-data):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | file | Yes | JPEG/PNG vehicle image (CCTV frame) |
| `customer_text` | string | Yes | Customer's service request text |
| `metadata_json` | string | No | JSON string matching VehicleMetadataIn schema |

**Example (curl)**:

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "image=@/path/to/vehicle.jpg" \
  -F "customer_text=My car was rear-ended. I need an insurance claim." \
  -F 'metadata_json={"year":2020,"km_driven":55000,"fuel_type":"Petrol"}'
```

**Response**:

```json
{
  "vehicle_type": "SUV",
  "detected_damage": ["rear bumper dent"],
  "customer_intent": "insurance_claim",
  "service_priority": "high",
  "vehicle_confidence": 0.92,
  "damage_severity": "moderate",
  "urgency_level": "high",
  "key_customer_concerns": ["rear bumper damage", "insurance documentation"],
  "damage_details": [
    {
      "damage_type": "dent",
      "location": "rear bumper",
      "severity": "moderate",
      "confidence": 0.87
    }
  ],
  "vehicle_metadata": {"year": 2020, "km_driven": 55000, "mileage_category": "medium"},
  "processing_time_ms": 1243.5,
  "classifier_source": "local_model",
  "damage_source": "claude_vision"
}
```

### `POST /api/v1/analyze/image` — Image Only

Returns vehicle type + damage detection without customer text.

### `POST /api/v1/analyze/text` — Text Only

Returns intent extraction from customer text alone.

### `GET /health` — Health Check

```json
{
  "status": "ok",
  "components": {
    "vehicle_classifier": "local_model",
    "damage_detector": "claude_vision_fallback",
    "llm_intent_extractor": "ready",
    "metadata_processor": "ready"
  },
  "version": "1.0.0"
}
```

---

## CLI Inference

Run without starting the server:

```bash
python scripts/run_inference.py \
  --image path/to/vehicle.jpg \
  --text "Windshield shattered in an accident. Need urgent repair." \
  --metadata '{"year": 2019, "km_driven": 72000}'
```

---

## Service Priority Rules

| Condition | Priority |
|-----------|----------|
| Emergency / insurance_claim intent | HIGH |
| Severe damage (shatter / dislocation) | HIGH |
| Repair / warranty intent | MEDIUM |
| Moderate damage (dent / scratch) | MEDIUM |
| Regular service / inspection | LOW |

---

## Docker Deployment

```bash
# Build and start
docker-compose up --build

# Or pull and run directly
docker build -t vehicle-intelligence-platform .
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=your_key \
  -v $(pwd)/models:/app/models \
  vehicle-intelligence-platform
```

---

## Running Tests

```bash
# Unit tests (no API key required)
pytest tests/ -v

# Include integration tests (requires ANTHROPIC_API_KEY)
pytest tests/ -v -m integration
```

---

## Project Structure

```
vehicle-intelligence-platform/
├── api/
│   ├── main.py                  # FastAPI app, middleware, startup
│   ├── schemas.py               # Pydantic request/response models
│   └── routes/
│       ├── health.py            # GET /health
│       └── vehicle.py           # POST /api/v1/analyze (and variants)
├── core/
│   ├── vision/
│   │   ├── vehicle_classifier.py   # EfficientNet-B0 + Claude Vision fallback
│   │   └── damage_detector.py      # EfficientNet-B0 + Claude Vision fallback
│   ├── nlp/
│   │   └── intent_extractor.py     # Claude claude-haiku-4-5-20251001 intent extraction
│   ├── data/
│   │   └── metadata_processor.py   # CarDekho CSV loader + enrichment
│   └── pipeline.py              # Multi-modal fusion + priority logic
├── training/
│   ├── data_prep.py             # Dataset loaders, Albumentations augmentation
│   ├── train_classifier.py      # Vehicle type classifier training
│   └── train_damage_detector.py # Damage classifier training (stage1 & stage2)
├── tests/
│   └── test_pipeline.py         # Unit + integration tests
├── scripts/
│   ├── download_datasets.py     # Kaggle dataset downloader
│   └── run_inference.py         # CLI inference tool
├── models/                      # Saved model weights (gitignored)
├── data/                        # Datasets (gitignored)
├── config.py                    # Centralised settings (pydantic-settings)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Evaluation Checklist

| Criterion | Implementation |
|-----------|---------------|
| Computer Vision | EfficientNet-B0 for vehicle classification + damage detection |
| LLM Integration | Claude claude-haiku-4-5-20251001 for intent extraction + CV fallback |
| Multi-modal Reasoning | Pipeline fuses CV + LLM + metadata → priority |
| Pipeline Building | `VehicleIntelligencePipeline` orchestrates all components |
| Working API | FastAPI with `/analyze`, `/analyze/image`, `/analyze/text` |
| Real-time Inference | Single HTTP request → structured JSON in ~1–3s |
