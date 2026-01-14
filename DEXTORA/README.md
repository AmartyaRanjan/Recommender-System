# DEXTORA Recommender System

A real-time, personality-adaptive recommender system for education, built with FastAPI, PyTorch (SAINT+PPO), and Redis.

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Redis (running locally or via Docker)
- PostgreSQL (optional for full features)

### Installation

1.  Clone the repository.
2.  Create a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Setup Environment:
    ```bash
    cp .env.example .env
    # Edit .env with your local config
    ```

### Running Locally

```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

## ☁️ Deployment & Architecture

This system is designed to run on **Google Cloud Run**.
Please refer to **[DEPLOYMENT_MANUAL.md](DEPLOYMENT_MANUAL.md)** for detailed instructions on:
-   Infrastructure Setup (Redis, GCS)
-   Model Management
-   API Usage (WebSocket Protocol)

## 🧠 AI Models

The application attempts to download the latest models from Google Cloud Storage on startup.
- Set `GCS_BUCKET_NAME` in your `.env` to enable this.
- If running offline, you can manually place `saint_weights.pt` and `ppo_student_policy.zip` in `app/ml_assets/`.

## 🔌 API Interaction

The primary interface is a WebSocket endpoint:
`ws://localhost:8080/ws/{student_id}`

See `DEPLOYMENT_MANUAL.md` for the JSON payload format.
