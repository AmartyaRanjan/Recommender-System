# DEXTORA Deployment Manual

This guide covers the complete end-to-end deployment of the DEXTORA Recommender System, including model storage, cloud infrastructure, and client usage via WebSockets.

---

## 🏗️ Architecture Overview

-   **Compute**: Google Cloud Run (Serverless Docker Container).
-   **Model Storage**: Google Cloud Storage (GCS).
-   **State/Session**: Google Cloud Memorystore (Redis).
-   **API**: Single WebSocket Endpoint (Request-Response).

---

## 1. Cloud Storage (The Brain)

Your AI models are stored in GCS and loaded when the application starts.

### 1.1 Create a Bucket
Create a bucket (e.g., `gs://dextora-ml-assets`) in the same region as your deployment (e.g., `us-central1`).

### 1.2. Upload Models
Upload your trained artifacts to the root of this bucket:
1.  `saint_weights.pt` (The Transformer Model)
2.  `ppo_student_policy.zip` (The Reinforcement Learning Agent)

> **Note**: If you retrain your models, simply overwrite these files in the bucket and restart your Cloud Run service.

---

## 2. Infrastructure Setup

### 2.1. Redis (Memorystore)
The app needs Redis to map `student_id` to their "Personality Vector".
1.  Create a **Redis instance** in Google Cloud Memorystore.
2.  Note the **IP Address** (e.g., `10.0.0.5`) and **Port** (usually `6379`).
3.  Ensure your Cloud Run service is connected to the same VPC via a "Serverless VPC Access Connector" so it can reach this internal IP.

### 2.2. Build Container
```bash
# Build locally or submit to Cloud Build
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/dextora-app .
```

### 2.3. Deploy to Cloud Run
Deploy the service with the necessary configuration:

```bash
gcloud run deploy dextora-service \
  --image gcr.io/YOUR_PROJECT_ID/dextora-app \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GCS_BUCKET_NAME=dextora-ml-assets \
  --set-env-vars REDIS_HOST=10.0.0.5 \
  --set-env-vars REDIS_PORT=6379 \
  --set-env-vars SECRET_KEY=change-this-in-prod
```

---

## 3. Client Usage (How to get results)

The system uses a **Single WebSocket Endpoint** for real-time interaction.

### Connection Details
-   **URL**: `wss://your-service-url.run.app/ws/{student_id}`
-   **Method**: WebSocket (Secure)

### Interaction Protocol (Request-Response)

1.  **Connect**: Open a socket for a specific student (e.g., `wss://.../ws/STU_1001`).
2.  **Send Request**: Send a JSON payload containing the telemetry batch.
3.  **Receive Response**: The server acts immediately and sends back the JSON Result.

#### Example Payload (Input)
```json
{
    "telemetry_batch": [
        {
            "context_id": 101,
            "behavior_id": 2,
            "timestamp": "2026-01-14T12:00:00Z",
            "intensity": 0.8,
            "engagement_metrics": {
                "tab_switches": 0
            }
        }
    ]
}
```

#### Example Response (Output)
```json
{
    "action": {
        "type": "NUDGE",
        "action": "SWITCH_TO_VIDEO",
        "route": "/video_player"
    },
    "dna": {
        "Mastery": 45.2,
        "Frustration": 12.5,
        "Attention_Span": 88.0
    },
    "trends": "✅ Steady Progress"
}
```

---

## 4. Verification

Use the provided python script to test the deployed endpoint:

1.  Open `tests/verify_deployment.py`.
2.  Update the URI:
    ```python
    uri = "wss://dextora-service-xyz.run.app/ws/STU_1001"
    ```
3.  Run the script:
    ```bash
    python tests/verify_deployment.py
    ```
