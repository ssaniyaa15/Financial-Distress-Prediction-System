# Cloud deployment (Google Cloud Run)

The local FastAPI services map 1:1 onto Cloud Run. Kubernetes is not required.

```
User
  → Cloud Run (fusion_service / API + frontend)
       → LangGraph orchestrator
       → Cloud Run (ml_service)
       → Cloud Run (ocr_service)
       → Cloud Run (vision_service)
       → Cloud Run (llm_service)
       → Cloud Run (data_service)
  → Cloud Storage  (uploaded PDFs / charts)
  → Cloud SQL Postgres or Firestore (prediction history)
```

## Prerequisites

* `gcloud` CLI authenticated to a project
* Trained model artifacts in `models/`
* Processed dataset in `data/processed/`
* Secret Manager values for `OPENAI_API_KEY` / `GEMINI_API_KEY` / `GROQ_API_KEY`

## One-time setup

```bash
PROJECT_ID=your-gcp-project
REGION=us-central1
gcloud config set project $PROJECT_ID

gcloud services enable run.googleapis.com artifactregistry.googleapis.com sqladmin.googleapis.com storage.googleapis.com secretmanager.googleapis.com

gcloud artifacts repositories create finance-risk \
  --repository-format=docker --location=$REGION

gcloud storage buckets create gs://${PROJECT_ID}-finance-risk --location=$REGION
```

Optional Postgres:

```bash
gcloud sql instances create finance-risk-db --database-version=POSTGRES_15 \
  --cpu=1 --memory=4GiB --region=$REGION
gcloud sql databases create risk --instance=finance-risk-db
```

Set `DATABASE_URL` to the Cloud SQL connection string, or keep SQLite for a demo.

## Build and deploy each service

From the repository root:

```bash
REGION=us-central1
REPO=${REGION}-docker.pkg.dev/$PROJECT_ID/finance-risk

for SVC in data_service ml_service ocr_service vision_service llm_service fusion_service; do
  docker build -f services/${SVC}/Dockerfile -t $REPO/${SVC}:latest .
  docker push $REPO/${SVC}:latest
done

# Data / ML / Vision / Fusion need more memory because they load models.
gcloud run deploy data-service --image $REPO/data_service:latest --region $REGION --allow-unauthenticated --port 8001 --memory 1Gi
gcloud run deploy ml-service --image $REPO/ml_service:latest --region $REGION --allow-unauthenticated --port 8002 --memory 2Gi
gcloud run deploy ocr-service --image $REPO/ocr_service:latest --region $REGION --allow-unauthenticated --port 8003 --memory 1Gi
gcloud run deploy vision-service --image $REPO/vision_service:latest --region $REGION --allow-unauthenticated --port 8004 --memory 2Gi
gcloud run deploy llm-service --image $REPO/llm_service:latest --region $REGION --allow-unauthenticated --port 8005 --set-secrets OPENAI_API_KEY=OPENAI_API_KEY:latest,GEMINI_API_KEY=GEMINI_API_KEY:latest,GROQ_API_KEY=GROQ_API_KEY:latest
gcloud run deploy fusion-service --image $REPO/fusion_service:latest --region $REGION --allow-unauthenticated --port 8000 --memory 2Gi \
  --set-env-vars GCS_BUCKET=${PROJECT_ID}-finance-risk
```

Point the fusion service at the other Cloud Run URLs with `--set-env-vars DATA_SERVICE_URL=...,ML_SERVICE_URL=...`.

## Storage usage

* Upload PDFs/images to `gs://${PROJECT_ID}-finance-risk/uploads/`
* Keep trained `models/` in the container image or a mounted GCS FUSE bucket
* Store prediction JSON in Cloud SQL (`predictions` table created by `shared/store.py`) or Firestore

## What not to do

Do not introduce GKE for this project. Six Cloud Run services plus Storage and one database are enough to show a production-shaped deployment.
