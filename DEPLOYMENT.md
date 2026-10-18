# Ultimo-MARL Cloud Deployment Guide

This guide covers deployment procedures for running the autonomous `Ultimo-MARL` reinforcement learning agent in containerized and cloud environments.

---

## 1. Local Container Deployment (Docker)

### Build the Image
```bash
docker build -t ultimo-marl:latest .
```

### Run Against Local or Remote Ultimo Server
```bash
# Connect to an Ultimo server running on localhost:8080
docker run --rm --net=host ultimo-marl:latest \
  --host=127.0.0.1 \
  --port=8080 \
  --name=DeepUltimo_Prod \
  --strategy=heuristic

# Using Docker Compose
docker compose up -d
```

---

## 2. Google Cloud Deployment (GCP)

### Option A: Google Cloud Run Jobs (Scheduled or On-Demand Tournaments)
1. Build and push the image to Google Artifact Registry:
   ```bash
   gcloud builds submit --tag gcr.io/<PROJECT_ID>/ultimo-marl:latest
   ```
2. Create and execute the Cloud Run Job:
   ```bash
   gcloud run jobs create ultimo-bot-runner \
     --image gcr.io/<PROJECT_ID>/ultimo-marl:latest \
     --region europe-west1 \
     --args="--host=<ULTIMO_SERVER_IP>,--port=8080,--name=MARL_GCP_Agent,--strategy=heuristic"

   gcloud run jobs execute ultimo-bot-runner --region europe-west1
   ```

### Option B: Background Worker on Google Compute Engine (GCE)
Run as a background systemd service or Docker container on an `e2-micro` / `e2-small` instance inside the same VPC network as the Ultimo game server.
