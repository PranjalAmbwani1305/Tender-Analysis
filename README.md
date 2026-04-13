# ATGDA — AI Tender Analyzer (Streamlit + Pinecone)

## Quick Start (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API keys in .env
# Edit .env and add your Anthropic key

# 3. Run
streamlit run app.py
# Open http://localhost:8501
```

## Docker

```bash
# Build
docker build -t atgda .

# Run
docker run -p 8501:8501 --env-file .env atgda

# Or with docker-compose
docker-compose up
```

## Kubernetes

```bash
# 1. Build and push image
docker build -t your-registry/atgda:latest .
docker push your-registry/atgda:latest

# 2. Create secrets
kubectl create secret generic atgda-secrets \
  --from-literal=ANTHROPIC_API_KEY=sk-ant-... \
  --from-literal=PINECONE_API_KEY=pcsk_... \
  --from-literal=PINECONE_INDEX=atgda-tenders

# 3. Apply manifests
kubectl apply -f k8s/pvc.yaml        # storage
kubectl apply -f k8s/deployment.yaml  # app pods
kubectl apply -f k8s/service.yaml     # service + ingress

# 4. Check status
kubectl get pods
kubectl get services
```

## Environment Variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Claude API key (sk-ant-...) |
| `PINECONE_API_KEY` | Pinecone API key (pcsk_...) |
| `PINECONE_INDEX` | Pinecone index name (default: atgda-tenders) |

## ⚠️ Security Note
Never commit `.env` to git. Use Kubernetes secrets or Docker secrets in production.
