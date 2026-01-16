# Deployment Guide

This guide covers different deployment options for the RAG-Anything service.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Google Cloud Run](#google-cloud-run)
- [Kubernetes](#kubernetes)
- [Environment Configuration](#environment-configuration)
- [Monitoring Setup](#monitoring-setup)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Services

1. **Firebase Project**
   - Firestore database enabled
   - Service account with Firestore access
   - Authentication configured (if using user auth)

2. **Google Cloud Storage**
   - Bucket for document storage
   - Service account with Storage Object Viewer permissions

3. **OpenAI API**
   - Valid API key
   - Sufficient quota for document processing

### System Requirements

- **CPU**: 2+ cores recommended
- **Memory**: 4GB+ RAM recommended
- **Storage**: 10GB+ for temporary files and storage
- **Network**: Stable internet connection for API calls

## Local Development

### Quick Start

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd law-unleashed-rag
   cp env.example .env
   ```

2. **Configure environment**:
   ```bash
   # Edit .env file
   FIREBASE_PROJECT_ID=your-firebase-project-id
   GCS_BUCKET_NAME=your-gcs-bucket-name
   OPENAI_API_KEY=your-openai-api-key
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
   ```

3. **Install dependencies**:
   ```bash
   # Install system dependencies (Ubuntu/Debian)
   sudo apt-get update
   sudo apt-get install -y libreoffice tesseract-ocr tesseract-ocr-eng
   
   # Install Python dependencies
   pip install -r requirements.txt
   ```

4. **Start the service**:
   ```bash
   ./start_rag_service.sh
   ```

### Manual Setup

If you prefer manual setup:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create storage directories
mkdir -p storage/kv storage/vector storage/graph storage/status temp

# Start the service
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

## Docker Deployment

### Single Container

1. **Build the image**:
   ```bash
   docker build -t rag-anything-service .
   ```

2. **Run the container**:
   ```bash
   docker run -d \
     --name rag-service \
     -p 8000:8000 \
     -e FIREBASE_PROJECT_ID=your-project-id \
     -e GCS_BUCKET_NAME=your-bucket \
     -e OPENAI_API_KEY=your-api-key \
     -v /path/to/service-account.json:/app/credentials.json \
     -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json \
     -v rag_storage:/app/storage \
     -v rag_temp:/app/temp \
     rag-anything-service
   ```

### Docker Compose

1. **Configure environment**:
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

2. **Start services**:
   ```bash
   # Start only the RAG service
   docker-compose up -d rag-anything-service
   
   # Start with monitoring
   docker-compose --profile monitoring up -d
   ```

3. **Access services**:
   - RAG Service: http://localhost:8000
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000 (admin/admin)

## Google Cloud Run

### Prerequisites

- Google Cloud SDK installed
- Project with Cloud Run API enabled
- Service account with necessary permissions

### Deployment Steps

1. **Build and push image**:
   ```bash
   # Set variables
   PROJECT_ID=your-gcp-project-id
   SERVICE_NAME=rag-anything-service
   REGION=us-central1
   
   # Build and push
   gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME
   ```

2. **Deploy to Cloud Run**:
   ```bash
   gcloud run deploy $SERVICE_NAME \
     --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
     --platform managed \
     --region $REGION \
     --allow-unauthenticated \
     --memory 4Gi \
     --cpu 2 \
     --timeout 3600 \
     --set-env-vars FIREBASE_PROJECT_ID=$PROJECT_ID \
     --set-env-vars GCS_BUCKET_NAME=your-bucket \
     --set-env-vars OPENAI_API_KEY=your-api-key \
     --set-env-vars GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json
   ```

3. **Set up service account**:
   ```bash
   # Create service account
   gcloud iam service-accounts create rag-service-account
   
   # Grant permissions
   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member="serviceAccount:rag-service-account@$PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/datastore.user"
   
   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member="serviceAccount:rag-service-account@$PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/storage.objectViewer"
   
   # Deploy with service account
   gcloud run deploy $SERVICE_NAME \
     --service-account=rag-service-account@$PROJECT_ID.iam.gserviceaccount.com \
     --image gcr.io/$PROJECT_ID/$SERVICE_NAME
   ```

### Cloud Run with Custom Domain

```bash
# Map custom domain
gcloud run domain-mappings create \
  --service $SERVICE_NAME \
  --domain rag-api.yourdomain.com \
  --region $REGION
```

## Kubernetes

### Prerequisites

- Kubernetes cluster (GKE, EKS, AKS, or local)
- kubectl configured
- Helm (optional, for easier deployment)

### Basic Deployment

1. **Create namespace**:
   ```bash
   kubectl create namespace rag-service
   ```

2. **Create secrets**:
   ```bash
   # Create secret for service account
   kubectl create secret generic rag-credentials \
     --from-file=credentials.json=path/to/service-account.json \
     --namespace=rag-service
   
   # Create secret for environment variables
   kubectl create secret generic rag-config \
     --from-literal=FIREBASE_PROJECT_ID=your-project-id \
     --from-literal=GCS_BUCKET_NAME=your-bucket \
     --from-literal=OPENAI_API_KEY=your-api-key \
     --namespace=rag-service
   ```

3. **Deploy application**:
   ```yaml
   # k8s-deployment.yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: rag-anything-service
     namespace: rag-service
   spec:
     replicas: 2
     selector:
       matchLabels:
         app: rag-anything-service
     template:
       metadata:
         labels:
           app: rag-anything-service
       spec:
         containers:
         - name: rag-service
           image: gcr.io/your-project/rag-anything-service:latest
           ports:
           - containerPort: 8000
           env:
           - name: GOOGLE_APPLICATION_CREDENTIALS
             value: /app/credentials.json
           - name: FIREBASE_PROJECT_ID
             valueFrom:
               secretKeyRef:
                 name: rag-config
                 key: FIREBASE_PROJECT_ID
           - name: GCS_BUCKET_NAME
             valueFrom:
               secretKeyRef:
                 name: rag-config
                 key: GCS_BUCKET_NAME
           - name: OPENAI_API_KEY
             valueFrom:
               secretKeyRef:
                 name: rag-config
                 key: OPENAI_API_KEY
           volumeMounts:
           - name: credentials
             mountPath: /app/credentials.json
             subPath: credentials.json
             readOnly: true
           - name: storage
             mountPath: /app/storage
           - name: temp
             mountPath: /app/temp
           resources:
             requests:
               memory: "2Gi"
               cpu: "1"
             limits:
               memory: "4Gi"
               cpu: "2"
         volumes:
         - name: credentials
           secret:
             secretName: rag-credentials
         - name: storage
           persistentVolumeClaim:
             claimName: rag-storage-pvc
         - name: temp
           emptyDir: {}
   ---
   apiVersion: v1
   kind: Service
   metadata:
     name: rag-anything-service
     namespace: rag-service
   spec:
     selector:
       app: rag-anything-service
     ports:
     - port: 80
       targetPort: 8000
     type: LoadBalancer
   ---
   apiVersion: v1
   kind: PersistentVolumeClaim
   metadata:
     name: rag-storage-pvc
     namespace: rag-service
   spec:
     accessModes:
       - ReadWriteOnce
     resources:
       requests:
         storage: 10Gi
   ```

4. **Apply deployment**:
   ```bash
   kubectl apply -f k8s-deployment.yaml
   ```

### Helm Chart (Optional)

```bash
# Create Helm chart
helm create rag-anything-service

# Install with custom values
helm install rag-service ./rag-anything-service \
  --namespace rag-service \
  --set image.repository=gcr.io/your-project/rag-anything-service \
  --set image.tag=latest \
  --set service.type=LoadBalancer \
  --set resources.requests.memory=2Gi \
  --set resources.requests.cpu=1
```

## Environment Configuration

### Required Variables

```bash
# Core Configuration
FIREBASE_PROJECT_ID=your-firebase-project-id
GCS_BUCKET_NAME=your-gcs-bucket-name
OPENAI_API_KEY=your-openai-api-key

# Service Configuration
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info
ENVIRONMENT=production

# Storage Configuration
RAG_STORAGE_DIR=/app/storage
RAG_TEMP_DIR=/app/temp
```

### Optional Variables

```bash
# OpenAI Configuration
OPENAI_BASE_URL=https://api.openai.com/v1

# Parser Configuration
PARSER=mineru
PARSE_METHOD=auto

# GPU Configuration (for MinerU)
CUDA_VISIBLE_DEVICES=0
DEVICE=cuda:0

# LibreOffice Configuration
LIBREOFFICE_PATH=/usr/bin/libreoffice

# Tesseract Configuration
TESSERACT_PATH=/usr/bin/tesseract
TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
```

## Monitoring Setup

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'rag-anything-service'
    static_configs:
      - targets: ['rag-anything-service:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### Grafana Dashboard

Import the following dashboard configuration:

```json
{
  "dashboard": {
    "title": "RAG-Anything Service",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(rag_anything_http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(rag_anything_http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Processing Duration",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(rag_anything_document_processing_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

### Health Checks

Configure health checks for your deployment:

```bash
# Kubernetes health check
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health/detailed
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
```

## Troubleshooting

### Common Issues

1. **Service won't start**:
   ```bash
   # Check logs
   docker logs rag-service
   kubectl logs -f deployment/rag-anything-service
   
   # Check environment variables
   docker exec rag-service env | grep -E "(FIREBASE|GCS|OPENAI)"
   ```

2. **Firebase connection issues**:
   ```bash
   # Verify service account
   gcloud auth activate-service-account --key-file=credentials.json
   gcloud projects list
   
   # Test Firestore access
   gcloud firestore databases list
   ```

3. **GCS access issues**:
   ```bash
   # Test GCS access
   gsutil ls gs://your-bucket-name
   
   # Check permissions
   gsutil iam get gs://your-bucket-name
   ```

4. **Memory issues**:
   ```bash
   # Monitor memory usage
   docker stats rag-service
   kubectl top pods -n rag-service
   
   # Increase memory limits
   # In docker-compose.yml or k8s deployment
   ```

5. **Processing failures**:
   ```bash
   # Check RAG-Anything logs
   docker logs rag-service | grep -i error
   
   # Verify dependencies
   docker exec rag-service libreoffice --version
   docker exec rag-service tesseract --version
   ```

### Performance Tuning

1. **Increase resources**:
   ```yaml
   resources:
     requests:
       memory: "4Gi"
       cpu: "2"
     limits:
       memory: "8Gi"
       cpu: "4"
   ```

2. **Optimize storage**:
   ```bash
   # Use SSD storage for better I/O
   # Configure appropriate storage class in Kubernetes
   ```

3. **Enable GPU support**:
   ```yaml
   # For MinerU with GPU
   env:
   - name: CUDA_VISIBLE_DEVICES
     value: "0"
   - name: DEVICE
     value: "cuda:0"
   ```

### Scaling

1. **Horizontal scaling**:
   ```bash
   # Kubernetes
   kubectl scale deployment rag-anything-service --replicas=3
   
   # Docker Compose
   docker-compose up -d --scale rag-anything-service=3
   ```

2. **Load balancing**:
   ```yaml
   # Use a load balancer or ingress controller
   apiVersion: networking.k8s.io/v1
   kind: Ingress
   metadata:
     name: rag-service-ingress
   spec:
     rules:
     - host: rag-api.yourdomain.com
       http:
         paths:
         - path: /
           pathType: Prefix
           backend:
             service:
               name: rag-anything-service
               port:
                 number: 80
   ```

## Security Considerations

1. **Service Account Permissions**:
   - Use least privilege principle
   - Rotate keys regularly
   - Monitor access logs

2. **Network Security**:
   - Use HTTPS in production
   - Configure proper firewall rules
   - Use VPC for internal communication

3. **Data Protection**:
   - Encrypt data at rest
   - Use secure communication
   - Implement proper backup strategies

4. **API Security**:
   - Implement rate limiting
   - Use authentication/authorization
   - Monitor for suspicious activity
