#!/bin/bash

# Text Analysis Microservices Deployment Script
# This script sets up the infrastructure for separated API and Worker services

set -e

# Configuration
PROJECT_ID="bni-prod-dma-bnimove-ai"
REGION="asia-southeast2"
SERVICE_ACCOUNT_EMAIL="369455734154-compute@developer.gserviceaccount.com"

# Pub/Sub Configuration
PUBSUB_TOPIC="character-screening-request"
PUBSUB_SUBSCRIPTION="character-screening-worker-sub"

# Service Names
API_SERVICE_NAME="text-analysis-api-service"
WORKER_SERVICE_NAME="text-analysis-worker-service"

echo "🚀 Setting up Text Analysis Microservices..."
echo "Project: $PROJECT_ID"
echo "Region: $REGION"

# Function to check if command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "❌ Error: $1 is not installed or not in PATH"
        exit 1
    fi
}

# Check prerequisites
echo "🔍 Checking prerequisites..."
check_command gcloud
check_command docker

# Set project
echo "🔧 Setting up Google Cloud project..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔧 Enabling required Google Cloud APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable pubsub.googleapis.com
gcloud services enable firestore.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# Create Pub/Sub topic if it doesn't exist
echo "📬 Setting up Pub/Sub infrastructure..."
if ! gcloud pubsub topics describe $PUBSUB_TOPIC &> /dev/null; then
    echo "Creating Pub/Sub topic: $PUBSUB_TOPIC"
    gcloud pubsub topics create $PUBSUB_TOPIC
else
    echo "Pub/Sub topic already exists: $PUBSUB_TOPIC"
fi

# Create Pub/Sub subscription if it doesn't exist
if ! gcloud pubsub subscriptions describe $PUBSUB_SUBSCRIPTION &> /dev/null; then
    echo "Creating Pub/Sub subscription: $PUBSUB_SUBSCRIPTION"
    gcloud pubsub subscriptions create $PUBSUB_SUBSCRIPTION \
        --topic=$PUBSUB_TOPIC \
        --ack-deadline=600 \
        --message-retention-duration=7d \
        --max-delivery-attempts=5
else
    echo "Pub/Sub subscription already exists: $PUBSUB_SUBSCRIPTION"
fi

# Set up IAM permissions for the service account
echo "🔐 Setting up IAM permissions..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/pubsub.subscriber" || true

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/pubsub.publisher" || true

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/datastore.user" || true

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/storage.objectViewer" || true

# Create Firestore database if it doesn't exist (this might require manual setup)
echo "💾 Checking Firestore database..."
echo "Note: If Firestore is not set up, please create it manually in the Google Cloud Console"

# Build and deploy services using Cloud Build
echo "🏗️ Building and deploying services..."

# Check if cloudbuild.yaml exists
if [ -f "cloudbuild.yaml" ]; then
    echo "Starting Cloud Build..."
    gcloud builds submit --config cloudbuild.yaml
else
    echo "❌ cloudbuild.yaml not found. Please ensure all files are in the correct directory."
    exit 1
fi

echo "✅ Deployment completed!"
echo ""
echo "📋 Service Information:"
echo "API Service: https://$API_SERVICE_NAME-$REGION.run.app"
echo "Worker Service: $WORKER_SERVICE_NAME (internal)"
echo "Pub/Sub Topic: $PUBSUB_TOPIC"
echo "Pub/Sub Subscription: $PUBSUB_SUBSCRIPTION"
echo ""
echo "🔍 To check service status:"
echo "gcloud run services list --region=$REGION"
echo ""
echo "📊 To view logs:"
echo "gcloud logs tail --follow --resource-type=cloud_run_revision --filter='resource.labels.service_name=\"$API_SERVICE_NAME\"'"
echo "gcloud logs tail --follow --resource-type=cloud_run_revision --filter='resource.labels.service_name=\"$WORKER_SERVICE_NAME\"'"
