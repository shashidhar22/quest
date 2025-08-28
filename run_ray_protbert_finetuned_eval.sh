#!/bin/bash
# Evaluate ProtBERT fine-tuned on smaller dataset
# This script evaluates a ProtBERT model that was fine-tuned on a smaller dataset

set -e  # Exit on any error

echo "🚀 Starting ProtBERT fine-tuned evaluation..."

# 1. Setup environment
echo "📝 Setting up environment..."
source setup_ray_env.sh

# 2. Get current AWS credentials 
echo "🔑 Getting AWS credentials..."
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "❌ AWS credentials not available. Please run: aws sso login --profile DataUserSageMakerAccess-953915750371"
    exit 1
fi

# Export credentials from SSO session
AWS_CREDS=$(aws configure export-credentials --profile DataUserSageMakerAccess-953915750371 --format env 2>/dev/null || echo "")
if [ -n "$AWS_CREDS" ]; then
    eval "$AWS_CREDS"
    echo "✅ AWS credentials exported successfully"
else
    echo "⚠️  Using existing environment credentials"
fi

# Test S3 access
echo "🧪 Testing S3 access..."
if aws s3 ls s3://fh-pi-warren-h-eco/ --region us-west-2 > /dev/null 2>&1; then
    echo "✅ S3 access confirmed"
else
    echo "❌ Cannot access S3 bucket. Please check permissions."
    exit 1
fi

# 3. Check if Ray cluster is running
echo "🔍 Checking Ray cluster status..."
if ray status config/ray/ray_protbert_eval.yaml > /dev/null 2>&1; then
    echo "✅ Ray cluster is running"
else
    echo "🚀 Starting Ray cluster (12 g4dn.xlarge instances)..."
    ray up config/ray/ray_protbert_eval.yaml --yes
fi

# 4. Wait for cluster to be ready and setup credentials
echo "⚙️  Setting up credentials on cluster..."

# Setup AWS credentials
ray exec config/ray/ray_protbert_eval.yaml "aws configure set aws_access_key_id '$AWS_ACCESS_KEY_ID' --profile default"
ray exec config/ray/ray_protbert_eval.yaml "aws configure set aws_secret_access_key '$AWS_SECRET_ACCESS_KEY' --profile default"
ray exec config/ray/ray_protbert_eval.yaml "aws configure set aws_session_token '$AWS_SESSION_TOKEN' --profile default"
ray exec config/ray/ray_protbert_eval.yaml "aws configure set region us-west-2 --profile default"

# Test S3 access on cluster
echo "Testing S3 access on cluster..."
ray exec config/ray/ray_protbert_eval.yaml "aws s3 ls s3://fh-pi-warren-h-eco/ --region us-west-2 && echo 'S3 access confirmed on cluster' || echo 'S3 access failed on cluster'"

# Copy the data from S3
echo "Copying ProtBERT tokenized data from S3..."
ray exec config/ray/ray_protbert_eval.yaml "aws s3 cp --recursive s3://fh-pi-warren-h-eco/quest/hf_final/protbert_tok /home/ubuntu/quest/data/protbert_tok --quiet && echo '✅ Data copy successful' || echo '❌ Data copy failed'"

# Copy fine-tuned model if it exists
echo "Checking for fine-tuned ProtBERT model..."
ray exec config/ray/ray_protbert_eval.yaml "
if aws s3 ls s3://fh-pi-warren-h-eco/quest/models/protbert_finetuned/ --region us-west-2 > /dev/null 2>&1; then
  echo 'Found fine-tuned ProtBERT model, downloading...'
  mkdir -p /home/ubuntu/quest/data/models/
  aws s3 sync s3://fh-pi-warren-h-eco/quest/models/protbert_finetuned/ /home/ubuntu/quest/data/models/protbert_finetuned/ --region us-west-2 --quiet
  echo '✅ Fine-tuned model downloaded'
else
  echo '❌ No fine-tuned ProtBERT model found. Please ensure the model exists at s3://fh-pi-warren-h-eco/quest/models/protbert_finetuned/'
  exit 1
fi
"

# 5. Start dashboard
echo "📊 Starting Ray dashboard..."
ray dashboard config/ray/ray_protbert_eval.yaml --port 8268 > /dev/null 2>&1 &
DASHBOARD_PID=$!
sleep 5  # Give dashboard time to start

# 6. Submit the evaluation job
echo "🎯 Submitting ProtBERT fine-tuned evaluation job..."
ray job submit \
  --address='http://localhost:8268' \
  --working-dir . \
  --runtime-env-json='{"env_vars": {"AWS_ACCESS_KEY_ID": "'$AWS_ACCESS_KEY_ID'", "AWS_SECRET_ACCESS_KEY": "'$AWS_SECRET_ACCESS_KEY'", "AWS_SESSION_TOKEN": "'$AWS_SESSION_TOKEN'", "AWS_DEFAULT_REGION": "us-west-2"}}' \
  -- python3 scripts/ray_evaluator.py \
      --raw_dataset_dir /home/ubuntu/quest/data/protbert_tok \
      --model_name /home/ubuntu/quest/data/models/protbert_finetuned \
      --batch_size 64 \
      --max_length 256 \
      --mlm_prob 0.15 \
      --wandb_project quest-protbert-finetuned-eval \
      --checkpoint_dir /home/ubuntu/quest/checkpoints \
      --resume_from_checkpoint

# 7. Show job status and instructions
echo "🧹 ProtBERT fine-tuned evaluation job submitted successfully!"
echo "📊 Dashboard running on port 8268 (PID: $DASHBOARD_PID)"
echo ""
echo "🎉 You can now close your local machine! The job will continue running on AWS."
echo ""
echo "💡 To monitor the job later:"
echo "   - Check job status: ray job list --address http://localhost:8268"
echo "   - View logs: ray job logs <job_id> --address http://localhost:8268"
echo "   - Stop dashboard: kill $DASHBOARD_PID"
echo "   - Stop cluster: ray down config/ray/ray_protbert_eval.yaml"
echo ""
echo "✅ ProtBERT fine-tuned evaluation is running independently on AWS!"