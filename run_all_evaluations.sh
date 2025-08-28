#!/bin/bash
# Quest Evaluation Suite - Run All Model Evaluations
# This script provides a menu to run different model evaluations

set -e

echo "🧬 QUEST Evaluation Suite"
echo "========================="
echo ""
echo "Available Evaluations:"
echo "1. ProtBERT Fine-tuned (on smaller dataset)"
echo "2. BERT with Character-wise Tokenization"
echo "3. ProtBERT Pretrained (out of the box)"
echo "4. Run All Evaluations Sequentially"
echo "5. Show Configuration Requirements"
echo "6. Exit"
echo ""

read -p "Select an option (1-6): " choice

case $choice in
    1)
        echo "🚀 Running ProtBERT Fine-tuned Evaluation..."
        ./run_ray_protbert_finetuned_eval.sh
        ;;
    2)
        echo "🚀 Running BERT Character-wise Evaluation..."
        ./run_ray_bert_charwise_eval.sh
        ;;
    3)
        echo "🚀 Running ProtBERT Pretrained Evaluation..."
        ./run_ray_protbert_pretrained_eval.sh
        ;;
    4)
        echo "🚀 Running All Evaluations Sequentially..."
        echo "This will run all three evaluations one after another."
        echo "Each evaluation will use the same Ray cluster configuration."
        echo ""
        read -p "Are you sure you want to continue? (y/N): " confirm
        if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
            echo "Starting evaluation 1/3: ProtBERT Fine-tuned..."
            ./run_ray_protbert_finetuned_eval.sh
            echo ""
            echo "Starting evaluation 2/3: BERT Character-wise..."
            ./run_ray_bert_charwise_eval.sh
            echo ""
            echo "Starting evaluation 3/3: ProtBERT Pretrained..."
            ./run_ray_protbert_pretrained_eval.sh
            echo ""
            echo "✅ All evaluations completed!"
        else
            echo "❌ Cancelled."
        fi
        ;;
    5)
        echo "📋 Configuration Requirements"
        echo "============================="
        echo ""
        echo "Before running evaluations, ensure you have:"
        echo ""
        echo "1. AWS Configuration:"
        echo "   - AWS SSO login: aws sso login --profile DataUserSageMakerAccess-953915750371"
        echo "   - S3 access to: s3://fh-pi-warren-h-eco/"
        echo ""
        echo "2. Ray Cluster Configuration:"
        echo "   - Update config/ray/ray_protbert_eval.yaml with your:"
        echo "     * KeyName (your AWS key pair)"
        echo "     * SecurityGroupIds"
        echo "     * SubnetId"
        echo "     * ImageId (if different)"
        echo ""
        echo "3. Data Requirements:"
        echo "   - ProtBERT tokenized data: s3://fh-pi-warren-h-eco/quest/hf_final/protbert_tok"
        echo "   - BERT char-wise data: s3://fh-pi-warren-h-eco/quest/hf_final/bert_char_tok"
        echo "   - Fine-tuned model: s3://fh-pi-warren-h-eco/quest/models/protbert_finetuned"
        echo ""
        echo "4. Environment:"
        echo "   - setup_ray_env.sh script available and working"
        echo "   - Python environment with required packages (see env/environment.yaml)"
        echo ""
        echo "5. Custom Model Support:"
        echo "   - Fixed ray_evaluator.py to support both HuggingFace and custom models"
        echo "   - Custom models should be .pt or .pth files with model_config saved"
        echo ""
        ;;
    6)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid option. Please choose 1-6."
        exit 1
        ;;
esac

echo ""
echo "🎯 Evaluation Tasks Completed!"
echo ""
echo "📝 Summary of what was accomplished:"
echo "1. ✅ Fixed custom model loading in ray_evaluator.py"
echo "2. ✅ Created ProtBERT fine-tuned evaluation script"
echo "3. ✅ Created BERT character-wise evaluation script"
echo "4. ✅ Created ProtBERT pretrained evaluation script"
echo "5. ✅ Created Ray cluster configuration template"
echo ""
echo "🔗 Useful Commands:"
echo "   - Monitor jobs: ray job list --address http://localhost:8268"
echo "   - View logs: ray job logs <job_id> --address http://localhost:8268"
echo "   - Stop cluster: ray down config/ray/ray_protbert_eval.yaml"
echo ""
echo "📊 Check your W&B projects for results:"
echo "   - quest-protbert-finetuned-eval"
echo "   - quest-bert-charwise-eval"
echo "   - quest-protbert-pretrained-eval"