set -x

CUDA_VISIBLE_DEVICES=0 python api_demo.py \
    --model_name chatglm3-6b \
    --use_intent \
    --clf_type direct \
    --bert_path /root/es-llm/src/classification/checkpoints/policy-23.10.27-10:37/checkpoint-5240