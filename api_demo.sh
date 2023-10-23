set -x

CUDA_VISIBLE_DEVICES=2,3 python api_demo.py \
    --model_name qwen-14b-chat