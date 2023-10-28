set -x

CUDA_VISIBLE_DEVICES=2,3 python api_demo.py \
    --model_name baichuan2-13b-chat \
    --es_top_k 5