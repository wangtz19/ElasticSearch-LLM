import sys
from typing import Any
from src.models.loader import LoaderCheckPoint

loaderCheckPoint: LoaderCheckPoint = None
# 此处请写绝对路径
llm_model_dict = {
    "chatglm2-6b": {
        "name": "chatglm2-6b",
        "pretrained_model_name": "THUDM/chatglm2-6b",
        "local_model_path": "/root/share/chatglm2-6b",
        "provides": "ChatGLM2"
    },
    "chatglm2-6b-32k": {
        "name": "chatglm2-6b-32k",
        "pretrained_model_name": "THUDM/chatglm2-6b-32k",
        "local_model_path": "/root/share/chatglm2-6b-32k",
        "provides": "ChatGLM2"
    },
    "baichuan2-13b-chat": {
        "name": "baichuan2-13b-chat",
        "pretrained_model_name": "baichuan-inc/Baichuan2-13B-Chat",
        "local_model_path": "/root/share/Baichuan2-13B-Chat",
        "provides": "Baichuan2"
    },
    "qwen-14b-chat": {
        "name": "qwen-14b-chat",
        "pretrained_model_name": "Qwen/Qwen-14B-Chat",
        "local_model_path": "/root/share/Qwen-14B-Chat",
        "provides": "Qwen"
    },
    "chatglm3-6b": {
        "name": "chatglm3-6b",
        "pretrained_model_name": "THUDM/chatglm3-6b",
        "local_model_path": "/root/share/chatglm3-6b",
        "provides": "ChatGLM3"
    },
    "chatglm3-6b-32k": {
        "name": "chatglm3-6b-32k",
        "pretrained_model_name": "THUDM/chatglm3-6b-32k",
        "local_model_path": "/root/share/chatglm3-6b-32k",
        "provides": "ChatGLM3"
    },
}


def loaderLLM() -> Any:
    pre_model_name = loaderCheckPoint.model_name
    llm_model_info = llm_model_dict[pre_model_name]

    loaderCheckPoint.model_name = llm_model_info['pretrained_model_name']

    loaderCheckPoint.model_path = llm_model_info["local_model_path"]

    loaderCheckPoint.reload_model()

    provides_class = getattr(sys.modules['src.models'], llm_model_info['provides'])
    modelInsLLM = provides_class(checkPoint=loaderCheckPoint)
    return modelInsLLM
