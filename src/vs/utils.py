import torch
import os

KB_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base")

EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def torch_gc():
    if torch.cuda.is_available():
        # with torch.cuda.device(DEVICE):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        try:
            from torch.mps import empty_cache
            empty_cache()
        except Exception as e:
            print(e)
            print("如果您使用的是 macOS 建议将 pytorch 版本升级至 2.0.0 或更高版本，以支持及时清理 torch 产生的内存占用。")


def get_vs_list():
    if not os.path.exists(KB_ROOT_PATH):
        return []
    lst = os.listdir(KB_ROOT_PATH)
    if not lst:
        return []
    lst.sort()
    return lst

def get_vs_path(local_doc_id: str):
    return os.path.join(KB_ROOT_PATH, local_doc_id, "vector_store")

def get_existing_vs_path(local_doc_id=None):
    if local_doc_id is not None:
        vs_path = get_vs_path(local_doc_id)
        if os.path.exists(vs_path):
            print(f"Exsiting knowledge base loaded from exsiting {vs_path}")
            return vs_path
        else:
            print(f"Error: {vs_path} is not a valid vector store path")
    vs_list = get_vs_list()
    if len(vs_list) > 0:
        print(f"Exsiting knowledge base loaded from {os.path.join(KB_ROOT_PATH, vs_list[-1], 'vector_store')}")
        return os.path.join(KB_ROOT_PATH, vs_list[-1], "vector_store")
    else:
        print("Error: no exsiting vector store found")
        return None