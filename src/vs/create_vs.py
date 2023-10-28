from langchain.vectorstores.faiss import FAISS
from langchain.schema.document import Document
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from pypinyin import lazy_pinyin
from utils import torch_gc, KB_ROOT_PATH, EMBEDDING_DEVICE
import os
import datetime
import argparse

# only consider FAQ currently
def load_file(filepath):
    if filepath.lower().endswith(".json"):
        return load_json(filepath)
    else:
        raise NotImplementedError
    
def load_json(filepath):
    import json
    with open(filepath, "r") as f:
        data = json.load(f)
    docs = [
        Document(
            page_content=f"问题: {item['instruction']}\n答案: {item['output']}",
            metadata={
                "source": "宝安区政府网站常见问题"
            }
        )
        for item in data
    ]
    return docs

def create_vector_store(filepath, embedding):
    docs = load_file(filepath)
    vector_store = FAISS.from_documents(docs, embedding)
    torch_gc()
    filename = os.path.basename(filepath)
    vs_path = os.path.join(KB_ROOT_PATH, 
                        f"""{"".join(lazy_pinyin(filename))}_FAISS_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}""",
                        "vector_store")
    vector_store.save_local(vs_path)
    return vector_store


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default="../data/faq.json")
    parser.add_argument("--embed_model_name", type=str, default="/root/chinese-bert-wwm")
    args = parser.parse_args()

    embedding = HuggingFaceBgeEmbeddings(
                    model_name=args.embed_model_name,
                    model_kwargs={"device": EMBEDDING_DEVICE}
                )
    create_vector_store(args.filepath, embedding)