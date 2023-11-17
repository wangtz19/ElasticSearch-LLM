import os
from text_splitter import split_by_heading
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from text2vec import SentenceModel
# import sys
# sys.path.append("")
# embedding_model_path="/root/share/text2vec-large-chinese"
# embedding_model_path="/root/share/gte-large"
# embedding_model_path="/root/share/m3e-base"
embedding_model_path="/root/share/stella-large-zh-v2"
t2v_model = SentenceModel(embedding_model_path)
# t2v_model = SentenceTransformer("/root/share/text2vec-large-chinese")


def unicode2str(filename):
    u = filename[:-len(".txt")]
    code_points = u.split("#")[1:]
    s = ""
    for cp in code_points:
        s += chr(int(cp[1:5], 16))
        s += cp[5:] # for trailing numbers not starting with #U
    return s



def load_data(filenames, split=True):
    doc_list = []
    for filename in tqdm(filenames):
        if ".pdf" not in str(filename):
            with open(filename, "r") as f:
                if len(filename.split("#")) > 2:
                    title = unicode2str(os.path.basename(filename))
                else:
                    title = os.path.basename(filename)[:-len(".txt")]
                if not split:
                    data = f.read()
                    doc_list.append({
                        "标题": title,
                        "子标题": "",
                        "内容": data
                    })
                else:
                    data = split_by_heading(filename)
                 
                   
                    for key, val in data.items():
                        sentence_embeddings=t2v_model.encode(title+"。"+key+"。"+val)
                        sentence_embeddings=np.resize(sentence_embeddings,(512,))
                        doc_list.append({
                            "标题": title,
                            "子标题": key,
                            "内容": val,
                            "vector":sentence_embeddings
                        })
    return doc_list

def load_data_from_dir(dir, split=True):
    filenames = [os.path.join(dir, filename) for filename in os.listdir(dir)]
    return load_data(filenames, split=split)

