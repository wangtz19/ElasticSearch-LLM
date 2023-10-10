from elasticsearch import Elasticsearch
from text_loader import load_data_from_dir
from tqdm import tqdm

def create_index(index_name, es):
    settings = {
        "analysis": {
            "analyzer": {
                "default": {
                    "type": "ik_max_word",
                    "tokenizer": "ik_max_word",
                },
                "default_search": {
                    "type": "ik_smart",
                    "tokenizer": "ik_smart",
                }
            }
        }
    }
    mappings = {
        "properties": {
            "vector": {
                "type": "dense_vector",
                "dims": 10,
                "index": True,
                "similarity": "l2_norm"
            }
        }
    }
    res = es.indices.create(index=index_name, settings=settings, mappings=mappings)
    print(res)

if __name__ == "__main__":
    # create es client
    es = Elasticsearch("http://127.0.0.1:9200")

    # test connection
    if es.ping():
        print("Elasticsearch connected successfully.")
    else:
        print("Elasticsearch connection failed.")

    index_name ="linewell-policy"
    # delete index if exists
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print("Index deleted successfully.")

    # create index
    create_index(index_name, es)

    # insert data
    doc_list = load_data_from_dir("/root/es-text2vec-llm-qa/data/cleaned_data_all")
    for doc in tqdm(doc_list):
        es.index(index=index_name, body=doc)
    print("Data inserted successfully.")
