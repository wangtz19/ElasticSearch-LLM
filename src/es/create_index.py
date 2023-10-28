from elasticsearch import Elasticsearch
from text_loader import load_data_from_dir
from tqdm import tqdm
from argparse import ArgumentParser
import json

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


def check_and_create_index(index_name, es):
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print("Index deleted successfully.")
    create_index(index_name, es)
    print("Index created successfully.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--index_name", type=str, default="linewell-policy")
    parser.add_argument("--es_url", type=str, default="http://127.0.0.1:9200")
    args = parser.parse_args()

    # create es client
    es = Elasticsearch(args.es_url)

    # test connection
    if es.ping():
        print("Elasticsearch connected successfully.")
    else:
        print("Elasticsearch connection failed.")

    index_name = args.index_name
    if index_name == "linewell-policy":
        check_and_create_index(index_name, es)
        doc_list = load_data_from_dir("/root/es-llm/data/cleaned_data_all")
        for doc in tqdm(doc_list):
            es.index(index=index_name, document=doc)
    else:
        for intent in ["basic_info", "award", "process", "materials", "condition"]:
            check_and_create_index(intent, es)
            doc_list = json.load(open(f"/root/es-llm/data/intent/{intent}.json", "r"))
            for doc in tqdm(doc_list):
                es.index(index=intent, document=doc)
            print(f"Index {intent} inserted successfully.")

    print("Data inserted successfully.")
