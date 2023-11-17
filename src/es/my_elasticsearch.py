from elasticsearch import Elasticsearch
import sys
sys.path.append('/root') # 即下图中标蓝的文件的路径
from text_similarity.simcse.predict_sup import main
from text2vec import SentenceModel,Similarity
import numpy as np

# embedding_model_path="/root/share/text2vec-large-chinese"
# embedding_model_path="/root/share/gte-large"
# embedding_model_path="/root/share/m3e-base"
embedding_model_path="/root/share/stella-large-zh-v2"
t2v_model = SentenceModel(embedding_model_path)

def EuclideanDistance_np(x, y):
    # np.linalg.norm 用于范数计算，默认是二范数，相当于平方和开根号
    return np.linalg.norm(np.array(x) - np.array(y))

class MyElasticsearch(Elasticsearch):
    def __init__(self, url, index_name="linewell-policy", fields=None) -> None:
        self.index_name = index_name
        self.client = Elasticsearch(url)
        self.fields = ["标题", "子标题", "内容"] if fields is None else fields
    

    # def search(self, query, top_k=0) -> list:

    #     query_body = {
    #         "query": {
    #             "multi_match": {
    #                 "analyzer": "ik_smart",
    #                 "query": query,
    #                 "fields": self.fields
    #             }
    #         }    
    #     }

    #     response = self.client.search(index=self.index_name, body=query_body)
    #     res = []
    #     for hit in response["hits"]["hits"]:
    #         score = hit["_score"]
    #         texts = "\n".join([hit["_source"][field] for field in self.fields])
    #         sources = {
    #             "title": hit['_source']['标题'],
    #             "content": hit['_source']['子标题'] + hit['_source']['内容'],
    #         }
    #         res.append((score, texts, sources))
    #     if top_k == 0:
    #         return res
    #     else:
    #         top_k = min(top_k, len(res))
    #         return res[:top_k]

    #         # top_k = min(top_k, len(res))
    #         # top_k_sentences=res[:top_k+1]
    #         # # print("top_k_sentences",top_k_sentences)
    #         # top_k_list=[]
    #         # for top_k_sentence in top_k_sentences:
    #         #     # print("top_k_sentence",top_k_sentence)
    #         #     top_k_dict={}
    #         #     sim_score=main(query.replace("什么","").replace("哪些",""),top_k_sentence[1].replace("\n","").replace("\t","").strip())
    #         #     # print("sim_score",sim_score)
    #         #     top_k_dict.update({"res":top_k_sentence,"sim_score":sim_score})
    #         #     # print("top_k_dict",top_k_dict)
    #         #     top_k_list.append(top_k_dict)

    #         # top_k_list = sorted(top_k_list, key=lambda x:x["sim_score"],reverse=True)    
    #         # print("top_k_list",top_k_list)
            
    #         # top_k_list=top_k_list[:top_k]

    #         # top_k_l=[]
    #         # for top_k_d in top_k_list:
    #         #     top_k_l.append({"res":top_k_d["res"],"sim_score":top_k_d["res"][0]})
    #         # print("top_k_l",top_k_l)

    #         # top_k_l = sorted(top_k_l, key=lambda x:x["sim_score"],reverse=True)    
    #         # print("(top_k_l[:top_k]",top_k_l[:top_k])

    #         # top_k_l_=[]
    #         # for top_k_dd in top_k_l:
    #         #     print("top_k_dd",top_k_dd)
    #         #     top_k_l_.append(top_k_dd["res"])

    #         # return top_k_l_[:top_k]


    def search(self, query, top_k=0) -> list:

        sentence_embeddings=t2v_model.encode(query)
        sentence_embeddings=np.resize(sentence_embeddings,(512,))

        print("sentence_embeddings.shape",sentence_embeddings.shape)

        query_body_zifu = {
                    "query": {
                        "multi_match": {
                            "analyzer": "ik_smart",
                            "query": query,
                            "fields": self.fields
                        }
                    }    
                }
        response_zifu = self.client.search(index=self.index_name, body=query_body_zifu)
        res_zifu = []
        for hit in response_zifu["hits"]["hits"]:
            score = hit["_score"]
            texts = "\n".join([hit["_source"][field] for field in self.fields])
            sources = {
                "title": hit['_source']['标题'],
                "content": hit['_source']['子标题'] + hit['_source']['内容'],
            }
            res_zifu.append((score, texts, sources))

        # # 精准KNN计算
        # query_body={
        #     "query": {
        #         "script_score": {
        #         "query": {
        #             "match_all": {}
        #         },
        #         "script": {
        #             "source": "cosineSimilarity(params.queryVector, 'vector')+1.0",
        #         "params": {"queryVector": sentence_embeddings
        #         }
        #         }
        #     }
        #     }
        # }

        # 近似KNN
        query_body_vec= {
                    "query": {
                        "multi_match": {
                        "analyzer": "ik_smart",
                        "query": query,
                        "fields": self.fields,
                        "boost": 0
                    },
                    },
                    "knn": {
                        "field": "vector",
                        "query_vector": sentence_embeddings,
                        "k": 5,
                        "num_candidates": 100,
                        "boost": 1
                    },
                    "size": 10
                    }

        response_vec = self.client.search(index=self.index_name, body=query_body_vec)
        res_vec = []
        for hit in response_vec["hits"]["hits"]:
            # print("hit",hit)
            score = hit["_score"]
            texts = "\n".join([hit["_source"][field] for field in self.fields])
            sources = {
                "title": hit['_source']['标题'],
                "content": hit['_source']['子标题'] + hit['_source']['内容'],
            }
            res_vec.append((score, texts, sources))

        print(top_k/2)
        print(type(top_k/2))
        res=res_zifu[:int(top_k/2)]+res_vec[:int(top_k/2)]

        if top_k == 0:
            return res
        else:
            top_k = min(top_k, len(res))
            return res[:top_k]

            # top_k = min(top_k, len(res))
            # top_k_sentences=res[:top_k+1]
            # # print("top_k_sentences",top_k_sentences)
            # top_k_list=[]
            # for top_k_sentence in top_k_sentences:
            #     # print("top_k_sentence",top_k_sentence)
            #     top_k_dict={}
            #     sim_score=main(query.replace("什么","").replace("哪些",""),top_k_sentence[1].replace("\n","").replace("\t","").strip())
            #     # print("sim_score",sim_score)
            #     top_k_dict.update({"res":top_k_sentence,"sim_score":sim_score})
            #     # print("top_k_dict",top_k_dict)
            #     top_k_list.append(top_k_dict)

            # top_k_list = sorted(top_k_list, key=lambda x:x["sim_score"],reverse=True)    
            # print("top_k_list",top_k_list)
            
            # top_k_list=top_k_list[:top_k]

            # top_k_l=[]
            # for top_k_d in top_k_list:
            #     top_k_l.append({"res":top_k_d["res"],"sim_score":top_k_d["res"][0]})
            # print("top_k_l",top_k_l)

            # top_k_l = sorted(top_k_l, key=lambda x:x["sim_score"],reverse=True)    
            # print("(top_k_l[:top_k]",top_k_l[:top_k])

            # top_k_l_=[]
            # for top_k_dd in top_k_l:
            #     print("top_k_dd",top_k_dd)
            #     top_k_l_.append(top_k_dd["res"])

            # return top_k_l_[:top_k]