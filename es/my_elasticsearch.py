from elasticsearch import Elasticsearch

class MyElasticsearch(Elasticsearch):
    def __init__(self, url, index_name="linewell-policy", fields=None) -> None:
        self.index_name = index_name
        self.client = Elasticsearch(url)
        self.fields = ["标题", "子标题", "内容"] if fields is None else fields
    def search(self, query, top_k=0) -> list:
        query_body = {
            "query": {
                "multi_match": {
                    "analyzer": "ik_smart",  # 使用ik分词器进行分词处理
                    "query": query,
                    "fields": self.fields
                }
            }    
        }
        response = self.client.search(index=self.index_name, body=query_body)
        res = []
        for hit in response["hits"]["hits"]:
            score = hit["_score"]
            doc_data = "\n".join([hit["_source"][field] for field in self.fields])
            res.append((score, doc_data))
        if top_k == 0:
            return res
        else:
            top_k = min(top_k, len(res))
            return res[:top_k]
