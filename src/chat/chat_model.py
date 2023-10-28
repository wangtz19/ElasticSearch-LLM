from src.chat.base_model import BaseModel
from text2vec import SentenceModel, semantic_search
import time

PROMPT_TEMPLATE = """已知信息：
{context} 

根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：{question}"""

PROMPT_TEMPLATE_WITH_HISTORY = """已知信息：
{context} 
历史问题：
{history}
根据上述已知信息和历史问题，详细和专业的来回答用户的本次问题。如果无法从已知信息中得到答案，请说 “无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 本次问题如下：{question}"""

class ChatModel(BaseModel):
    def __init__(
            self, 
            es_url="http://127.0.0.1:9200",
            es_top_k=3, 
            es_lower_bound=25,
            histrory_len=3,
            llm_params=None,
            vec_top_k=2,
            use_vec=False,
            vec_model_path="/root/share/text2vec-large-chinese",
        ):
        self.vec_top_k = vec_top_k
        self.use_vec = use_vec
        if self.use_vec:
            self.vec_model = SentenceModel(vec_model_path)
        
        super().__init__(
            es_url=es_url,
            es_top_k=es_top_k,
            histrory_len=histrory_len,
            es_lower_bound=es_lower_bound,
            llm_params=llm_params
        )
    
    def chat(
            self, 
            query,
            streaming=False,
            chat_history=[]
        ):
        start = time.time()
        docs = self.es.search(query, self.es_top_k)
        history_query = ""
        chat_history_query = [h[0] for h in chat_history if h[0] is not None]
        if docs[0].metadata["score"] < self.es_lower_bound:
            for h in chat_history_query[::-1]: # in reverse order
                new_docs = self.es.search(h + " " + query, self.es_top_k)
                history_query = h
                if new_docs[0].metadata["score"] > self.es_lower_bound:
                    break
        print(f"es search time: {time.time() - start} s")
        
        all_texts = [doc.page_content for doc in docs]
        if self.use_vec:
            start = time.time()
            query_embedding = self.vec_model.encode(query)
            answer_embeddings = self.vec_model.encode(all_texts)
            hits = semantic_search(query_embedding, answer_embeddings, top_k=self.vec_top_k)
            print(f"vec search time: {time.time() - start} s")
            hits = hits[0]
            context = "\n".join([all_texts[hit['corpus_id']] for hit in hits])
            source_documents = [{
                "source": docs[hit['corpus_id']].metadata["source"],
                "content": docs[hit['corpus_id']].page_content,
                "score": hit['score']
            } for hit in hits]
        else:
            context = "\n".join(all_texts)
            source_documents = [{
                "source": doc.metadata["source"],
                "content": doc.page_content,
                "score": doc.metadata["score"]
            } for doc in docs]
        if len(history_query) > 0:
            prompt = PROMPT_TEMPLATE_WITH_HISTORY.format(
                context=context,
                history=history_query,
                question=query
            )
        else:
            prompt = PROMPT_TEMPLATE.format(
                context=context,
                question=query
            )
        print(f"prompt: {prompt}")
        for resp, history in self.get_answer(query=query, prompt=prompt, chat_history=chat_history, streaming=streaming):
            yield resp, history, source_documents

    def stream_chat(
            self,
            query,
            chat_history,
            es_top_k,
            vec_top_k,
    ):
        self.es_top_k = es_top_k
        self.vec_top_k = vec_top_k
        for resp, history, source_documents in self.chat(
            query, streaming=True, chat_history=chat_history):
            source = "\n\n" + "".join(
                [f"""<details> <summary>【相关度】{doc["score"]} -【出处{i + 1}】 {doc["source"]}</summary>\n"""
                f"""{doc["content"]}\n"""
                f"""</details>"""
                for i, doc in
                enumerate(source_documents)])
            history[-1][-1] += source
            yield history, ""