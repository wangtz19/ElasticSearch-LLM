from src.es.my_elasticsearch import MyElasticsearch
from src.models.loader import LoaderCheckPoint
from src.models.base import BaseAnswer
import src.models.shared as shared

from text2vec import SentenceModel, semantic_search
import time

PROMPT_TEMPLATE = """已知信息：
{context} 

根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：{question}"""

class ChatModel:
    def __init__(
            self, 
            es_top_k=5, 
            vec_top_k=3,
            use_vec=False,
            histrory_len=3,
            vec_model_path="/root/share/text2vec-large-chinese",
            es_url="http://127.0.0.1:9200",
            llm_params=None
        ):
        self.vec_model = SentenceModel(vec_model_path)
        self.es = MyElasticsearch(es_url)
        self.es_top_k = es_top_k
        self.vec_top_k = vec_top_k
        self.use_vec = use_vec
    
        shared.loaderCheckPoint = LoaderCheckPoint(params=llm_params)
        llm_model_ins = shared.loaderLLM()
        llm_model_ins.set_history_len(histrory_len)
        self.llm: BaseAnswer = llm_model_ins

        try:
            generator = self.llm.generatorAnswer("你好")
            for answer_result in generator:
                print(answer_result.llm_output)
        except Exception as e:
            print(e)
            raise Exception("LLM model load failed")
    
    def get_answer(
            self, 
            query, 
            context, 
            chat_history=[], 
            streaming: bool = False
        ):
        prompt = PROMPT_TEMPLATE.format(question=query, context=context)
        for answer_result in self.llm.generatorAnswer(prompt=prompt, history=chat_history, streaming=streaming):  
            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            history[-1][0] = query
            yield resp, history
    
    def chat(
            self, 
            query,
            streaming=False,
            history=[]
        ):
        start = time.time()
        instructions = self.es.search(query, self.es_top_k)     #[(score, text, source)]
        print(f"es search time: {time.time() - start} s")
        all_texts = [ins[1] for ins in instructions]
        
        if self.use_vec:
            start = time.time()
            query_embedding = self.vec_model.encode(query)
            answer_embeddings = self.vec_model.encode(all_texts)
            hits = semantic_search(query_embedding, answer_embeddings, top_k=self.vec_top_k)
            print(f"vec search time: {time.time() - start} s")
            hits = hits[0]
            context = "\n".join([all_texts[hit['corpus_id']] for hit in hits])
            source_documents = [{
                "source": instructions[hit['corpus_id']][2],
                "score": hit['score']
            } for hit in hits]
        else:
            context = "\n".join(all_texts)
            source_documents = [{
                "source": ins[2],
                "score": ins[0]
            } for ins in instructions]
        print(f"source_documents: {source_documents}")
        for resp, history in self.get_answer(query=query, context=context, chat_history=history, streaming=streaming):
            yield resp, history, source_documents

    def stream_chat(
            self,
            query,
            es_top_k,
            vec_top_k,
    ):
        self.es_top_k = es_top_k
        self.vec_top_k = vec_top_k
        for resp, history, source_documents in self.chat(query, streaming=True):
            yield resp, source_documents
