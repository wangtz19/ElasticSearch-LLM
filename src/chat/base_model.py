from src.es.my_elasticsearch import MyElasticsearch
from src.models.loader import LoaderCheckPoint
from src.models.base import BaseAnswer
import src.models.shared as shared


class BaseModel:
    def __init__(
            self, 
            es_url="http://127.0.0.1:9200",
            es_top_k=3, 
            es_lower_bound=25,
            histrory_len=3,
            llm_params=None
        ):
        self.es = MyElasticsearch(es_url)
        self.es_top_k = es_top_k
        self.es_lower_bound = es_lower_bound # for auxiliary history query retrieval score 

        self.history_len = histrory_len
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
            prompt, 
            chat_history=[], 
            streaming: bool = False
        ):
        
        for answer_result in self.llm.generatorAnswer(prompt=prompt, history=chat_history, streaming=streaming):  
            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            history[-1][0] = query
            yield resp, history