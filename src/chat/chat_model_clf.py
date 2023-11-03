from src.chat.base_model import BaseModel
from src.classification import BertClassifier
from src.vs import get_existing_vs_path, EMBEDDING_DEVICE
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from src.chat.template import PROMPT_TEMPLATE_TOP1, intent_map
from src.classification.src.constants import ID2LABEL
import random

class ChatModelClassifier(BaseModel):

    def __init__(
            self,
            es_url="http://127.0.0.1:9200",
            es_top_k=1, 
            es_lower_bound=25,
            histrory_len=3,
            llm_params=None,
            clf_type="direct",
            bert_path=None,
            bert_path_fisrt=None,
            bert_path_second=None,
            vs_path=None,
            embed_model_name="/root/share/chinese-bert-wwm",
            use_vs=False,
    ):
        if clf_type == "direct":
            self.clf = BertClassifier(
                id2label=ID2LABEL["second_level"]["policy"], # for fast deploy, change later
                model_checkpoint=bert_path
            )
        elif clf_type == "two_level":
            self.clf_first = BertClassifier(bert_path_fisrt)
            self.clf_second = BertClassifier(bert_path_second)
        self.clf_type = clf_type

        if use_vs:
            embedding = HuggingFaceBgeEmbeddings(
                            model_name=embed_model_name,
                            model_kwargs={"device": EMBEDDING_DEVICE})
            vs_path = get_existing_vs_path() if vs_path is None else vs_path
            assert vs_path is not None, "Error: no exsiting vector store found"
            self.vs = FAISS.load_local(vs_path, embedding)
        self.use_vs = use_vs
        
        super().__init__(
            es_url=es_url,
            es_top_k=es_top_k,
            histrory_len=histrory_len,
            es_lower_bound=es_lower_bound,
            llm_params=llm_params
        )
    
    def get_index_name(self, query):
        if self.clf_type == "direct":
            index_name = self.clf.predict(query)["label"]
        elif self.clf_type == "two_level":
            clf_result_first = self.clf_first.predict(query)["label"]
            if clf_result_first == "knowledge_base":
                clf_result = "knowledge_base"
            else: # only support `policy` for second classifier currently
                clf_result = self.clf_second.predict(query)["label"]
            index_name = clf_result if clf_result in intent_map.keys() else \
                        random.choice(intent_map.keys())
        else:
            index_name = "project" # no intent classifier
        return index_name
    
    def get_es_search_docs(self, query, index_name, chat_history_query=[]):
        docs = self.es.search(query, self.es_top_k, index_name=index_name)
        # TODO: add multi-turn search optimization
        # if docs[0].metadata["score"] < self.es_lower_bound:
        #     for h in chat_history_query[::-1]: # in reverse order
        #         new_docs = self.es.search(h + " " + query, self.es_top_k, index_name=index_name)
        #         # history_query = h
        #         if new_docs[0].metadata["score"] > self.es_lower_bound:
        #             docs = new_docs
        #             break
        return docs
    
    def get_vs_search_docs(self, query, chat_history_query=[]):
        docs_with_score = self.vs.similarity_search_with_score(query, k=self.es_top_k)
        # TODO: add multi-turn search optimization
        docs = []
        for doc, score in docs_with_score:
            doc.metadata["score"] = score
            docs.append(doc)
        return docs

    def chat(
            self,
            query,
            streaming=False,
            chat_history=[]
    ):
        chat_history_query = [h[0] for h in chat_history if h[0] is not None]
        index_name = self.get_index_name(query)
        docs = self.get_es_search_docs(query, index_name, chat_history_query)
        if self.use_vs:
            docs_vs = self.get_vs_search_docs(query, chat_history_query)
            docs = docs_vs + docs
            # TODO: add rerank
            
        if self.es_top_k == 1:
            prompt = PROMPT_TEMPLATE_TOP1.format(
                title=docs[0].metadata["source"],
                label=intent_map[index_name],
                content=docs[0].page_content,
                question=query
            )
        else:
            # TODO
            pass
        source_documents = [{
            "source": doc.metadata["source"],
            "content": doc.page_content,
            "score": doc.metadata["score"],
            "second_intent": intent_map[index_name],
            "prompt": prompt,
        } for doc in docs]

        for resp, history in self.get_answer(query=query, prompt=prompt, chat_history=chat_history, streaming=streaming):
            yield resp, history, source_documents
