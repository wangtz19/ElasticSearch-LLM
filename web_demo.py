from text2vec import SentenceModel, semantic_search
import gradio as gr
from es import MyElasticsearch
import models.shared as shared
from models.loader import LoaderCheckPoint
from models.base import BaseAnswer

model = SentenceModel()
es = MyElasticsearch("http://127.0.0.1:9200")

PROMPT_TEMPLATE = """已知信息：
{context} 

根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：{question}"""

def get_answer(query, context, chat_history=[], streaming: bool = False):
    prompt = PROMPT_TEMPLATE.replace("{question}", query).replace("{context}", context)
    for answer_result in llm.generatorAnswer(prompt=prompt, history=chat_history, streaming=streaming):
        
        resp = answer_result.llm_output["answer"]
        history = answer_result.history
        history[-1][0] = query
        response = {"query": query,
                    "result": resp}
        
        print("----",response,"----")
        print("----",history,"----")
        yield resp, history

def chat(
    query,
    es_top_k=3,
    vec_top_k=1,
):
    instructions = es.search(query, es_top_k)     #[(score, doc)]
    ans = [ins[1] for ins in instructions]

    query_embedding = model.encode(query)
    answer_embeddings = model.encode(ans)
    hits = semantic_search(query_embedding, answer_embeddings, top_k=vec_top_k)
    print(hits)
    print(ans)
    res = ""
    hits = hits[0]
    for hit in hits:
        res += ans[hit['corpus_id']] + ","

    for resp, history in get_answer(query=query, context=res, chat_history=[], streaming=False):
        yield resp, res
    

if __name__ == '__main__':
    shared.loaderCheckPoint = LoaderCheckPoint()
    llm_model_ins = shared.loaderLLM()
    llm_model_ins.set_history_len(3)
    llm: BaseAnswer = llm_model_ins
    try:
        generator = llm.generatorAnswer("你好")
        for answer_result in generator:
            print(answer_result.llm_output)
        reply = """模型已成功加载"""
    except Exception as e:
        reply = """模型未成功加载"""

    ui = gr.Interface(
        fn = chat,
        inputs=[
            gr.inputs.Textbox(
                lines=2, label="Input", placeholder="请输入问题"
            ),
            gr.components.Slider(
                minimum=1, maximum=100, step=1, value=3, label="ES Top k"),
            gr.components.Slider(minimum=1, maximum=10, step=1, value=1, label="VEC Top k"),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=6,
                label="Generator Output",
            ),
            gr.inputs.Textbox(
                lines=10,
                label="Vec Output",
            ),
            # gr.inputs.Textbox(
            #     lines=25,
            #     label="ES Output",
            # ),

        ],
        title="基于ES检索+text2vec向量匹配召回+chatglm生成->知识问答系统",
        description="输入问题--->es检索--->召回相关--->向量化--->计算相似度--->召回相关--->作为prompt输入给chatglm--->生成回答",
    )
    ui.queue().launch(share=True)