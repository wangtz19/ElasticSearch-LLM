import gradio as gr
from src.chat.chat_model import ChatModel
    

if __name__ == '__main__':
    chat_model = ChatModel(llm_params={
        "model_name": "chatglm2-6b",
    })
    

    ui = gr.Interface(
        fn = chat_model.stream_chat,
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
                label="Source Documents",
            ),
        ],
        title="基于ES检索+text2vec向量匹配召回+chatglm生成->知识问答系统",
        description="输入问题--->es检索--->召回相关--->向量化--->计算相似度--->召回相关--->作为prompt输入给chatglm--->生成回答",
    )
    ui.queue().launch(share=True)