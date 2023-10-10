import gradio as gr
import argparse
from src.chat.chat_model import ChatModel

block_css = """.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}
.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}"""

webui_title = """
# 🎉ElasticSearch LLM WebUI🎉
👍 [https://github.com/wangtz19/ElasticSearch-LLM](https://github.com/wangtz19/ElasticSearch-LLM)
"""

default_theme_args = dict(
    font=["Source Sans Pro", 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
)

def init_model(args_dict):
    global chat_model
    chat_model = ChatModel(llm_params={
        "model_name": args_dict["model_name"],
    })

def get_ui():
    with gr.Blocks(css=block_css, theme=gr.themes.Default(**default_theme_args)) as demo:
        gr.Markdown(webui_title)
        with gr.Tab("对话"):
            with gr.Row():
                with gr.Column(scale=10):
                    chatbot = gr.Chatbot([[None, "你好，我是ES LLM，欢迎提问政务政策相关的问题。"]],
                                        elem_id="chat-box",
                                        show_label=False)
                    query = gr.Textbox(show_label=False,
                                    placeholder="请输入提问内容，按回车进行提交").style(container=False)
                with gr.Column(scale=5):
                    # mode = gr.Radio(["LLM 对话", "知识库问答"],
                    #                 label="请选择使用模式",
                    #                 value="知识库问答", )
                    es_top_k = gr.Slider(minimum=1, maximum=100, step=1, value=3, label="ES Top k")
                    vec_top_k = gr.Slider(minimum=1, maximum=10, step=1, value=1, label="VEC Top k")
                    clear_btn = gr.Button("清空对话")
        query.submit(
            chat_model.stream_chat,
            inputs=[query, chatbot, es_top_k, vec_top_k],
            outputs=[chatbot, query],
            show_progress=True,
        )
        clear_btn.click(lambda: ([]), outputs=[chatbot], show_progress=True)
    return demo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default="chatglm2-6b")
    args = parser.parse_args()
    
    init_model(vars(args))
    demo = get_ui()
    (demo.queue(concurrency_count=3)
        .launch(server_name='0.0.0.0',
            server_port=7860,
            show_api=False,
            share=True,
            inbrowser=False))