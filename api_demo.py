import argparse
import os
from typing import List, Optional
import pydantic
import uvicorn
from fastapi import Body, FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing_extensions import Annotated
from starlette.responses import RedirectResponse

from src.chat.chat_model import ChatModel
from src.chat.chat_model_clf import ChatModelClassifier


class BaseResponse(BaseModel):
    code: int = pydantic.Field(200, description="HTTP status code")
    msg: str = pydantic.Field("success", description="HTTP status message")

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
            }
        }


class ListDocsResponse(BaseResponse):
    data: List[str] = pydantic.Field(..., description="List of document names")

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
                "data": ["doc1.docx", "doc2.pdf", "doc3.txt"],
            }
        }


class ChatMessage(BaseModel):
    question: str = pydantic.Field(..., description="Question text")
    response: str = pydantic.Field(..., description="Response text")
    history: List[List[str]] = pydantic.Field(..., description="History text")
    source_documents: List[str] = pydantic.Field(
        ..., description="List of source documents and their scores"
    )
    first_intent: Optional[str] = pydantic.Field(
        None, description="First intent label"
    )
    second_intent: Optional[str] = pydantic.Field(
        None, description="Second intent label"
    )
    title: Optional[str] = pydantic.Field(None, description="Title of the document")
    content: Optional[str] = pydantic.Field(None, description="Content of the document")
    prompt: Optional[str] = pydantic.Field(None, description="Prompt text")

    class Config:
        schema_extra = {
            "example": {
                "question": "工伤保险如何办理？",
                "response": "根据已知信息，可以总结如下：\n\n1. 参保单位为员工缴纳工伤保险费，以保障员工在发生工伤时能够获得相应的待遇。\n2. 不同地区的工伤保险缴费规定可能有所不同，需要向当地社保部门咨询以了解具体的缴费标准和规定。\n3. 工伤从业人员及其近亲属需要申请工伤认定，确认享受的待遇资格，并按时缴纳工伤保险费。\n4. 工伤保险待遇包括工伤医疗、康复、辅助器具配置费用、伤残待遇、工亡待遇、一次性工亡补助金等。\n5. 工伤保险待遇领取资格认证包括长期待遇领取人员认证和一次性待遇领取人员认证。\n6. 工伤保险基金支付的待遇项目包括工伤医疗待遇、康复待遇、辅助器具配置费用、一次性工亡补助金、丧葬补助金等。",
                "history": [
                    [
                        "工伤保险是什么？",
                        "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                    ]
                ],
                "source_documents": [
                    "出处 [1] 广州市单位从业的特定人员参加工伤保险办事指引.docx：\n\n\t( 一)  从业单位  (组织)  按“自愿参保”原则，  为未建 立劳动关系的特定从业人员单项参加工伤保险 、缴纳工伤保 险费。",
                    "出处 [2] ...",
                    "出处 [3] ...",
                ],
            }
        }


async def local_doc_chat(
    question: str = Body(..., description="Question", example="工伤保险是什么？"),
    history: List[List[str]] = Body(
        [],
        description="History of previous questions and answers",
        example=[
            [
                "工伤保险是什么？",
                "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
            ]
        ],
    ),
):
    for resp, history, sources in chat_model.chat(
        query=question, streaming=False, chat_history=history
    ):
        pass
    source_documents = [
        f"""出处 [{inum + 1}] {doc['source']}\n\t{doc['content']}\n"""
        f"""相关度：{doc['score']}\n\n"""
        for inum, doc in enumerate(sources)
    ]

    return ChatMessage(
        question=question,
        response=resp,
        history=history,
        source_documents=source_documents,
        title=sources[0]["source"],
        content=sources[0]["content"],
        prompt=sources[0]["prompt"],
        second_intent=sources[0]["second_intent"],
    )


async def document():
    return RedirectResponse(url="/docs")


def api_start(args):
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.get("/", response_model=BaseResponse)(document)
    app.post("/local_doc_qa/local_doc_chat", response_model=ChatMessage)(local_doc_chat)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", "-ho", type=str, default="0.0.0.0")
    parser.add_argument("--port", "-p", type=int, default=7861)
    parser.add_argument("--filepath", "-f", type=str, default="data/cleaned_data",
                        help="path to the local knowledge file")
    parser.add_argument("--model_name", "-m", type=str, default="chatglm2-6b",
                        help="model name, e.g. chatglm2-6b, baichuan2-13b-chat, qwen-14b-chat")
    parser.add_argument("--es_top_k", "-ek", type=int, default=3,
                        help="top k for es search")
    parser.add_argument("--use_intent", "-ui", action="store_true",
                        help="whether to use intent classifier")
    parser.add_argument("--bert_path", "-bp", type=str, default="hfl/chinese-roberta-wwm-ext",
                        help="bert model name or path, e.g. hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--clf_type", "-ct", type=str, default="direct",
                        help="intent classifier type, e.g. direct, two_level, none")
    parser.add_argument("--use_vs", "-uv", action="store_true",
                        help="whether to use vector store")
    args = parser.parse_args()

    args_dict = vars(args)
    chat_model = ChatModel(llm_params=args_dict) if not args.use_intent else \
                ChatModelClassifier(llm_params=args_dict, bert_path=args.bert_path,
                                    clf_type=args.clf_type, use_vs=args.use_vs)
    
    api_start(args)
