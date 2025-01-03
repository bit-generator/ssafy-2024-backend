import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import ChatUpstage
from langchain_upstage import UpstageEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser #문자열(text, string)만 나오게 하는 출력 파서
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel

load_dotenv()

# upstage models
chat_upstage = ChatUpstage(model="solar-pro")
embedding_upstage = UpstageEmbeddings(model="embedding-query")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "fine"

# create new index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=4096,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

pinecone_vectorstore = PineconeVectorStore(index=pc.Index(index_name), embedding=embedding_upstage)

pinecone_retriever = pinecone_vectorstore.as_retriever(
    search_type='mmr',  # default : similarity(유사도) / mmr 알고리즘
    search_kwargs={"k": 5, "include_metadata": True}  # 쿼리와 관련된 chunk를 3개 검색하기 (default : 4)
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class AssistantRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


class ChatRequest(BaseModel):
    messages: List[ChatMessage]  # Entire conversation for naive mode


class MessageRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat_endpoint(req: MessageRequest):
    chat_prompt = ("너는 금융 상품 전문 챗봇으로, 주어진 문서를 정확하게 이해해서 답변을 해야해."
            "문서에 있는 내용으로만 답변하고 내용이 없다면, 잘 모르겠다고 답변해."
            "수치가 있을 경우 명확한 수치를 제시하고, 수치가 여러 개일 경우 가장 중요한 수치를 우선적으로 제시하되, 주요한 수치를 모두 제시해."
            "그리고 금융용어는 전문 용어를 풀어서 쉽게 설명해줘.중학생도 이해할 수 있을 정도로."
            "금융 용어의 경우, 금융과 관련되지 않은 질문을 하면 모른다고 답변하고, 금융과 관련된 용어로만 답변해. 다른 동의어는 무시해."
            "만약 금융과 관련된 용어라면 문서에 대한 정보를 최우선으로 고려하여 답변하고, 문서에 내용이 부족하다면 일반적인 정의로 답변해.")
    clean_msg = req.message.strip()     # 사용자 질문
    full_prompt = chat_prompt + clean_msg
    qa = RetrievalQA.from_chain_type(llm=chat_upstage,
                                     chain_type="stuff",
                                     retriever=pinecone_retriever,
                                     return_source_documents=True)

    result = qa(full_prompt)      # 챗봇 답변

    llm = ChatUpstage()
    prompt = ChatPromptTemplate.from_messages(
        [
            # User Query
            ("human", result['result']),
        ]
    )

    chain = prompt | llm | StrOutputParser() # with output parser

    #invoke the chain
    c_result = chain.invoke({})
    return {"reply": c_result}

@app.get("/health")
@app.get("/")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
