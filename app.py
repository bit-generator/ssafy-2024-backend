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
chat_upstage = ChatUpstage()
embedding_upstage = UpstageEmbeddings(model="embedding-query")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "eco"

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
    search_kwargs={"k": 3}  # 쿼리와 관련된 chunk를 3개 검색하기 (default : 4)
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
    clean_msg = req.message.strip()
    qa = RetrievalQA.from_chain_type(llm=chat_upstage,
                                     chain_type="stuff",
                                     retriever=pinecone_retriever,
                                     return_source_documents=True)

    result = qa(clean_msg)

    llm = ChatUpstage()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "너는 금융용어 및 금융상품 설명 챗봇이 올바르게 답변하는지를 필터링하는 봇이야." \
             "다음과 같은 요구사항에 따라 출력해야 돼." \
                 "금융상품 관련 내용: 가장 적합한 내용을 찾아 답변을 제시하고, 수치가 있을 경우 수치를 함께 제시. 수치는 가장 중요한 값을 우선적으로 제시하고, 다른 수치가 있다면 함께 제시. 만약 해당 금융상품에 대한 내용이 문서에 없다면 모른다고 답변해" \
                    "음식이나 날씨, 스포츠, 게임 등 금융과 관련되지 않은 내용: 금융 관련 챗봇이기 때문에 모르겠다거나 설명해드릴 수 없다고 답할 것."),

            # User Query
            ("human", result['result']),
        ]
    )

    chain = prompt | llm | StrOutputParser() # with output parser

    #invoke the chain
    c_result = chain.invoke({})
    return {"reply": c_result}


# @app.post("/assistant")
# async def assistant_endpoint(req: AssistantRequest):
#     assistant = await openai.beta.assistants.retrieve("asst_tc4AhtsAjNJnRtpJmy1gjJOE")

#     if req.thread_id:
#         # We have an existing thread, append user message
#         await openai.beta.threads.messages.create(
#             thread_id=req.thread_id, role="user", content=req.message
#         )
#         thread_id = req.thread_id
#     else:
#         # Create a new thread with user message
#         thread = await openai.beta.threads.create(
#             messages=[{"role": "user", "content": req.message}]
#         )
#         thread_id = thread.id

#     # Run and wait until complete
#     await openai.beta.threads.runs.create_and_poll(
#         thread_id=thread_id, assistant_id=assistant.id
#     )

#     # Now retrieve messages for this thread
#     # messages.list returns an async iterator, so let's gather them into a list
#     all_messages = [
#         m async for m in openai.beta.threads.messages.list(thread_id=thread_id)
#     ]
#     print(all_messages)

#     # The assistant's reply should be the last message with role=assistant
#     assistant_reply = all_messages[0].content[0].text.value

#     return {"reply": assistant_reply, "thread_id": thread_id}


@app.get("/health")
@app.get("/")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
