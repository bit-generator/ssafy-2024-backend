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
index_name = "eco4"

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
    search_kwargs={"k": 10}  # 쿼리와 관련된 chunk를 3개 검색하기 (default : 4)
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
    clean_msg = req.message.strip()     # 사용자 질문
    qa = RetrievalQA.from_chain_type(llm=chat_upstage,
                                     chain_type="stuff",
                                     retriever=pinecone_retriever,
                                     return_source_documents=True)

    result = qa(clean_msg)      # 챗봇 답변
    resultMsg = "Q: " + clean_msg + "\n" + "A: " + result['result']

    llm = ChatUpstage()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "너는 금융용어 및 금융상품 설명 챗봇이 올바르게 답변하는지를 필터링하는 봇이야." \
             "내용은 Q: (질문 내용)\\nA: (답변 내용) 과 같은 형식으로 들어오는데, 이것을 다음과 같은 요구사항에 따라 출력해야 돼." \
                 "(질문 내용)이 금융상품 관련 내용이면 무조건 (답변 내용)을 출력해야 돼." \
                    "(질문 내용)이 음식, 날씨, 스포츠, 게임, 컴퓨터, 언어, 음악, 드라마, 애니메이션, 영화, 사회 등 금융과 관련되지 않은 내용이라면 금융 관련 챗봇이기 때문에 모르겠다거나 설명해드릴 수 없다고 출력해야 돼." \
                        "(답변 내용)에 (질문 내용)에서 묻지 않은 회사의 내용이 포함되어 있다면, (답변 내용)에서 그 부분은 제외하고 다듬어서 출력해야 돼." \
                        "(질문 내용)에서 \"국민은행\"에 대해 묻는다면 \"KB국민은행\"으로 취급해 줘."
                        ),

            # few-shot prompting
            ("human", "Q: 감가상각이 뭐지?\\nA: 감가상각은 ~ 입니다."),  # human request
            ("ai", "감가상각은 ~ 입니다."),      # LLM response
            ("human", "Q: 오늘 점심 추천 좀 해줘.\\nA: 오늘 점심으로는 ~을 추천드립니다."),
            ("ai",  "음식은 제 전문이 아니므로 답변해드릴 수 없습니다."),
            ("human", "Q: C++에서 클래스가 뭐지?\\nA: C++에서 클래스는 ~을 의미합니다."),
            ("ai",  "프로그래밍은 제 전문이 아니므로 답변해드릴 수 없습니다."),
            ("human", "Q: 국민은행의 금융상품에는 뭐가 있는지 궁금해.\\nA: 국민은행의 금융상품에는 'KB국민첫재테크적금', '청년 처음적금', '신한 땡겨요 적금', 'KB스타적금Ⅱ' 등이 있습니다."),
            ("ai",  "국민은행의 금융상품에는 'KB국민첫재테크적금', '청년 처음적금', 'KB스타적금Ⅱ' 등이 있습니다."),
            ("human", "Q: 오늘 금융상담 중 \"만기\"라는 용어가 나왔는데 무슨 뜻인지 모르겠다. 알려줄 수 있어?\\nA: 금융에서 만기란 ~을 말하며, ~"),
            ("ai",  "금융에서 만기란 ~을 말하며, ~"),
            ("human", "Q: 리그 오브 레전드에 대해 설명해줘.\\nA: 리그 오브 레전드는 라이엇 게임즈에서 개발한 ~"),
            ("ai",  "게임은 제 전문이 아니므로 답변해드릴 수 없습니다."),
            ("human", "Q: 안녕하세요는 일본어로 뭐라고 해?\\nA: \"안녕하세요\"는 일본어로 \"곤니치와\"라고 합니다."),
            ("ai",  "언어는 제 전문이 아니므로 답변해드릴 수 없습니다."),
            ("human", "Q: 국민은행의 ~ 상품의 금리는 얼마인지 궁금해.\\nA: ~ 상품의 금리는 다음과 같습니다. ~"),
            ("ai",  "~ 상품의 금리는 다음과 같습니다. ~"),
            ("human", "Q: 신한은행에는 어떤 금융상품이 있는지 설명해 줘.\\nA: 신한은행의 금융상품에는 'KB국민첫재테크적금', '청년 처음적금', '신한 땡겨요 적금', '신한 쏠만해 적금' 등이 있습니다."),
            ("ai",  "신한은행의 금융상품에는 '신한 땡겨요 적금', '신한 쏠만해 적금' 등이 있습니다."),
            ("human", "Q: 우리은행의 금융상품에는 어떤 게 있는지 알려줘.\\nA: 우리은행의 금융상품에는 '신한 땡겨요 적금'과 '외화보통예금'이 있습니다."),
            ("ai",  "우리은행의 금융상품에는 '외화보통예금'이 있습니다."),

            # User Query
            ("human", resultMsg),
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
