import os
import chromadb
from dataclasses import dataclass
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

@dataclass(frozen=True)
class Constants:
    TAX_WITH_MARKDOWN_PATH = "./tax_with_markdown.docx" # 제일 적합
    CHROMA_DB_PATH = "./chroma"
    CHROMA_COLLECTION_NAME = "chroma-tax"
    MODEL_CHATGPT = "gpt-4o-mini"
    MODEL_EMBEDDING = "text-embedding-3-large"


def get_retriever():
    embedding = OpenAIEmbeddings(model=Constants.MODEL_EMBEDDING)

    exist_db = False
    if os.path.isdir(Constants.CHROMA_DB_PATH):
        exist_db = True

    if exist_db:
        database = Chroma(
            collection_name=Constants.CHROMA_COLLECTION_NAME,
            persist_directory=Constants.CHROMA_DB_PATH,
            embedding_function=embedding
        )
    else:
        chromadb.api.client.SharedSystemClient.clear_system_cache()

        # 문서를 쪼갠다.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # 내부 문서 객체 하나가 가질 수 있는 토큰 수
            chunk_overlap=200  # 위/아래 문맥을 겹치게 하여 문서 객체 사이의 겹치는 토큰의 정도 (유사도를 위해)
        )
        loader = Docx2txtLoader(Constants.TAX_WITH_MARKDOWN_PATH)
        document = loader.load_and_split(text_splitter=text_splitter)

        database = Chroma.from_documents(
            documents=document,
            embedding=embedding,
            collection_name=Constants.CHROMA_COLLECTION_NAME,  # RDB로 치면 테이블 이름
            persist_directory=Constants.CHROMA_DB_PATH  # 데이터 저장 위치
        )

    retriever = database.as_retriever(search_kwargs={'k': 4})
    return retriever


def get_llm(model='gpt-4o'):
    return ChatOpenAI(model=model)


def get_dictionary_chain():
    llm = get_llm(model=Constants.MODEL_CHATGPT)

    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    prompt = ChatPromptTemplate.from_template(f"""
            사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
            만약 변경할 필요가 없다고 판단되면, 사용자의 질문을 변경하지 않아도 됩니다.
            그런 경우에는 질문만 리턴해주세요
            사전: {dictionary}

            질문: {{query}}
        """)

    dictionary_chain = prompt | llm | StrOutputParser() # LCEL
    return dictionary_chain


def get_qa_chain():
    llm = get_llm(model=Constants.MODEL_CHATGPT)

    rag_template = PromptTemplate(
        template="""[Identity]
        - 당신은 최고의 한국 소득세 전문가 입니다.
        - [Context]를 참고해서 사용자의 질문에 답변해주세요.

        [Context]
        {retrieved_docs}

        Question: {query}
        """,
        input_variables=['retrieved_docs', 'query'])
    rag_chain = rag_template | llm # LCEL
    return rag_chain


def get_ai_message(user_message):
    retriever = get_retriever()
    retrieved_docs = retriever.invoke(user_message)

    dictionary_chain = get_dictionary_chain()
    qa_chain = get_qa_chain()
    tax_chain = {"query": dictionary_chain, "retrieved_docs": lambda x: retrieved_docs} | qa_chain

    ai_response = tax_chain.invoke({'retrieved_docs': retrieved_docs, "query": user_message})
    return ai_response.content
