import os

import pandas as pd

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['OPENAI_API_BASE'] = 'https://api.chatanywhere.tech/v1'
os.environ['OPENAI_API_KEY'] = 'sk-uy02H0tfVl3HDLpqx2kblLGxeBKUX7QPGydDjdtxwvKfbytS'
os.environ["ZHIPUAI_API_KEY"] = "360d12eaef31ec85f985879b429802f1.krN7dcfOrM6lYRmq"

import re
import io
import streamlit as st
from PIL import Image
import time
import glob
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# LangSmith
from langsmith.wrappers import wrap_openai
from langsmith import traceable

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]="XMU双创竞赛Chatbot"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"


project_name = "YOUR PROJECT NAME"  # Update with your project name


# 文档库
current_dir_path = os.path.dirname(os.path.realpath(__file__))
def get_docx_filenames(directory):
    os.chdir(directory)
    return glob.glob('*.docx')
path = current_dir_path
file = get_docx_filenames(path)
filename = [path + f for f in file]

data = []
for i in range(len(filename)):
    loader = Docx2txtLoader(filename[i])
    data.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    keep_separator=False,
)
texts = text_splitter.split_documents(data)

# 为每个chunk附加额外的文件名元信息
for i in range(len(texts)):
    texts_name = os.path.basename(texts[i].metadata['source'])
    texts_name, _ = os.path.splitext(texts_name)
    texts[i].page_content = texts_name + "\n\n" + texts[i].page_content
    texts[i].metadata['source'] = texts_name

# 构建矢量库
embedding = HuggingFaceEmbeddings(
    model_name="moka-ai/m3e-base",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True},
)

# vectorstore = Chroma.from_documents(
#     documents=texts,
#     embedding=embedding,
# )
vectorstore = FAISS.from_documents(
    documents=texts,
    embedding=embedding,
)

retriever = vectorstore.as_retriever(
    # search_type="similarity",
    search_type="mmr", # 最大边际相关性搜索
    search_kwargs={
        "k": 5,
    },
)

def generate_history_chain():
    # 构造一个聊天模型包装器,key和url从函数输入中获取
    llm = ChatZhipuAI(
        model="glm-4",
        temperature=0.0,
    )

    # contextualize_system_prompt = """给定聊天历史记录和可能参考聊天历史记录中上下文的最新用户问题，制定一个独立的问题，该问题可以在没有聊天历史记录的情况下理解。
    # 不要回答这个问题，如果需要，只需重新制定，否则按原样返回。"""
    #
    # contextualize_prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", contextualize_system_prompt),
    #         MessagesPlaceholder("chat_history"),
    #         ("human", "{input}"),
    #     ]
    # )

    # history_aware_retriever = create_history_aware_retriever(
    #     llm, retriever, contextualize_prompt
    # )

    # 构造一个模板template和一个prompt
    template = """你是回答问题的助理。这些是你应该认为是同一个概念的词：竞赛=中国国际大学生创新大赛；主赛道=高教主赛道；红旅=青年红色筑梦之旅；产业赛道=产业命题赛道。
    尽量只利用以下检索到的上下文来尽可能详细回答问题。如果你不知道答案，就说你不知道。

    Context: {context}

    Question: {input}

    Answer:"""
    prompt = PromptTemplate.from_template(template)

    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", template),
    #         MessagesPlaceholder("chat_history"),
    #         ("human", "{input}"),
    #     ]
    # )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # qa_chain = create_stuff_documents_chain(llm, prompt)
    # rag_history_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    # 构造一个输出解析器和链
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}  # 包含用户的原始问题
        | prompt
        | llm
        | StrOutputParser()  # 接受字符串或BaseMessage并返回python字符串，最后从invoke方法返回
    )
    # return rag_history_chain
    return rag_chain

# 生成回复
def generate_response(query, rag_history):
# rag_history: {'input': "", 'chat_history': [], 'context': "", 'answer': ""}
# rag_history: [{'role': 'user', 'content': '高教主赛道的比赛安排是什么?'}]
    rag_chain = generate_history_chain()
    # response = rag_chain.invoke({"input": query, "chat_history": rag_history})
    response = rag_chain.invoke(query)
    #################################
    print("print response\n", response)
    # rag_history.append([HumanMessage(content=query), response["answer"]])
    stream_r = rag_chain.stream(query)
    return response, stream_r
    # return response["answer"]#, history


# streamlit 工作流
# 设置page title
st.title('XMU双创竞赛Chatbot')
# st.title('高教主赛道的比赛安排是什么?')

rag_history = [] # 输入glm的history

if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.history_down = []

###############################
# for message in st.session_state.history:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
# 显示历史对话
for message in st.session_state.history:
    if message["role"] == "user":
        with st.chat_message(name="user", avatar="user"):
            st.markdown(message["content"])
    elif message["role"] == "assistant" and message["content"] != None:
        with st.chat_message(name="assistant", avatar="assistant"):
            st.markdown(message["content"])

# 清除历史按钮
buttonClean = st.sidebar.button("清理会话历史", key="clean")
if buttonClean:
    st.session_state.history = []
    rag_history = []
    st.rerun()

# 接收用户输入
if query_text := st.chat_input("请输入您的问题"):
    with st.chat_message("user"):
        st.markdown(query_text)
    # 添加用户消息到记录中
    st.session_state.history.append({"role": "user", "content": query_text})
    st.session_state.history_down.append({"role": "user", "content": query_text})

    response, stream_r = generate_response(query_text, rag_history)

    #
    # # 获取glm回复和检索文档
    # # response, rag_history = generate_response(query_text, rag_history)
    # response = generate_response(query_text, rag_history)
    # print("print response\n", response)
    # rag_history.extend(HumanMessage(content=query_text))
    # print("print rag_history\n", rag_history)
    # print("print HumanMessage(content=query_text)\n", HumanMessage(content=query_text))
    # rag_history.extend(response)
    # ###################################################
    # print("print rag_history\n", str(rag_history))

    with st.chat_message("assistant"):
        # st.markdown(response)
        # 流式输出response
        st.write_stream(stream_r)

    # 添加AI消息到记录中
    st.session_state.history.append({"role": "assistant", "content": response})
    st.session_state.history_down.append({"role": "assistant", "content": response})

    # 只保留十轮对话
    if len(st.session_state.history) > 20:
        st.session_state.messages = st.session_state.messages[-20:]
        rag_history = rag_history[-20:]

    df = pd.DataFrame(st.session_state.history_down) # , columns=["role", "content"]
    df.to_excel('history.xlsx', index=True)
    csv = df.to_csv().encode("utf-8")
    st.download_button(
        label="DOWNLOAD!",
        data=csv,
        file_name="聊天记录.txt",
        mime="text/plain"
    )
