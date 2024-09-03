import nltk
import torch
import os
import gradio as gr
import numpy as np  

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import ChatGLM
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain import LLMChain
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 文档加载函数
def load_documents(txt_path):
    loader = TextLoader(txt_path)
    document = loader.load()
    return document

# 文档分割函数
def splitter_documents(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 32, chunk_overlap = 0)
    documents = text_splitter.split_documents(document)
    return documents

# 文档嵌入函数
def embedding_documents():
    model_name = "shibing624/text2vec-base-chinese"
    model_kwargs = {"device": "cuda:0"}
    encode_kwargs = {'normalize_embeddings': True}
    embedding = HuggingFaceBgeEmbeddings(
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs,
                )
    return embedding

# 向量库创建函数
def house_build(txt_path,documents,embedding):

    # 提取向量库的目录名称
    vector_db_name = txt_path.split('/')[-1]

    # 检查向量库是否已经存在
    if not os.path.exists(vector_db_name):
        # 如果向量库不存在，则创建它
        db = Chroma.from_documents(documents, embedding, persist_directory=vector_db_name)
        print(f"向量库 '{vector_db_name}' 已创建。")
    else:
        print(f"向量库 '{vector_db_name}' 已存在，跳过创建步骤。")
        
    house = Chroma(persist_directory=vector_db_name,embedding_function=embedding)
    return house

# 提示体生成函数
def create_prompt():
    template = "你是一个助手，首先你需要向我问好，然后需要根据参考资料回答我的问题。\n参考资料：{message}\n问题：{input}"
    prompt = PromptTemplate(
        input_variables=["message","input"],
        template=template
    )
    return prompt



######################### 生成函数 #####################################

def generate(txt_path,input,k):

    # 准备工作：先查看是否已经存在向量库：
    # 提取向量库的目录名称
    vector_db_name = txt_path.split('/')[-1]

    if os.path.exists(vector_db_name):

        print(f"向量库 '{vector_db_name}' 已存在，跳过创建步骤。")

        # 生成嵌入模型
        embedding = embedding_documents()
        # 加载向量库
        house = Chroma(persist_directory=vector_db_name,embedding_function=embedding)
         # 搜索相似文本
        message = house.similarity_search(input,k)
        # 生成提示体prompt
        prompt = create_prompt()
        # 与本地大模型建立连接
        endpoint_url = "http://127.0.0.1:8000"
        llm = ChatGLM(
            endpoint_url = endpoint_url,
            max_token = 80000,
            top_p = 0.9
        )
        # 生成模型的LLMChain链
        chain = LLMChain(llm=llm, prompt=prompt)
        # 模型进行回答
        ans = chain.run({"message":message,"input":input})

    else:
        print(f"向量库 '{vector_db_name}' 不存在，准备创建向量库。")

        # 首先读入文本
        document = load_documents(txt_path)
        # 接着分割文本
        documents = splitter_documents(document)
        # 生成嵌入模型
        embedding = embedding_documents()
        # 构建向量库
        house = house_build(txt_path,documents,embedding)
        # 搜索相似文本
        message = house.similarity_search(input,k)
        # 生成提示体prompt
        prompt = create_prompt()
        # 与本地大模型建立连接
        endpoint_url = "http://127.0.0.1:8000"
        llm = ChatGLM(
            endpoint_url = endpoint_url,
            max_token = 80000,
            top_p = 0.9
        )
        # 生成模型的LLMChain链
        chain = LLMChain(llm=llm, prompt=prompt)
        # 模型进行回答
        ans = chain.run({"message":message,"input":input})

    print(ans)
    print(message)
    return ans





# generate("/home/tcx/DLzhlwd/novel/《雪中悍刀行》.txt","徐凤年的父亲是？",2)

# document = load_documents("/home/tcx/DLzhlwd/rag/phi3rag/archive2/zhongyi.txt")
#         # 接着分割文本
# documents = splitter_documents(document)
# print(documents)

