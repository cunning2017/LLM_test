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
from sentence_transformers import SentenceTransformer

from langchain_core.documents.base import Document
from langchain import LLMChain


import sys
sys.path.append('...')
from text2vec import Similarity

model = Similarity('/home/tcx/DLzhlwd/model/embedding/shibing624-text2vec-base-chinese')

# 文档加载函数
def load_documents(txt_path):
    loader = TextLoader(txt_path)
    document = loader.load()
    return document

# 文档分割函数
def splitter_documents(document):
    text_splitter = CharacterTextSplitter(
        separator="。",
        chunk_size=5,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=True,
    )

    documents = text_splitter.split_documents(document)

        # 假设 paragraphs_or_sentences 已经初始化  
    paragraphs_or_sentences = []  
    # 初始化一个变量来存储当前正在拼接的句子链  
    current_paragraph = ""  
    # 遍历 demo 列表中的每个元素，除了最后一个（因为不需要检查与不存在的下一个元素的相似度）  
    for i in range(len(documents) - 1):  
        # 如果当前没有正在拼接的句子链，则开始一个新的  
        if not current_paragraph:  
            current_paragraph = documents[i].page_content  
        # 计算当前 Document 和下一个 Document 的相似度分数  
        score = model.get_score(documents[i].page_content, documents[i + 1].page_content)  
        # 如果相似度分数达到或超过60%，则继续拼接  
        if score >= 0.6:  
            # 添加分隔符（如果需要的话），然后添加下一个句子的内容  
            current_paragraph += "。" + documents[i + 1].page_content  
        else:  
            # 如果相似度不足，将当前拼接的句子链添加到列表中，并重置以开始新的链  
            paragraphs_or_sentences.append(current_paragraph)  
            current_paragraph = documents[i + 1].page_content  # 也可以不立即设置，视情况而定  

    # 不要忘记添加最后一个句子链（如果它存在的话）  
    if current_paragraph:  
        paragraphs_or_sentences.append(current_paragraph)  

    updated_documents = [Document(metadata={'source': '/home/tcx/DLzhlwd/novel/《雪中悍刀行》语义分割版.txt'}, page_content=p) for p in paragraphs_or_sentences]

    return updated_documents

# 文档搜索函数
# def documents_search(input,house,k):
    


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



txt_path = "/home/tcx/DLzhlwd/novel/《雪中悍刀行》语义分割版.txt"
vector_db_name = txt_path.split('/')[-1]
# 生成嵌入模型
embedding = embedding_documents()


# 加载向量库
house = Chroma(persist_directory=vector_db_name,embedding_function=embedding)

print()



# 搜索相似文本
message = house.similarity_search("徐凤年的父亲？",2)

