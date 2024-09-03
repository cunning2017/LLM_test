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
        chunk_size=32,
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

# # 文档搜索函数
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




generate("/home/tcx/DLzhlwd/novel/《雪中悍刀行》语义分割版.txt","徐凤年姐姐是谁",5)



############################# 以下是以大于0.6为阈值分割 #######################################


# # 假设 paragraphs_or_sentences 已经初始化  
# paragraphs_or_sentences = []  
# # 初始化一个变量来存储当前正在拼接的句子链  
# current_paragraph = ""  
# # 遍历 demo 列表中的每个元素，除了最后一个（因为不需要检查与不存在的下一个元素的相似度）  
# for i in range(len(demo) - 1):  
#     # 如果当前没有正在拼接的句子链，则开始一个新的  
#     if not current_paragraph:  
#         current_paragraph = demo[i].page_content  
#     # 计算当前 Document 和下一个 Document 的相似度分数  
#     score = model.get_score(demo[i].page_content, demo[i + 1].page_content)  
#     # 如果相似度分数达到或超过60%，则继续拼接  
#     if score >= 0.6:  
#         # 添加分隔符（如果需要的话），然后添加下一个句子的内容  
#         current_paragraph += "。" + demo[i + 1].page_content  
#     else:  
#         # 如果相似度不足，将当前拼接的句子链添加到列表中，并重置以开始新的链  
#         paragraphs_or_sentences.append(current_paragraph)  
#         current_paragraph = demo[i + 1].page_content  # 也可以不立即设置，视情况而定  

# # 不要忘记添加最后一个句子链（如果它存在的话）  
# if current_paragraph:  
#     paragraphs_or_sentences.append(current_paragraph)  


########################################################################################




###############################  以下是以0.05浮动为阈值分割  ###############################

# # 假设 paragraphs_or_sentences 和 demo 已经正确初始化  
# paragraphs_or_sentences = []  
# # 初始化一个变量来存储当前正在拼接的句子链  
# current_paragraph = ""  
# # 初始化一个变量来存储前一个文档的相似度分数（初始化为None，表示还没有前一个分数）  
# prev_score = None  
  
# # 遍历 demo 列表中的每个元素，除了最后一个  
# for i in range(len(demo) - 1):  
#     # 如果当前没有正在拼接的句子链，则开始一个新的  
#     if not current_paragraph:  
#         current_paragraph = demo[i].page_content  
      
#     # 计算当前 Document 和下一个 Document 的相似度分数  
#     score = model.get_score(demo[i].page_content, demo[i + 1].page_content)  
      
#     # 检查是否有前一个相似度分数以计算差值  
#     if prev_score is not None:  
#         # 如果当前分数与前一个分数的差值小于0.05，则继续拼接  
#         if abs(score - prev_score) < 0.05:  
#             current_paragraph += "。" + demo[i + 1].page_content  
#         else:  
#             # 如果差值不小于0.05，则将当前拼接的句子链添加到列表中，并重置以开始新的链  
#             paragraphs_or_sentences.append(current_paragraph)  
#             current_paragraph = demo[i + 1].page_content  
      
#     # 更新前一个相似度分数  
#     prev_score = score  
  
# # 遍历结束后，检查是否还有未添加的句子链  
# if current_paragraph:  
#     paragraphs_or_sentences.append(current_paragraph)  
  

###########################################################################################




