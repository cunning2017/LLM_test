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

from ragas.langchain.evalchain import RagasEvaluatorChain
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.metrics import faithfulness, answer_relevancy, context_relevancy, context_recall
from ragas.langchain.evalchain import RagasEvaluatorChain


# 文档加载函数
def load_documents(txt_path):
    loader = TextLoader(txt_path)
    document = loader.load()
    return document

# 文档分割函数
def splitter_documents(document):
    text_splitter = CharacterTextSplitter(chunk_size = 256, chunk_overlap = 32)
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





######################### 生成函数 #####################################

def generate(txt_path,input,k):

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
    prompt = PromptTemplate(template="你是一个助手，需要根据参考资料回答我的问题。\n参考资料：{message}\n问题：{input}",)
    # 与本地大模型建立连接
    endpoint_url = "http://127.0.0.1:8000"
    llm = ChatGLM(
        endpoint_url = endpoint_url,
        max_token = 80000,
        top_p = 0.9
    )
    # 生成模型的Qa链
    retriever = house.as_retriever()
    qa = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = 'stuff',
    retriever = retriever)
    # 模型进行回答
   
    ans = qa.run(input)
    print(ans)
    return ans


result = generate("/home/tcx/DLzhlwd/novel/《雪中悍刀行》.txt","徐凤年的弟弟是谁？",5)
# result_with_truth = result
# result_with_truth["ground_truths"] = "徐龙象"
# # create evaluation chains
# eval_chains = {
#     m.name: RagasEvaluatorChain(metric=m) 
#     for m in [faithfulness, answer_relevancy, context_relevancy, context_recall]
# }
# # evaluate
# for name, eval_chain in eval_chains.items():
#     score_name = f"{name}_score"
#     print(f"{score_name}: {eval_chain(result)[score_name]}")


