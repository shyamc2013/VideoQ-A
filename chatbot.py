import streamlit as st

import warnings
warnings.filterwarnings("ignore")

import os
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import Dict
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch

os.environ["OPENAI_API_KEY"] = "b7d5aa82d15a4b99a1c730f681ec2bbc"
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://hmh-digitalhub-azure-openai.openai.azure.com/"
os.environ["CHAT_MODEL"] = "gpt-35-turbo"
os.environ["CHAT_MODEL_DEPLOYMENT_NAME"] = "gpt-35-turbo"
os.environ["EMBEDDINGS_MODEL"] = "text-embedding-ada-002"
os.environ["EMBEDDINGS_MODEL_DEPLOYMENT_NAME"] = "text-embedding-ada-002"

llm = AzureChatOpenAI(  
            model_name ="gpt-35-turbo",
            deployment_name= "gpt-35-turbo",
            temperature=0,
            openai_api_version ="2023-07-01-preview",
            openai_api_key="b7d5aa82d15a4b99a1c730f681ec2bbc",
            azure_endpoint="https://hmh-digitalhub-azure-openai.openai.azure.com/"
        )

embeddings = AzureOpenAIEmbeddings()


st.title('Conversion RAG chatbot')