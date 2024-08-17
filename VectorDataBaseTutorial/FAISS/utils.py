from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
import tiktoken
from glob import glob
from tqdm import tqdm
import pathlib
import yaml
import pandas as pd

def load_config():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

def num_tokens_from_string(customer_queries):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    all_emails = "\n\n".join(f"e-mail {index+1}:\n\n{email}" for index, email in enumerate(customer_queries))    
    num_tokens = len(encoding.encode(all_emails))
    return num_tokens

def embedding_pdf_documents(directory : str):
    """Loads all documents from a directory and returns a list of Document objects
    args: directory format = directory/
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = config["TextSplitter"]["chunk_size"], 
                                                   chunk_overlap = config["TextSplitter"]["chunk_overlap"])
    documents = []
    for item_path in tqdm(glob(directory + "*.pdf")):
        loader = PyPDFLoader(item_path)
        documents.extend(loader.load_and_split(text_splitter=text_splitter))
    
    embeddings = SentenceTransformerEmbeddings(model_name=config['embeddings']['model_name']) #"all-MiniLM-L6-v2"
    pdf_vec_db = FAISS.from_documents(documents,embeddings)
    return pdf_vec_db

def embedding_text_documents(directory : str):
    """Loads all documents from a directory and returns a list of Document objects
    args: directory format = directory/
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = config["TextSplitter"]["chunk_size"], 
                                                   chunk_overlap = config["TextSplitter"]["chunk_overlap"])

    docs = [
    Document(page_content=open(doc, encoding='utf-8').read(), metadata={"filename": pathlib.Path(doc).stem})    
    for doc in tqdm(glob(directory + "*.txt"))
    ]

    docs = text_splitter.split_documents(docs)
    embeddings = SentenceTransformerEmbeddings(model_name=config['embeddings']['model_name']) #"all-MiniLM-L6-v2"
    txt_vec_db = FAISS.from_documents(docs, embeddings)
    return txt_vec_db   

def embedding_excel_faq_documents(excel_file_path, sheet_name):
    df = pd.read_excel(excel_file_path,sheet_name=sheet_name)
    #query_response = []
    combined_user_query_res = [Document(page_content = f"query: {row['User_Query']}\n\nresponse:{row['Response']}",
                                        metadata = {"file_Name":excel_file_path, "Sheet_Name":sheet_name,"Category":row['Category']})
                                        for _,row in df.iterrows()
                                        ]    
    embeddings = SentenceTransformerEmbeddings(model_name=config['embeddings']['model_name']) #"all-MiniLM-L6-v2"
    faq_vectordb = FAISS.from_documents(combined_user_query_res, embeddings)    
    return faq_vectordb

def embedding_excel_email_template(excel_file_path, sheet_name):
    df = pd.read_excel(excel_file_path,sheet_name=sheet_name)
    #query_response = []
    combined_user_query_res = [Document(page_content = f"query: {row['Input']}\n\ntemplate:{row['Templates']}",
                                        metadata = {"file_Name":excel_file_path, "Sheet_Name":sheet_name})
                                        for _,row in df.iterrows()
                                        ]    
    embeddings = SentenceTransformerEmbeddings(model_name=config['embeddings']['model_name']) #"all-MiniLM-L6-v2"
    email_template_vectordb = FAISS.from_documents(combined_user_query_res, embeddings)    
    return email_template_vectordb    

def save_db(db, vec_db_path, index_name):
    db.save_local(vec_db_path, index_name)
    
def load_db(save_path, index_name):
    embeddings = SentenceTransformerEmbeddings(model_name=config['embeddings']['model_name']) #"all-MiniLM-L6-v2"
    db = FAISS.load_local(folder_path=save_path, index_name=index_name, embeddings = embeddings)
    return db    