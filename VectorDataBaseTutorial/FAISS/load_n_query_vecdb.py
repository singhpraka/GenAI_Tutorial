import os,yaml
import random
from datetime import datetime as dt
import logging

from utils import utils
def load_config():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config
config = load_config()

def load_vecdb():
    logging.info("loading pdf or txt embeddings")
    vec_db = utils.load_db(config['faiss_indexstore']['vec_db_path'],config['faiss_indexstore']['index_name']) 
    return vec_db

def query_faq_db():
    logging.info(f"Vector DB: Loading FAQ Vector database")
    faq_vec_db = utils.load_db(config['faiss_indexstore']['vec_db_path'],"faq_Samsung")
    user_query = config['user_faq_query']
    selected_qry = user_query[-1]
    logging.info(f"User Query: {selected_qry}")
    #search_results = faq_vec_db.similarity_search(selected_qry, k=1)
    search_results = faq_vec_db.similarity_search_with_score(selected_qry, k=1)
    res = search_results[0][0].page_content
    res = res.split("response:")[-1]  
    logging.info(f"FAQ VectorDB Search Result Score: {search_results[0][1]}")
    logging.info(f"FAQ VectorDB Search Result:\n\n{res}\n\n")  
    

def query_email_template():
    logging.info(f"Vector DB: Loading template vector database")
    temp_vec_db = utils.load_db(config['faiss_indexstore']['vec_db_path'],"email_template_Samsung")
    user_query = config['user_template_query']
    #selected_qry = random.choice(user_query)
    selected_qry = user_query[-1]
    logging.info(f"Input Query: {selected_qry}")
    search_results = temp_vec_db.similarity_search(selected_qry, k=1)
    res = search_results[0][0].page_content
    res = res.split("template:")[-1]
    logging.info(f"E-mail Template VectorDB Search Result Score: {search_results[0][1]}")
    logging.info(f"E-mail Template:\n\n{res}\n\n")
    

if __name__ == '__main__':
    start_timer = dt.now()
    log_file_name = f"{config['log_file_path']}/{int(dt.now().timestamp())}_load_n_query_vecdb.log"
    log_format='%(asctime)s - %(funcName)s - %(lineno)d : %(message)s'
    logging.basicConfig(filename= log_file_name, filemode='w',format= log_format, level=logging.INFO)    
    logging.info(f"Log File: Log file successfully created here:{log_file_name}")  
    query_faq_db()
    #query_email_template()
    end_timer = dt.now()
    logging.info(f"Total Duration:{end_timer-start_timer}") 