from src.utils import upsert_all_pinecone_data, dense_embed

import os
import sys
import logging
import warnings
import pandas as pd
from tqdm import trange
from dotenv import load_dotenv
import pickle
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
hf_key = os.getenv('HUGGINGFACE_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')
dense_embedder_api = os.getenv("HF_API_URL")

# Initialize clients
pc = Pinecone(api_key=pinecone_api_key)

# Define model
chat_model = "llama3-8b-8192"
index = pc.Index('hsi-notes')
namespace = 'Chapter-1'

file_path = '../data/HSI1000-chapter1.pdf'


def load_bm25_instance(pickle_path):
    with open(pickle_path, 'rb') as file:
        bm25_instance = pickle.load(file)
    return bm25_instance

def pretty_print_table_matches(result):
        print(f"Namespace searched: {result['namespace']}\n")
        num_results = len(result['matches'])
        print(f"Top {num_results} relevant chunks found:\n")
        for i in range(num_results):
            print(f"Found on page {int(result['matches'][i]['metadata']['Page'])}:")
            print(f"{result['matches'][i]['metadata']['Table']}")
            print(f"Dotproduct score: {result['matches'][i]['score']}")
            print("-" * 80)
                   
def get_llm_context_table(query, top_k, bm25_instance):
    index_stats = pc.describe_index(os.environ['PINECONE_INDEX_NAME'])
    if index_stats['status']['ready'] and index_stats['status']['state'] == "Ready":
        dense_query = dense_embed(query)
        sparse_query = bm25_instance.encode_queries(query)
        relevant_matches = index.query( 
            namespace=namespace,
            filter={ 
                    'ElementType': 'Table'
                },
            top_k=top_k, 
            vector=dense_query, 
            sparse_vector=sparse_query, 
            include_metadata=True
            )
    pretty_print_table_matches(relevant_matches)
    # ideally its just to combine the first 2 matches. Or maybe to go by dotproduct score and difference 
    context = ""
    for i in range(len(relevant_matches['matches'])):
        context += f"Page: {int(relevant_matches['matches'][i]['metadata']['Page'])} " + relevant_matches['matches'][i]['metadata']['Table'] + "\n"
    return context

def pretty_print_text_matches(result):
        print(f"Namespace searched: {result['namespace']}\n")
        num_results = len(result['matches'])
        print(f"Top {num_results} relevant chunks found:\n")
        for i in range(num_results):
            print(f"Found on page {int(result['matches'][i]['metadata']['Page'])}:")
            print(f"{result['matches'][i]['metadata']['Text']}")
            print(f"Dotproduct score: {result['matches'][i]['score']}")
            print("-" * 80) 
            
def get_llm_context_text(query, top_k, bm25_instance):
    index_stats = pc.describe_index(os.environ['PINECONE_INDEX_NAME'])
    if index_stats['status']['ready'] and index_stats['status']['state'] == "Ready":
        dense_query = dense_embed(query)
        sparse_query = bm25_instance.encode(query)
        relevant_matches = index.query( 
            namespace=namespace,
            top_k=top_k, 
            vector=dense_query, 
            sparse_vector=sparse_query, 
            include_metadata=True
            )
    pretty_print_text_matches(relevant_matches)
    # ideally its just to combine the first 2 matches. Or maybe to go by dotproduct score and difference 
    context = ""
    for i in range(len(relevant_matches['matches'])):
        context += f"Page: {int(relevant_matches['matches'][i]['metadata']['Page'])} " + relevant_matches['matches'][i]['metadata']['Text'] + "\n"
    return context

def llama_chat(user_question, k, bm25_instance):
    context = get_llm_context_text(user_question, k, bm25_instance)
    chat = ChatGroq(temperature=0, model_name=chat_model)
    system = '''
            You are a Science Professor in a university. 
            Given the user's question and relevant excerpts from a set of school notes about scientific methodology and the history of science,
            you will also answer the question in a professional tone by including direct quotes from the notes, \
            along with the page number where the answer or answers can be found.
            
            For example:
            Answer:
            Reference Page(s): 
            '''
    human = "{text}"
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human)
        ])
    chain = prompt | chat
    return chain.invoke({"text": f"User Question: " + user_question + "\n\nRelevant section in textbook:\n\n" + context})

def check_pinecone_vectors(namespace):
    # Check if the index exists and has vectors
    index_status = index.describe_index_stats()
    
    # Check if pinecone if namespace in index has vectors
    if index_status['namespaces'][namespace]['vector_count'] > 0:
        index_stats = pc.describe_index()
        # Check if namespace is ready
        if index_stats['status']['ready'] and index_stats['status']['state'] == "Ready":
            return True
        else:
            print("Error connecting to the pinecone index.. Check setup of this pinecone index... Exiting now...")
    else:
        print("There are no vectors in this namespace... Exiting now...")
        return False
    
def main():
    namespace = "your_namespace"
    has_vectors = check_pinecone_vectors(namespace)

    if has_vectors:
        user_input = input("Found existing data in Pinecone. Do you want to upload a new file? (yes/no): ")
    else:
        user_input = input("No data found in Pinecone. Do you want to upload a new file? (yes/no): ")

    if user_input.lower() == 'yes':
        file_path = input("Enter the file path: ")
        # Call your function to upload data
        upsert_all_pinecone_data(file_path)
    else:
        bm25_instance = load_bm25_instance('components/bm25_model.pkl')
        query = input("Enter your query: ")
        top_k = int(input("Enter the value for top_k: "))
        # Call your function to query data
        result = llama_chat(query, top_k, bm25_instance)
        print(result)    
    
        
if __name__ == "__main__":
    main()
    
    
## TODOS:
# Re run the hsi_qa.py to generate the pickle model, then 
# Fix bug of groq chat not working for the table part