# Standard library imports
import os
import json
from tqdm import trange
import time
import sys
import warnings
from typing import Any, List, Dict

# Third-party imports
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import requests

# Suppress warnings
warnings.filterwarnings("ignore")

# Pinecone imports
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

# Langchain and related imports
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# Groq imports
from groq import Groq
from langchain_groq import ChatGroq

# Ensure you have the correct Groq import if it's from Langchain

# Unstructured imports
from unstructured.staging.base import elements_to_json, convert_to_dict
from unstructured.partition.pdf import partition_pdf

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
hf_key = os.getenv('HUGGINGFACE_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
model = "llama3-8b-8192"


###################
# Code taken from unstructured website and stack overflow 
path_to_hsi = "../data/HSI1000_1to9.pdf"
raw_pdf_elements = partition_pdf("../data/HSI1000_1to9.pdf", 
                        strategy="hi_res", 
                        hi_res_model_name="yolox",
                        infer_table_structure=True
                        )

# Save output to json file (Future use mongodb maybe)
convert_to_dict(raw_pdf_elements)

element_output_file = "../data/element_entities.json"
elements_to_json(raw_pdf_elements, filename=element_output_file)
with open("../data/element_entities.json", "r", encoding='utf-8') as fin:
    read_elements = json.load(fin)
print(f"length before filtering: {len(read_elements)}")

unwanted_types = ['Footer', 'Image', 'FigureCaption', 'UncategorizedText']
filtered_el = []
for el in read_elements:
    if el['type'] in unwanted_types:
        continue
    else:
        filtered_el.append(el)
print(f"length after filtering: {len(filtered_el)}")
filtered_el[0]
###################


###################
table_elements =  [
    {'type': el['type'], 
     'Page': el['metadata']['page_number'],
     "text": el['metadata']['text_as_html']
     } for el in filtered_el if el['type'] == 'Table']
print(f"Number of tables identified: {len(table_elements)}")
text_elements =  [{'type': el['type'], 
     'Page': el['metadata']['page_number'],
     "text": el['text']
     } for el in filtered_el if el['type'] != 'Table']
print(f"Number of text elements identified: {len(text_elements)}")
###################





###################
def get_file_docs(element: List[Dict]) -> List[Dict]:
    def get_num_pages(elements):
        num = 0
        for el in elements:
            if el['Page'] > num:
                num = el['Page']
        return num

    def generate_chunks(text: str, page_num: int) -> List[Dict]:
        separator_ls = ["\n\n", "\n", ". ", "!", "?", ",", " ", ""]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30,
            length_function=len,
            separators=separator_ls
        )
        separated_list = text_splitter.split_text(text)
        # Add page number to each chunk
        return [{'Page': page_num, 'text': chunk} for chunk in separated_list]
    
    file_chunks = []
    num_pages = get_num_pages(text_elements)
    
    for i in range(num_pages):
        page_ls = []
        for el in element:
            if el['Page'] == i:
                page_ls.append(el['text'])
        
        page_text = "\n".join(page_ls)
        text_chunks = generate_chunks(page_text, i)
        file_chunks.extend(text_chunks)
    
    return file_chunks

text_documents = get_file_docs(text_elements)
###################




###################
# Initialize the BM25Encoder and SentenceTransformer model
bm25 = BM25Encoder()

# Load embeddings. Need to change from ...co/models/ to ...co/pipeline/feature-extraction/...
HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-mpnet-base-v2"
headers = {"Authorization": f"Bearer {hf_key}"}

def dense_embed(payload: str) -> str:
	response = requests.post(HF_API_URL, headers=headers, json=payload)
	return response.json()
###################



###################
# # Convert text_documents to DataFrame
# df = pd.DataFrame(text_documents)

# pinecone_upserts = []
# db_dense_embeddings = []
# db_text_chunks = []

# batch_size = 32

# # Create something to check the status of the pinecone index before upserting

# # Loop through the DataFrame 'df' in batches of size 'batch_size'
# for i in trange(0, len(df), batch_size):
#     i_end = min(i+batch_size, len(df)) # Determine the end index of the current batch
#     df_batch = df.iloc[i:i_end] # Extract the current batch from the DataFrame
#     df_dict = df_batch.to_dict(orient="records") # Convert the batch to a list of dictionaries
    
#     meta_batch = [
#         f"Page {row['Page']}: {row['text']}" for _, row in df_batch.iterrows()
#     ]
    
#     # bm25.fit(meta_batch)

#     text_chunks = df_batch['text'].tolist()
#     db_text_chunks.extend(text_chunks)
    
#     # Encode combined metadata and text using BM25Encoder to create sparse embeddings
#     # sparse_embeddings = bm25.encode_documents([combined for combined in meta_batch])

#     # Encode text using SentenceTransformer to create dense embeddings
#     dense_embeddings = dense_embed(text_chunks)
#     db_dense_embeddings.extend(dense_embeddings)
    
#     # Generate a list of IDs for the current batch
#     ids = ['vec' +str(x) for x in range(i, i_end)]
#     time.sleep(10)
#     pinecone_batch_upserts = []
    
#     for _id, dense, meta in zip(ids, dense_embeddings, df_dict):
#         pinecone_batch_upserts.append({
#             'id': _id,
#             'values': dense,
#             'metadata': meta
#         })
    
#     index = pc.Index('hsi-notes')
    
#     # RUN ONLY WHEN WANT TO UPSERT NEW BATCH
#     if isinstance(dense_embeddings, list):
#         upsert_response = index.upsert(vectors = pinecone_batch_upserts, namespace='page-1to9-texts')
#     else:
#         print("Embedding model not connected properly. Dense embeddings not generated. ")
#         sys.exit()
#     print(f"Batch starting with index {i} upserted")
#     pinecone_upserts.append(pinecone_batch_upserts)
###################



###################
def get_relevant_chunks(query, top_k):
    # Create dense vector of user query
    dense_query = dense_embed(query)
    matches = index.query( 
        namespace='page-1to9-texts',
        top_k=top_k, 
        vector=dense_query, 
        include_metadata=True
        )
    return matches

def pretty_print_matches(result):
    print(f"Namespace searched: {result['namespace']}\n")
    num_results = len(result['matches'])
    print(f"Top {num_results} relevant chunks found:\n")
    for i in range(num_results):
        print(f"Found on page {int(result['matches'][i]['metadata']['Page'])}:")
        print(f"{result['matches'][i]['metadata']['text']}")
        print(f"Dotproduct score: {result['matches'][i]['score']}")
        print("-" * 80)

def get_llm_context(query, top_k):
    index_stats = pc.describe_index(os.environ['PINECONE_INDEX_NAME'])
    if index_stats['status']['ready'] and index_stats['status']['state'] == "Ready":
        relevant_matches = get_relevant_chunks(query, top_k)        
    # ideally its just to combine the first 2 matches. Or maybe to go by dotproduct score and difference 
    context = ""
    for i in range(len(relevant_matches['matches'])):
        context += f"Page number: {int(relevant_matches['matches'][i]['metadata']['Page'])}" + relevant_matches['matches'][i]['metadata']['text'] + "\n"
    return context
###################





###################

def llama_chat(user_question, k):
    context = get_llm_context(user_question, k)
    chat = ChatGroq(temperature=0, model_name="llama3-8b-8192")
    system = '''
            You are a science professor in a university. Given the user's question and relevant sections from a set of school notes about scientific methodology and the history of science.
            You will also answer the question by including direct quotes from the notes, along with the page number where the answer or answers can be found.
            '''
    human = "{text}"
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system", system
            ),
            (
                "human", human
                )
        ]
    )
    chain = prompt | chat
    return chain.invoke({"text": f"User Question: " + user_question + "\n\nRelevant section in textbook:\n\n" + context})

answer = llama_chat("What is Cadaverous Poisoning?", 5)
print(answer.content)
###################



###################

###################

