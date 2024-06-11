from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Vietcombank Chatbot")
parser.add_argument('--embedding_path', type=str, default='all-MiniLM-L6-v2.gguf2.f16.gguf', help='Path to the embedding model file')
parser.add_argument('--vector_db_path', type=str, default="./data/dbv3", help='Path to the vector database file')
parser.add_argument('--csv_path', type=str, default='./data/questions_answers.csv', help='Path to the CSV file containing questions and answers')
args = parser.parse_args()

def create_vector_from_text():
    df = pd.read_csv(args.csv_path)
    # breakpoint()
    raw_answer = ''
    raw_question = ''
    chunks = []
    for index, row in df.iterrows():
        raw_answer += row['answer'] + ' \n '
        chunks.append(row['question'] + '?')
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 700,
        chunk_overlap = 0,
        length_function = len,
    )
    # breakpoint()
    chunks += text_splitter.split_text(raw_answer)
    
    # breakpoint()
    model_name = args.embedding_path
    gpt4all_kwargs = {'allow_download': 'True'}
    embedding_model = GPT4AllEmbeddings(
        model_name=model_name,
        gpt4all_kwargs=gpt4all_kwargs
    )
    db = FAISS.from_texts(texts=chunks, embedding = embedding_model)
    db.save_local(args.vector_db_path)
    return db

create_vector_from_text()