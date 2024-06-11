from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import torch
def create_vector_from_text():
    # Load the CSV file
    df = pd.read_csv('questions_answers.csv')
    
    # Concatenate question and answer into a single text
    raw_text = ''
    for index, row in df.iterrows():
        raw_text += row['question'] + ' ' + row['answer'] + '\n'
    
    # Split the concatenated text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_text(raw_text)
    # breakpoint()
    # Initialize PhoBERT
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    model = AutoModel.from_pretrained("vinai/phobert-base")
    
    def get_embeddings(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    # Get the embeddings for each chunk
    embeddings = [get_embeddings(chunk) for chunk in chunks]
    # Initialize a FAISS index
    index = FAISS(embedding_function=None, index=None, docstore=None, index_to_docstore_id=None)
    # Add the embeddings to the index
    index.add_vectors(embeddings, chunks)
    # Save the index
    index.save_local("./dbv2")
    return index

# Call the function to create the vector store
create_vector_from_text()
