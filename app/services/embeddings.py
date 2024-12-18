import numpy as np
import torch
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from app.services.vectorizer_manager import VectorizerManager
import os

def create_bert_embeddings(texts, model, tokenizer):
    """
    Get embeddings for a list of texts using a transformer model
    
    Args:
        texts (List[str]): List of input texts
        model: Transformer model
        tokenizer: Corresponding tokenizer
    
    Returns:
        numpy.ndarray: 2D array of embeddings
    """

    
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract the first token's embedding (usually [CLS] token)
    # Ensure we get a 2D numpy array, even for a single text
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    return embeddings

def create_tfidf_embeddings(texts: list, tfidf_vectorizer: TfidfVectorizer, collection_name: str):
    """
    Create TF-IDF embeddings for a list of texts.
    
    Args:
        texts (list): List of text strings
        
    Returns:
        scipy.sparse.csr_matrix: TF-IDF embeddings matrix
    """
    vectorizer_manager = VectorizerManager()
    tfidf_matrix = vectorizer_manager.create_vectorizer(collection_name, texts)
    return tfidf_matrix

def create_hybrid_embeddings(chunks, model, tokenizer, tfidf_vectorizer, collection_name: str):
    """
    Creates both BERT and TF-IDF embeddings for each chunk's full text.
    
    Args:
        chunks (list[dict]): List of processed chunks
        
    Returns:
        tuple: (bert embeddings, TF-IDF embeddings, processed texts)
    """
    processed_texts = [chunk['full_text'] for chunk in chunks]
    
    bert_embeddings = []
    # Create BERT embeddings
    for text in processed_texts:
        bert_embedding = create_bert_embeddings(text, model, tokenizer)
        bert_embeddings.append(bert_embedding.squeeze(0))  # Remove extra dimension
    
    # Convert to numpy array with shape (num_chunks, 1024)
    bert_embeddings = np.array(bert_embeddings)
    print(f"Created BERT embeddings with shape: {bert_embeddings.shape}")
    
    # Create TF-IDF embeddings
    tfidf_embeddings = create_tfidf_embeddings(processed_texts, tfidf_vectorizer, collection_name)

    return bert_embeddings, tfidf_embeddings, processed_texts


def process_json_chunks(json_chunks):
    """
    Process JSON chunks to create separate entries for each content array element.
    Each element will include section number and title.
    
    Args:
        json_chunks (list): List of dictionaries containing chunk data
        
    Returns:
        list[dict]: Processed chunks with individual content elements
    """
    
    processed_chunks = []
    for chunk in json_chunks:
        section_num = chunk['section_num']
        title = chunk['title']
        content_array = chunk['content'] if isinstance(chunk['content'], list) else [chunk['content']]
        
        for content_element in content_array:
            processed_chunk = {
                'section_num': str(section_num),
                'title': title,
                'content': content_element.strip(),
                'full_text': f"Section {section_num}: {title}\n\n{content_element.strip()}"
            }
            processed_chunks.append(processed_chunk)
    
    
    return processed_chunks

def store_embeddings_in_db(filename, collection_name, client, model, tokenizer, tfidf_vectorizer):
    try:
        file_path = os.path.abspath("../data/processed/json/Finance/banking-offence-and-punishment-act.json")
        print(f"Using file path: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            json_chunks = json.load(f)
    except FileNotFoundError:
        print("Error: JSON file not found")
        exit(1)
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
        exit(1)

    processed_chunks = process_json_chunks(json_chunks)
    print(f"Total number of chunks: {len(processed_chunks)}")

    bert_embeddings, tfidf_embeddings, processed_texts = create_hybrid_embeddings(processed_chunks, model, tokenizer, tfidf_vectorizer, collection_name=collection_name)
    print(f"Created BERT embeddings with shape: {bert_embeddings.shape}")
    print(f"Created TF-IDF embeddings with shape: {tfidf_embeddings.shape}")
    
    # Create Qdrant collection with correct size for BERT embeddings
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=1024,  # BERT embedding size
            distance=Distance.COSINE
        )
    )
    print(f"Created Qdrant collection: {collection_name}")

    # Prepare points for Qdrant
    points = [
        PointStruct(
            id=int(doc['section_num']),
            vector=bert_embedding.tolist(),
            payload={
                'section_num': doc['section_num'],
                'title': doc['title'],
                'content': doc['content'],
                'tfidf_vector': tfidf_embedding.toarray().tolist()  # Convert TF-IDF sparse matrix to list
            }
        )
        for doc, bert_embedding, tfidf_embedding 
        in zip(processed_chunks, bert_embeddings, tfidf_embeddings)
    ]
    print(f"Prepared {len(points)} points for upsert")
    
    # Upsert points to Qdrant
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    print("Upserted points to Qdrant")
