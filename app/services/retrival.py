import numpy as np
from sentence_transformers import CrossEncoder
import torch
from typing import List, Dict
from app.services.embeddings import create_bert_embeddings, create_sentence_embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.services.vectorizer_manager import VectorizerManager
from app.services.llm_prompt import get_query_expansion
from app.services.grokLlm import expand_query

def hybrid_retriever(
    client,
    collection_name,
    query,
    model,
    tokenizer,
    tfidf_vectorizer,
    top_k: int = 5,
    bert_weight: float = 0.7
) -> List[Dict]:
    query = get_query_expansion(query)
    print('expanded query', query)
    # Convert query embedding to list of floats
    query_embedding = create_bert_embeddings([query], model, tokenizer)
    print('shape of query embedding', query_embedding.shape)
    
    # Convert NumPy array to list
    query_embedding_list = query_embedding[0].tolist()

    vectorizer_manager = VectorizerManager()
    vectorizer = vectorizer_manager.get_vectorizer(collection_name)
    query_tfidf = vectorizer.transform([query])

    points_count = client.count(collection_name=collection_name)
    print(f"Number of points in collection: {points_count}")

    # Pass list of floats to Qdrant
    qdrant_results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding_list,
        limit=top_k
    )

    results = []

    for result in qdrant_results:
        # Ensure TF-IDF vector is properly converted
        tfidf_vector = np.array(result.payload['tfidf_vector']).reshape(1, -1)
        print('shape of tfidf vector', tfidf_vector.shape)
        
        # Cosine similarity returns a scalar, not an array
        tfidf_similarity = cosine_similarity(query_tfidf, tfidf_vector)[0][0]

        # Weighted combination
        combined_similarity = bert_weight * result.score + (1 - bert_weight) * tfidf_similarity

        results.append({
            'section_num': result.payload.get('section_num'),
            'content': result.payload.get('content'),
            'bert_similarity': float(result.score),  # Ensure float
            'tfidf_similarity': float(tfidf_similarity),  # Ensure float
            'combined_similarity': float(combined_similarity)  # Ensure float
        })

    # Sort results by combined similarity
    return sorted(results, key=lambda x: x['combined_similarity'], reverse=True)

def hybrid_retriever_reranked(
    client,
    collection_name: str,
    query: str,
    model,
    tfidf_vectorizer,
    top_k: int = 5,
    schemantic_weight: float = 0.7,
    rerank_weight: float = 0.4
):
    """
    Enhanced hybrid retrieval with cross-encoder reranking
    """
    # Initialize cross-encoder for reranking
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Initial retrieval
    query = expand_query(query)
    print('query after expansion:', query)
    query_embedding = create_sentence_embeddings([query], model)
    vectorizer_manager = VectorizerManager()
    vectorizer = vectorizer_manager.get_vectorizer(collection_name)
    query_tfidf = vectorizer.transform([query])
    
    # Get more candidates for reranking
    initial_results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding[0].tolist(),
        limit=top_k * 3,
        with_payload=True
    )
    
    results = []
    pairs_to_rerank = []
    
    for result in initial_results:
        payload = result.payload
        document = payload.get('content', 'No content')
        
        # Prepare pairs for batch reranking
        pairs_to_rerank.append([query, document])
        
        try:
            tfidf_vector = np.array(payload.get('tfidf_vector', [])).reshape(1, -1)
            tfidf_similarity = cosine_similarity(query_tfidf, tfidf_vector)[0][0]
        except Exception as e:
            print(f"TF-IDF computation error: {e}")
            tfidf_similarity = 0
        
        schemantic_score = result.score
        initial_score = (schemantic_weight * schemantic_score + 
                        (1 - schemantic_weight) * tfidf_similarity)
        
        results.append({
            'section_num': result.payload.get('section_num'),
            'title': result.payload.get('title'),
            'content': result.payload.get('content'),
            'schemantic_similarity': float(result.score),  # Ensure float
            'tfidf_similarity': float(tfidf_similarity),  # Ensure float
            'initial_score': initial_score
        })
    
    # Batch reranking
    rerank_scores = reranker.predict(pairs_to_rerank)
    
    # Combine scores
    for idx, result in enumerate(results):
        result['rerank_score'] = rerank_scores[idx]
        result['combined_similarity'] = (
            (1 - rerank_weight) * result['initial_score'] + 
            rerank_weight * result['rerank_score']
        )
    
    
    
    return sorted(results, key=lambda x: x['combined_similarity'], reverse=True)[:top_k]