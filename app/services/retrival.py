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
    sections_json_path: str,
    top_k: int = 3,
    schemantic_weight: float = 0.7,
    rerank_weight: float = 0.4
):
    """
    Enhanced hybrid retrieval with cross-encoder reranking on chunks
    and final section merging from JSON file
    """
    import json
    
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
        print('initial score:', initial_score)
        
        results.append({
            'section_num': result.payload.get('section_num'),
            'title': result.payload.get('title'),
            'content': result.payload.get('content'),
            'schemantic_similarity': float(result.score),
            'tfidf_similarity': float(tfidf_similarity),
            'initial_score': initial_score
        })
    
    # Batch reranking
    rerank_scores = reranker.predict(pairs_to_rerank)
    print('rerank scores:', rerank_scores)
# After getting rerank_scores, add normalization
    min_score = min(rerank_scores)
    max_score = max(rerank_scores)
    normalized_rerank_scores = [(score - min_score) / (max_score - min_score) 
                           if max_score != min_score else 0.5 
                           for score in rerank_scores]
    print('normalized rerank scores:', normalized_rerank_scores)
    # Then use normalized_rerank_scores instead of rerank_scores
    for idx, result in enumerate(results):
        result['rerank_score'] = normalized_rerank_scores[idx]
        result['combined_similarity'] = (
            (1 - rerank_weight) * result['initial_score'] + 
            rerank_weight * result['rerank_score']
        )
        print('combined similarity:', result['combined_similarity'])
    
    # Get top_k ranked results
    ranked_results = sorted(results, key=lambda x: x['combined_similarity'], reverse=True)[:top_k]
    
    # Load complete sections from JSON
    with open(sections_json_path, 'r') as f:
        complete_sections = json.load(f)
    
    # Convert to dictionary for easy lookup
    sections_dict = {str(section['section_num']): section for section in complete_sections}
    
    # Get unique section numbers from ranked results
    unique_sections = {str(result['section_num']) for result in ranked_results}
    
    # Prepare final results with complete sections
    final_results = []
    for section_num in unique_sections:
        if section_num in sections_dict:
            # Find the best score for this section from ranked results
            section_results = [r for r in ranked_results if str(r['section_num']) == section_num]
            best_result = max(section_results, key=lambda x: x['combined_similarity'])
            
            # Get complete section from JSON
            complete_section = sections_dict[section_num]
            
            # Ensure content is a string by joining if it's a list
            content = complete_section['content']
            if isinstance(content, list):
                content = ' '.join(content)
            
            final_results.append({
                'section_num': section_num,
                'title': complete_section['title'],
                'content': content,  # Now guaranteed to be a string
                'schemantic_similarity': best_result['schemantic_similarity'],
                'tfidf_similarity': best_result['tfidf_similarity'],
                'combined_similarity': best_result['combined_similarity'],
                'rerank_score': best_result['rerank_score']
            })
    
    # Sort final results by combined similarity
    return sorted(final_results, key=lambda x: x['combined_similarity'], reverse=True)