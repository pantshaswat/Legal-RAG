from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List, Optional
from pydantic import BaseModel
from app.services.embeddings import create_bert_embeddings, create_tfidf_embeddings, create_hybrid_embeddings, store_embeddings_in_db
from app.services.qdrant_client_init import get_qdrant_client
from models.models_init import init_models
from app.services.retrival import hybrid_retriever
from app.services.llm_prompt import get_answer
router = APIRouter()

model, tokenizer, tfidf_vectorizer = init_models()
client = get_qdrant_client()


import os
import json

# Path to the JSON metadata file
FILES_METADATA_PATH = os.path.abspath("../data/files_metadata.json")

@router.get("/available-files", response_model=List[dict])
async def get_available_files():
    """
    Get a list of available files with descriptions.
    """
    try:
        with open(FILES_METADATA_PATH, "r", encoding="utf-8") as f:
            files_metadata = json.load(f)
        return files_metadata
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Files metadata not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid metadata format")


class TestRequest(BaseModel):
 
    act_name: str
    collection_name: str
    context_documents: List[str]
    file_name: str

class TestResponse(BaseModel):
    answer: str
    referenced_documents: List[str]

    


@router.post("/test", response_model=TestResponse)
async def legal_document_query(request: TestRequest):
    """
    Query legal documents with retrieval-augmented generation
    """
    try:
        file_path = os.path.abspath(f"../data/processed/json/Finance/{request.file_name}.json")
        print('storing in db')
        store_embeddings_in_db(
            file_path=file_path,
            collection_name=request.collection_name,
            client=client,
            model=model,
            tokenizer=tokenizer,
            tfidf_vectorizer=tfidf_vectorizer
        )
        print('stored in db')

        response = QueryResponse(
            answer="Stored embeddings in the database",
            referenced_documents=request.context_documents
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

class QueryRequest(BaseModel):
    question: str
    act_name: str
    collection_name: str
    context_documents: List[str]

class QueryResponse(BaseModel):
    answer: List




@router.post("/query", response_model=QueryResponse)
async def legal_document_query(request: QueryRequest):
    """
    Query legal documents with retrieval-augmented generation
    """
    try:
       
        result = hybrid_retriever(
           client=client,
           collection_name=request.collection_name,
           query=request.question,
           model=model,
           tokenizer=tokenizer,
           tfidf_vectorizer=tfidf_vectorizer,
        )
        

        response = QueryResponse(
            answer=result,
            
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


class EvaluationRequest(BaseModel):
    retrieved_documents: List[List[str]]
    relevant_documents: List[List[str]]

class EvaluationResponse(BaseModel):
    mrr: float
    map: float




@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_retrieval_performance(request: EvaluationRequest):
    """
    Evaluate retrieval performance using MRR and MAP.
    """
    try:
        def calculate_mrr(retrieved_docs: List[List[str]], relevant_docs: List[List[str]]) -> float:
            reciprocal_ranks = []
            for retrieved, relevant in zip(retrieved_docs, relevant_docs):
                rank = 0
                for idx, doc in enumerate(retrieved, start=1):
                    if doc in relevant:
                        rank = 1 / idx
                        break
                reciprocal_ranks.append(rank)
            return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

        def calculate_map(retrieved_docs: List[List[str]], relevant_docs: List[List[str]]) -> float:
            average_precisions = []
            for retrieved, relevant in zip(retrieved_docs, relevant_docs):
                precision_at_k = []
                relevant_count = 0
                for k, doc in enumerate(retrieved, start=1):
                    if doc in relevant:
                        relevant_count += 1
                        precision_at_k.append(relevant_count / k)
                average_precisions.append(sum(precision_at_k) / len(relevant) if relevant else 0.0)
            return sum(average_precisions) / len(average_precisions) if average_precisions else 0.0

        # Calculate MRR and MAP
        mrr = calculate_mrr(request.retrieved_documents, request.relevant_documents)
        map_score = calculate_map(request.retrieved_documents, request.relevant_documents)

        return EvaluationResponse(mrr=mrr, map=map_score)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))