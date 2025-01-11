from fastapi import APIRouter, HTTPException, Path, Query, UploadFile, File
from typing import List, Optional
from pydantic import BaseModel, Field
from app.services.embeddings import create_bert_embeddings, create_tfidf_embeddings, create_hybrid_embeddings, store_embeddings_in_db
from app.services.qdrant_client_init import get_qdrant_client
from models.models_init import init_models
from app.services.retrival import hybrid_retriever, hybrid_retriever_reranked
from app.services.llm_prompt import get_answer
router = APIRouter()
import logging
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model, tfidf_vectorizer = init_models()
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
    

class ChunkInfo(BaseModel):
    content: str = Field(..., description="Content of the document chunk")
    section_num: str = Field(..., description="Section number of the document chunk")
    title: str = Field(..., description="Title of the section")
    schemantic_similarity: float = Field(..., description="Schemantic similarity score")
    tfidf_similarity: float = Field(..., description="TF-IDF similarity score")
    combined_similarity: float = Field(..., description="Combined similarity score")
    collection_name: str = Field(..., description="ID of the source document")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="LLM-generated answer")
    collection_name: str = Field(..., description="Document ID that was queried")
    referenced_chunks: List[ChunkInfo] = Field(..., description="Source chunks used for the answer")
    confidence_score: float = Field(..., description="Overall confidence score for the answer")





@router.post("/query", response_model=QueryResponse)
async def legal_document_query(      
    collection_name: str,
    question: str = Query(..., description="Question to ask about the document")):
    """
    Query legal documents with retrieval-augmented generation
    """
    try:
       
        results = hybrid_retriever_reranked(
           client=client,
           collection_name=collection_name,
           query=question,
           model=model,
           tfidf_vectorizer=tfidf_vectorizer,
        )
        if not results:
            return QueryResponse(
                answer="No relevant information found in this document.",
                collection_name=collection_name,
                referenced_chunks=[],
                confidence_score=0.0
            )
        chunks = [
            ChunkInfo(
                content=result['content'],
                section_num=result['section_num'],
                title=result['title'],
                schemantic_similarity=result['schemantic_similarity'],
                tfidf_similarity=result['tfidf_similarity'],
                combined_similarity=result['combined_similarity'],
                collection_name=collection_name,
               
            )
            for result in results
        ]

        documents = [chunk.content for chunk in chunks]
        llm_answer = get_answer(question, documents)

        confidence_score = sum(r['combined_similarity'] for r in results) / len(results)

        return QueryResponse(
            answer=llm_answer,
            collection_name=collection_name,
            referenced_chunks=chunks,
            confidence_score=confidence_score
        )
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