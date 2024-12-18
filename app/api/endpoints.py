from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List, Optional
from pydantic import BaseModel
from app.services.embeddings import create_bert_embeddings, create_tfidf_embeddings, create_hybrid_embeddings, store_embeddings_in_db
from app.services.qdrant_client_init import get_qdrant_client
from models.models_init import init_models
from app.services.retrival import hybrid_retriever
router = APIRouter()

model, tokenizer, tfidf_vectorizer = init_models()
client = get_qdrant_client()


class TestRequest(BaseModel):
 
    act_name: str
    collection_name: str
    context_documents: List[str]

class TestResponse(BaseModel):
    answer: str
    referenced_documents: List[str]

    


@router.post("/test", response_model=TestResponse)
async def legal_document_query(request: TestRequest):
    """
    Query legal documents with retrieval-augmented generation
    """
    try:
        
        print('storing in db')
        store_embeddings_in_db(
            filename='Finance',
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
    

    
