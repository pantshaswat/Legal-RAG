import os
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

def expand_query(query):
    headers = {
        'Content-Type': 'Application/json',
        'Authorization': f'Bearer {GROQ_API_KEY}'
    }
    response = requests.post(
        'https://api.groq.com/openai/v1/chat/completions',
        headers=headers, 
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{
                "role": "system",
                "content": """You are an AI language model. Your task is to generate five different versions of the given user query
            to retrieve relevant documents from a vector database. Your goal is to help overcome limitations of distance similarity search
            by providing some really good query expansions. The query will be mostly law related. Provide these alternative queries
            separated by commas. These are legal specific documents, so if you 
             diverse the query by more, the result might not be as expected, some other chunk might be fetched, so 
               the version should just contain synonyms or similar words to the original query
            and not necessarily be a question, just like some keywords and dont add more extra, unnecessary information just synonym and very similar words
                    """
            },
            {
                "role": "user",
                "content": query
            }],
            "max_tokens": 300
        }
    )
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return query
    
def get_answer(user_query, top_5_chunks):
    headers = {
        'Content-Type': 'Application/json',
        'Authorization': f'Bearer {GROQ_API_KEY}'
    }

    response = requests.post(
        'https://api.groq.com/openai/v1/chat/completions',
        headers=headers, 
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{
                "role": "system",
                "content": f"""You are an AI language model. Based on this document, you need to generate a short answer to the user query.
            The answer should be a summary of the relevant information in the document that answers the user query. The answer should be
            concise and clear, and should be based on the information in the document. If you cannot find a relevant topic in the document, just
            say 'I don't have information about it.' The document sections are:
                    {top_5_chunks}
                    """
            },
            {
                "role": "user",
                "content": user_query
            }],
            "max_tokens": 600
        }
    )

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return response.text