import getpass
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()

def setup_google_api():
    
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("gemini-api-key")


def get_query_expansion(user_query):
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    llm = ChatGoogleGenerativeAI(
        api_key=GOOGLE_API_KEY,
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
          timeout=60,  # Increase timeout
    max_retries=5,  # Increase retries
    retry_delay=10  # Add delay between retries
    )
    messages = [
        (
            "system",
            """You are an AI language model. Your task is to generate five different versions of the given user query
            to retrieve relevant documents from a vector database. Your goal is to help overcome limitations of distance similarity search
            by providing some really good query expansions. The query will be mostly law related. Provide these alternative queries
            separated by a new line and no numbering, just the queries. These are legal specific documents, so if you 
             diverse the query by more, the result might not be as expected, some other chunk might be fetched, so 
               the version should just contain synonyms or similar words to the original query
            and not necessarily be a question, just like some keywords and dont add more extra, unnecessary information just synonym and very similar words
            """,
        ),
        ("human", user_query),
    ]
    try:
        ai_msg = llm.invoke(messages)
        ai_gen_query_formatted = ai_msg.content.split("\n")
        ai_gen_query_formatted = user_query + ", " + ", ".join(ai_msg.content.split("\n"))
        return ai_gen_query_formatted
    except Exception as e:
        print(f"Error during query expansion: {e}")
        return user_query

def get_answer(user_query, top_5_chunks):
    """
    Generate an answer to the user query based on the top 5 most relevant chunks.
    
    Args:
        user_query (str): The user's question
        top_5_chunks (list): List of text chunks from the document
        
    Returns:
        str: Generated answer based on the document content
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=30,
        max_retries=2,
    )
    
    # Properly format chunks by joining the list of chunks with newlines and section separators
    formatted_chunks = "\n\nSection ---\n".join(top_5_chunks)
    
    messages = [
        (
            "system",
            f"""You are an AI language model. Based on this document, you need to generate a short answer to the user query.
            The answer should be a summary of the relevant information in the document that answers the user query. The answer should be
            concise and clear, and should be based on the information in the document. If you cannot find a relevant topic in the document, just
            say 'I don't have information about it.' The document sections are:

            {formatted_chunks}
            """
        ),
        ("human", user_query),
    ]
    
    try:
        ai_msg = llm.invoke(messages)
        return ai_msg.content
    except Exception as e:
        print(f"Error during answer generation: {e}")
        return "Sorry, I encountered an error while generating the answer."
