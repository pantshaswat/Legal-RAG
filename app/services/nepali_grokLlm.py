import os
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

def expand_query(query):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {GROQ_API_KEY}'
    }
    response = requests.post(
        'https://api.groq.com/openai/v1/chat/completions',
        headers=headers,
        json={
            "model": "llama-3.3-70b-versatile",  # or whatever Groq Llama model you use
            "messages": [
                {
                    "role": "system",
                    "content": """तपाईं एक भाषा मोडेल हुनुहुन्छ। तपाईंको काम दिइएको प्रयोगकर्ता क्वेरीको पाँचवटा भिन्न संस्करणहरू उत्पादन गर्नु हो
डोकुमेन्ट फेच गर्दा राम्रो परिणाम ल्याउनका लागि। यी विस्तारहरू समानार्थी शब्दहरू वा समान भाव भएका शब्दहरू प्रयोग गरेर हुनुपर्छ।
कृपया क्वेरीलाई धेरै फरक नपार्नुहोस् — नयाँ कुराहरू नथप्नुहोस्, केवल समान शब्दहरू वा वाक्यांशहरू प्रयोग गर्नुहोस्।
परिणामहरूलाई अल्पविराम (comma) द्वारा अलग गर्नुहोस्।""",
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            "max_tokens": 300
        }
    )
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return query


def get_answer(user_query, top_5_chunks):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {GROQ_API_KEY}'
    }
    response = requests.post(
        'https://api.groq.com/openai/v1/chat/completions',
        headers=headers,
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {
                    "role": "system",
                    "content": f"""तपाईं एक भाषा मोडेल हुनुहुन्छ। तपाईंले तलका दस्तावेज खण्डहरूको आधारमा प्रयोगकर्ताको सोधाइको जवाफ दिनु पर्नेछ।
उत्तर संक्षिप्त, स्पष्ट र दस्तावेजमा आधारित हुनुपर्छ। यदि दस्तावेजमा सम्बन्धित जानकारी छैन भने, कृपया "यस विषयमा जानकारी छैन।" भनेर जवाफ दिनुहोस्।
दस्तावेज खण्डहरू:
{top_5_chunks}
""",
                },
                {
                    "role": "user",
                    "content": user_query
                }
            ],
            "max_tokens": 600
        }
    )
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return response.text
