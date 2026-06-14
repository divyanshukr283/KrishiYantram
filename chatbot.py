from google import genai
import os
from dotenv import load_dotenv

# Securing authentication values
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = None
if api_key:
    client = genai.Client(api_key=api_key)
else:
    print("Warning: GEMINI_API_KEY not found.")
def ask_question(question: str) -> str:
    if not client:
     return "AI service is not available. Please check your API key."
    try:
        prompt_structure = f"""
        You are an agriculture expert helping farmers.

        Instructions:
        - Give simple and practical advice.
        - Keep the answer short and easy to understand.
        - Suggest affordable solutions whenever possible.
        - Focus on crops, fertilizers, irrigation, pests, and soil health.

        Question:
        {question}
        """
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt_structure
        )
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

def generate_farming_tips(crop: str, fertilizer: str, ph: float, soil_type: str) -> str:
    if not client:
        return ""
    try:
        prompt_structure = f"""
        You are an agriculture expert.

        A machine learning model recommended:

        Crop: {crop}
        Fertilizer: {fertilizer}
        Soil Type: {soil_type}
        Soil pH: {ph}

        Give 2 short farming tips related to:
        - Soil health
        - Nutrient management

        Keep the answer under 100 words and use simple language.
        """
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt_structure
        )
        return response.text
    except Exception:
        return ""