from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

app = FastAPI()

# CORS configuration for your frontend running on localhost:5174
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],  # Allow frontend to make requests
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define the request model to receive the prompt and max_length
class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: int = 200  # Max length for text generation

@app.post("/generate_text")
async def generate_text(request: TextGenerationRequest):
    try:
        # Hugging Face Inference API URL for the model
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"  # Replace with your model name
        api_token = "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"  # Replace with your Hugging Face API token
        url = f"https://api-inference.huggingface.co/models/{model_name}"

        # Set the headers with your Hugging Face API token
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }

        # Define the payload with the prompt and other parameters
        payload = {
            "inputs": request.prompt,
            "parameters": {
                "max_length": request.max_length,
                "temperature": 0.7  # Controls creativity of the text (higher = more creative)
            }
        }

        # Send the POST request to Hugging Face Inference API
        response = requests.post(url, headers=headers, json=payload)

        # Check for successful response
        if response.status_code == 200:
            result = response.json()
            return {"text": result[0]['generated_text']}  # Return the generated text
        else:
            return {"error": f"Error: {response.status_code}, {response.text}"}

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "Hello, welcome to the Hugging Face API-powered backend!"}
























"""from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pydantic import BaseModel
import torch

app = FastAPI()

# Enable CORS for frontend running on localhost:5174
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],  # React frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load Mistral 7B
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

text_gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: int = 500  # Reduced for better response time

@app.post("/generate_text")
async def generate_text(request: TextGenerationRequest):
    try:
        result = text_gen_pipeline(request.prompt, max_length=request.max_length, do_sample=True, temperature=0.7)
        return {"text": result[0]['generated_text']}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "Hello World"}"""
