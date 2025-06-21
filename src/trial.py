from sentence_transformers import SentenceTransformer
from chromadb import Client
from transformers import pipeline
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np
from optimum.onnxruntime import ORTModelForCausalLM

snippets = [
    "The soul is eternal, never born, and never dies. It is not destroyed when the body dies.",
    "Krishna is the Supreme Personality of Godhead, and devotional service to Him is the highest form of yoga.",
    "The material world is temporary and full of suffering, but surrendering to Krishna leads to liberation.",
    "Chanting the Hare Krishna mantra purifies the heart and awakens love of God.",
    "Real knowledge means understanding the difference between the body and the soul."
]


# Embedding model & DB
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma = Client().get_or_create_collection("prabhupada_wisdom")

# Index string snippets
for i, text in enumerate(snippets):
    embedding = embedder.encode(text).tolist()
    chroma.add(documents=[text], ids=[str(i)], embeddings=[embedding])

# Load tokenizer compatible with Mistral 7B (you may need to customize this path)
model = ORTModelForCausalLM.from_pretrained("ONNX/", provider="DmlExecutionProvider")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Pseudocode for user query
def answer_query(user_query):
    query_vector = embedder.encode(user_query).tolist()
    results = chroma.query(query_embeddings=[query_vector], n_results=3)
    context = " ".join(results["documents"][0])
    prompt = f"Context: {context}\n\nQ: {user_query}\nA:"

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=256, truncation=True)
    outputs = model.generate(**inputs)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    generated_answer = response_text.split("A:")[-1].strip()
    return generated_answer


# Example query
print(answer_query("What happens to the soul after death?"))