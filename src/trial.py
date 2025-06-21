import sys
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import time

# Set this switch to 'cpu' or 'cuda' to control AI workload device
USE_DEVICE = "cpu"  # Change to "cpu" or "cuda"

print("PyTorch version:", torch.__version__)
print("Torch CUDA available:", torch.cuda.is_available())
print("Torch CUDA version:", torch.version.cuda)
if USE_DEVICE == "cuda" and torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
    device = "cuda"
else:
    print("CUDA not available or CPU forced, using CPU.")
    device = "cpu"

start_time = time.time()

# Load embedding model on GPU if available
print("Load embedding model")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Define chunks and metadata
print("Define chunks and metadata")
chunks = [
    "void Can_Init(const Can_ConfigType* Config) { /* Initializes CAN controller */ }",
    "Can_ConfigType config = { .baudRate = 500000 }; Can_Init(&config);",
    "The Can_Init function initializes the CAN controller with the given configuration.",
    "add_subdirectory(middleware/can) to include CAN module in CMake."
]
metadata = [
    {"type": "implementation", "source": "can.c", "function": "Can_Init"},
    {"type": "usage", "source": "main.c", "function": "main"},
    {"type": "documentation", "source": "can_doc.md", "function": "Can_Init"},
    {"type": "config", "source": "CMakeLists.txt", "function": None}
]

# Generate embeddings
print("Generate embeddings")
embeddings = model.encode(chunks, device=device)

# Create FAISS index
print("Create FAISS index")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Store metadata in a parallel dictionary
print("Store metadata in a parallel dictionary")
metadata_store = {i: metadata[i] for i in range(len(metadata))}

# Accept a user query and embed it
print("Accept a user query and embed it")
query = "How do I initialize CAN communication?"
query_embedding = model.encode([query], device=device)

k = 2
D, I = index.search(np.array(query_embedding), k)

# Build prompt
prompt = "You are generating code to invoke ETAS middleware APIs.\n\n"
prompt += f"User Query:\n{query}\n\n"
prompt += "Relevant Context:\n"
for idx in I[0]:
    meta = metadata_store[idx]
    meta_str = ", ".join(f"{k}: {v}" for k, v in meta.items() if v is not None)
    prompt += f"[{meta_str}]\n{chunks[idx]}\n\n"
prompt += "Task:\nProvide only the final code implementation based on the above context, without any explanation or formatting."

# Generate code
model_name = "codellama/CodeLlama-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
codegen_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
generator = pipeline("text-generation", model=codegen_model, tokenizer=tokenizer, device=0 if device=="cuda" else -1)
response = generator(prompt, max_new_tokens=200, temperature=0.3, do_sample=True)

# Remove the prompt from the generated text
full_output = response[0]["generated_text"]
generated_code = full_output[len(prompt):].strip()
print("Generated Code:\n", generated_code)

end_time = time.time()
total_time_sec = end_time - start_time
print(f"Total execution time: {total_time_sec/60:.2f} minutes")
