from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class Trial:
    def build_prompt(self):
        self.prompt = "You are generating code to invoke ETAS middleware APIs.\n\n"
        self.prompt += f"User Query:\n{self.query}\n\n"
        self.prompt += "Relevant Context:\n"

        for idx in self.I[0]:
            meta = self.metadata_store[idx]
            meta_str = ", ".join(
                f"{k}: {v}" for k, v in meta.items() if v is not None)
            self.prompt += f"[{meta_str}]\n{self.chunks[idx]}\n\n"

        self.prompt += "Task:\nProvide only the final code implementation based on the above context, without any explanation or formatting."

    def generate_code(self):
        model_name = "codellama/CodeLlama-7b-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        generator = pipeline(
            "text-generation", model=model, tokenizer=tokenizer)
        response = generator(self.prompt, max_new_tokens=200,
                             temperature=0.3, do_sample=True)

        # Remove the prompt from the generated text
        full_output = response[0]["generated_text"]
        generated_code = full_output[len(self.prompt):].strip()
        print("Generated Code:\n", generated_code)

    def run(self):
        print("Load embedding model")
        model = SentenceTransformer('all-MiniLM-L6-v2')

        print("Define chunks and metadata")
        self.chunks = [
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

        print("Generate embeddings")
        embeddings = model.encode(self.chunks)

        print("Create FAISS index")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))

        print("Store metadata in a parallel dictionary")
        self.metadata_store = {i: metadata[i] for i in range(len(metadata))}

        print("Accept a user query and embed it")
        # How do I initialize CAN communication?
        # self.query = input("Enter your query: ")
        self.query = "How do I initialize CAN communication?"
        query_embedding = model.encode([self.query])

        k = 2
        D, self.I = index.search(np.array(query_embedding), k)

        self.build_prompt()
        self.generate_code()
