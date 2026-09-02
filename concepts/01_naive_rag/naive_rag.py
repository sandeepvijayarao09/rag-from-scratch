"""Stage 1 (beginner): naive RAG. The original 60-line implementation.

This is the loop every RAG tutorial teaches, and it genuinely works:
embed the corpus, embed the query, rank by cosine, stuff the top-k into the
prompt, generate. Run it once to feel the shape of the thing.

What it hides is the subject of every stage after this one:
  - no chunking          readlines() only works on one-fact-per-line data
  - no evaluation        every change is unverifiable
  - dense-only           embeddings miss exact terms, IDs, rare words
  - no abstention        the prompt says "don't make things up" with no mechanism
  - re-embeds each run   150 serial HTTP calls per experiment
  - O(N) pure-python     recomputes every chunk's norm on every query

Stage 2 fixes the last two so the rest can be measured at all.
"""

import os

import ollama

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# 1. Load the dataset
dataset = []
with open('cat-facts.txt', 'r', encoding="utf8") as file:
    dataset = file.readlines()
    print(f'Loaded {len(dataset)} entries')

# 2. Implement the Vector Database
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

VECTOR_DB = []

def add_chunk_to_database(chunk):
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    VECTOR_DB.append((chunk, embedding))

print("Indexing dataset into vector database (this will take a moment)...")
for i, chunk in enumerate(dataset):
    add_chunk_to_database(chunk)

print("Database indexing complete.")

# 3. Implement the retrieval function
def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=3):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# 4. Chat generation phase
input_query = input('\nAsk me a question about cats: ')
retrieved_knowledge = retrieve(input_query)

print('\nRetrieved knowledge:')
for chunk, similarity in retrieved_knowledge:
    print(f' - (similarity: {similarity:.2f}) {chunk.strip()}')

# Format the context cleanly to avoid f-string syntax errors
contexts = "\n".join([f' - {chunk.strip()}' for chunk, similarity in retrieved_knowledge])
instruction_prompt = f"""You are a helpful chatbot. Use only the following pieces of context to answer the question. Don't make up any new information:
{contexts}"""

stream = ollama.chat(
    model=LANGUAGE_MODEL,
    messages=[
        {'role': 'system', 'content': instruction_prompt},
        {'role': 'user', 'content': input_query},
    ],
    stream=True,
)

print('\nChatbot response:')
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
print('\n')