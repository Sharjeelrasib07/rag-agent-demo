__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import shutil
import sqlite3
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from groq import Groq 
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader

# 1. Setup & Config
load_dotenv()
DOCS_FOLDER = "docs"
DB_FILE = "chat_history.db" 
os.makedirs(DOCS_FOLDER, exist_ok=True)

print("🚀 Starting RAG Agent Server...")

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("❌ ERROR: GROQ_API_KEY is missing in .env file!")
groq_client = Groq(api_key=api_key)

# Use the default lightweight embedding function from ChromaDB
embed_fn = embedding_functions.DefaultEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path="rag_db")
# Use get_or_create_collection so it doesn't crash on the first run!
collection = chroma_client.get_or_create_collection(name="knowledge_base", embedding_function=embed_fn)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    role: str = "Assistant"

# --- Database Helper Functions ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT, content TEXT)''')
    conn.commit()
    conn.close()

init_db()

def save_message(role, content):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO messages (role, content) VALUES (?, ?)', (role, content))
    conn.commit()
    conn.close()

def get_chat_history():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT role, content FROM messages ORDER BY id ASC')
    rows = cursor.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in rows]

# --- PDF Helper ---
def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# --- API Endpoints ---

@app.get("/documents")
def get_documents():
    data = collection.get()
    unique_files = set()
    if data["metadatas"]:
        for meta in data["metadatas"]:
            if "source" in meta: unique_files.add(meta["source"])
    return {"documents": list(unique_files)}

@app.get("/history")
def get_history():
    return get_chat_history()

@app.post("/clear_history")
def clear_history():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM messages')
    conn.commit()
    conn.close()
    return {"status": "cleared"}

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(DOCS_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        content = ""
        if file.filename.endswith(".pdf"): content = read_pdf(file_path)
        elif file.filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f: content = f.read()
        else: return {"status": "error", "message": "Unsupported file type"}

        chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
        ids = [f"{file.filename}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file.filename} for _ in range(len(chunks))]

        if chunks:
            collection.add(ids=ids, documents=chunks, metadatas=metadatas)
            return {"status": "success", "filename": file.filename}
        else: return {"status": "warning", "message": "File was empty"}

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/delete_doc")
def delete_document(filename: str = Form(...)):
    collection.delete(where={"source": filename})
    return {"status": "success", "message": f"Deleted {filename}"}

@app.post("/chat")
def chat_endpoint(request: QueryRequest):
    user_query = request.query
    selected_role = request.role
    
    print(f"💬 User ({selected_role}): {user_query}")
    save_message("user", user_query)
    
    # 1. Search Vector DB
    results = collection.query(query_texts=[user_query], n_results=3)
    
    sources = []
    # Only use document content if it exists
    if results['documents'] and results['documents'][0]:
        raw_context = "\n".join(results['documents'][0])
        retrieved_context = raw_context[:2500]
        for meta in results['metadatas'][0]:
            sources.append(meta['source'])
    else:
        retrieved_context = ""
        sources = []
    
    unique_sources = list(set(sources))
    
    # --- 2. INTELLIGENT ROUTING & PERSONA LOGIC ---
    # Set the personality based on the dropdown selection
    if selected_role == "Coder":
        base_prompt = "You are an Expert Software Engineer. You write clean, efficient code and explain technical concepts clearly."
    elif selected_role == "Analyst":
        base_prompt = "You are a Data Analyst. You summarize complex information into concise bullet points."
    else:
        base_prompt = "You are a friendly and professional AI Assistant."

    # THIS IS THE PART THAT FIXES YOUR ISSUE
    # We explicitly tell the AI to ignore the context for greetings.
    system_prompt = f"""
    {base_prompt}
    
    You have access to the user's uploaded documents (Context provided below).
    
    STRICT INSTRUCTIONS:
    1. **Greetings:** If the user says "Hello", "Hi", "How are you", or general chit-chat, DO NOT look at the Context. Just reply warmly and naturally (e.g., "Hello! How can I help you today?").
    2. **Document Questions:** If the user asks a specific question about the uploaded files, USE the Context to answer accurately.
    3. **Context Awareness:** Do not say "According to the context" unless you are actually answering a question about the documents.

    Context from uploaded files:
    {retrieved_context}
    """
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # 3. Add Recent Chat History
    full_history = get_chat_history()
    recent_msgs = full_history[-4:] # Context window of last 4 messages
    for msg in recent_msgs:
        messages.append({"role": msg['role'], "content": msg['content'][:500]})
    
    messages.append({"role": "user", "content": user_query})

    # 4. Generate Response
    try:
        chat_completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            stream=False 
        )
        
        bot_reply = chat_completion.choices[0].message.content
        
        # LOGIC TO HIDE SOURCES ON GREETINGS
        # We only append the source list if the user didn't just say "Hello"
        greeting_keywords = ["hello", "hi", "hey", "good morning", "good evening", "thanks", "thank you"]
        is_greeting = any(word in user_query.lower() for word in greeting_keywords) and len(user_query) < 20
        
        if unique_sources and not is_greeting:
            bot_reply += f"\n\n**📚 Sources:** {', '.join(unique_sources)}"
            
        save_message("assistant", bot_reply)
        return {"reply": bot_reply}

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"reply": f"System Error: {str(e)}"}