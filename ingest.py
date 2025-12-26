import os
import chromadb
from chromadb.utils import embedding_functions
from pydantic import BaseModel
from pypdf import PdfReader

# 1. Setup Database
print("⚙️  Connecting to database...")
chroma_client = chromadb.PersistentClient(path="rag_db")
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Reset to ensure we add the new metadata structure
try:
    chroma_client.delete_collection(name="knowledge_base")
    print("🗑️  Deleted old database to rebuild with Citations.")
except:
    pass

collection = chroma_client.create_collection(
    name="knowledge_base",
    embedding_function=embed_fn
)

# 2. Configuration
DOCS_FOLDER = "docs"

def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# 3. Process Files
if not os.path.exists(DOCS_FOLDER):
    os.makedirs(DOCS_FOLDER)
    print(f"📁 Created folder '{DOCS_FOLDER}'.")
    exit()

files = os.listdir(DOCS_FOLDER)
print(f"📂 Found {len(files)} files...")

all_chunks = []
all_ids = []
all_metadatas = []  # NEW: Store filenames here
id_counter = 0

for filename in files:
    file_path = os.path.join(DOCS_FOLDER, filename)
    
    if not os.path.isfile(file_path):
        continue

    print(f"📄 Processing: {filename}")
    
    content = ""
    if filename.endswith(".pdf"):
        content = read_pdf(file_path)
    elif filename.endswith(".txt"):
        content = read_txt(file_path)
    else:
        continue

    # Chunking
    chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
    
    for chunk in chunks:
        all_chunks.append(chunk)
        all_ids.append(f"doc_{id_counter}")
        # NEW: Save the filename for every chunk
        all_metadatas.append({"source": filename}) 
        id_counter += 1

# 4. Save to Database
if all_chunks:
    print(f"💾 Saving {len(all_chunks)} chunks with citations...")
    
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i : i + batch_size]
        batch_ids = all_ids[i : i + batch_size]
        batch_meta = all_metadatas[i : i + batch_size] # NEW
        
        collection.add(
            ids=batch_ids,
            documents=batch_chunks,
            metadatas=batch_meta # NEW
        )
    print("✅ Ingestion complete! Metadata saved.")
else:
    print("⚠️  No text found.")