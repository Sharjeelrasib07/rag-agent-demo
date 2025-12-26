# Agentic RAG Workspace

I designed and built this "Retrieval-Augmented Generation (RAG)" platform to bridge the gap between static documents and dynamic AI analysis. Unlike standard chatbots, I engineered this system with a Multi-Agent Architecture that allows me to switch the AI's behavior based on the specific task (e.g., acting as a Senior Coder vs. a Data Analyst).

## Key Features

- Agentic Personas: I implemented dynamic system prompting that allows the AI to switch roles (Coder, Analyst, Assistant) to process data differently depending on my needs.

- Knowledge Base: The system allows for uploading PDF or TXT files directly to a vector database (ChromaDB) for instant semantic search.

- Smart Routing: I added intelligent logic to detect casual greetings. If a user just says "Hello," the system bypasses the expensive RAG pipeline to reply naturally, saving resources.

- Persistent Memory: The application saves chat history and context using SQLite, ensuring the agent remembers previous turns in the conversation.

- High-Performance Inference: I utilized Groq’s LPU (Llama-3.3-70b) to ensure the token generation is near-instantaneous.

## Tech Stack

- Backend: Python, FastAPI
- AI Engine: Groq API (Llama-3.3)
- Vector Database: ChromaDB
- Frontend: Vanilla JS, HTML5, CSS3
- Deployment: Render (Backend) + Netlify (Frontend)

## How to Run Locally

1. Clone the repository
   git clone https://github.com/Sharjeelrasib07/rag-agent-demo.git
   cd rag-agent-demo

2. Install Dependencies
   pip install -r requirements.txt

3. Setup Environment
   Create a .env file and add your API key:
   GROQ_API_KEY=your_groq_api_key_here

4. Run the Server
   uvicorn main:app --reload

   Then open index.html in your browser to start.