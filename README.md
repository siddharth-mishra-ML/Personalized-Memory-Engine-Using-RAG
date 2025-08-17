# 🧠 Personalized Memory Engine (M.Tech Dissertation Project)

This repository contains the source code for the dissertation project, "Design and Development of a Personalized Memory Engine using Retrieval-Augmented Generation." It is a full-stack, local-first chatbot application that maintains a persistent, long-term memory for each user, ensuring 100% privacy and user control.

The system is built using a Retrieval-Augmented Generation (RAG) architecture, leveraging a local LLM (via Ollama) and a local vector database (ChromaDB) to create a conversational AI that can remember and recall user-specific information across sessions.

!(placeholder.png) 
*Note: Replace placeholder.png with a screenshot of your app.*

---
## ✨ Key Features

* **Permanent, Long-Term Memory**: Stores user-provided information in a persistent local vector database.
* **Conversational Context**: Maintains short-term memory of the current conversation to understand follow-up questions.
* **User-Controlled Storage**: An explicit "💾 Save to Memory" button gives the user full control over what information is permanently stored.
* **Private & Local-First**: The entire application, including the LLM, runs 100% locally. No user data ever leaves the machine.
* **Multi-User Support**: Creates a separate, private memory store for each user.
* **Advanced RAG Pipeline**:
    * Uses a hybrid reranking algorithm (semantic similarity + recency) to retrieve the most relevant memories.
    * Employs a rule-based intent system to reliably handle questions.
* **Memory Maintenance**: Automatically prunes old memories to keep the database relevant.
* **Interactive Web UI**: A user-friendly chat interface built with Streamlit, including admin controls to view or delete all memories.

---
## 🛠️ Technology Stack

* **Backend**: Python
* **Frontend**: Streamlit
* **LLM Runtime**: Ollama (with Mistral or Phi-3)
* **Vector Database**: ChromaDB
* **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)

---
## 🚀 How to Run

### **Prerequisites**
* Python 3.10+
* [Ollama](https://ollama.com/) installed and running.
* At least one model pulled via Ollama (e.g., Mistral).
    ```bash
    ollama pull mistral
    ```

### **Setup Instructions**

1.  **Clone the repository:**
    ```bash
    git clone [Your-GitHub-Repo-Link]
    cd Personalized-Memory-Engine
    ```

2.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    * Make sure your Ollama desktop application is running.
    * Execute the Streamlit app from your terminal:
        ```bash
        streamlit run app.py
        ```

4.  **Open your browser** to the local address provided by Streamlit (usually `http://localhost:8501`).

5.  **Start chatting!** Enter a username to begin a session. The first time a user named "siddharth" logs in, the database will be seeded with sample memories. All other users will start with a blank slate.
