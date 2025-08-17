# engine.py

import uuid
import time
import datetime as dt
import numpy as np
import ollama
import subprocess
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Tuple

# --- 1. INITIALIZE CORE COMPONENTS ---
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client(Settings(persist_directory="./chroma_store", is_persistent=True))
collection = client.get_or_create_collection(name="memory_store")


# --- 2. HELPER FUNCTIONS ---

def now_ts() -> float:
    return time.time()


def store_memory(entry_text: str, user_id: str, topic: str = "general"):
    emb = model.encode(entry_text).tolist()
    doc_id = str(uuid.uuid4())
    metadata = {"user_id": user_id.lower(), "timestamp": now_ts(), "topic": topic}
    collection.add(documents=[entry_text], embeddings=[emb], metadatas=[metadata], ids=[doc_id])
    print(f"Stored for {user_id.lower()}: {entry_text[:60]}...")


def prune_old_memories(user_id: str, days_to_keep: int = 90):
    if not isinstance(days_to_keep, int) or days_to_keep < 0: return
    cutoff_timestamp = time.time() - (days_to_keep * 24 * 60 * 60)
    old_memories = collection.get(
        where={"$and": [{"user_id": {"$eq": user_id.lower()}}, {"timestamp": {"$lt": cutoff_timestamp}}]}
    )
    ids_to_delete = old_memories.get("ids")
    if ids_to_delete:
        print(f"🧹 Pruning {len(ids_to_delete)} old memories for user '{user_id.lower()}'...")
        collection.delete(ids=ids_to_delete)


# --- RESTORED HELPER FUNCTIONS ---
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Calculates the cosine similarity between two vectors."""
    a, b = np.asarray(a), np.asarray(b)
    num = np.dot(a, b)
    den = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(num / den)


def rerank_results(query_emb: List[float], docs: List[str], metas: List[Dict[str, Any]], embs: List[List[float]]) -> \
List[Tuple[str, Dict, float]]:
    """Reranks results using a hybrid score of similarity and recency."""
    now = now_ts()
    ranked = []
    alpha = 0.7  # Weight for similarity
    decay_days = 14.0  # Recency decay period

    for d, m, e in zip(docs, metas, embs):
        sim = _cosine(np.asarray(query_emb), np.asarray(e))
        age_days = max(0.0, (now - float(m.get("timestamp", now))) / (60 * 60 * 24))
        rec = np.exp(-age_days / decay_days)
        score = alpha * sim + (1 - alpha) * rec
        ranked.append((d, m, score))

    ranked.sort(key=lambda x: x[2], reverse=True)
    return ranked


# --- END RESTORED FUNCTIONS ---

def format_chat_history(chat_history: List[Dict]) -> str:
    if not chat_history: return ""
    return "\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in chat_history[-4:]])


def query_local_llm(prompt: str, model_name: str = "mistral") -> str:
    ollama_path = r"C:\Users\crazz\AppData\Local\Programs\Ollama\ollama.exe"
    try:
        result = subprocess.run([ollama_path, "run", model_name], input=prompt.encode("utf-8"), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, check=True)
        return result.stdout.decode("utf-8").strip()
    except Exception as e:
        return f"⚠️ An unexpected error occurred: {e}"


# --- 3. THE FINAL "BRAIN" ---

def rag_chatbot(user_query: str, user_id: str, chat_history: List[Dict]) -> Tuple[str, list]:
    history_str = format_chat_history(chat_history)
    user_query_lower = user_query.lower()
    question_words = ["who", "what", "where", "when", "why", "how", "did", "do", "am i", "what's"]
    is_question = any(user_query_lower.startswith(word) for word in question_words) or user_query.endswith("?")

    if is_question:
        print("DEBUG: Intent is [question] by rule.")

        # --- NEW: Two-Step Retrieval Process ---
        # 1. Context Retrieval: Find keywords from chat history in long-term memory.
        context_query = history_str + "\n" + user_query
        q_emb = model.encode(context_query).tolist()

        res = collection.query(
            query_embeddings=[q_emb],
            n_results=10,
            where={"user_id": user_id.lower()},
            include=["documents", "metadatas", "embeddings"]
        )
        docs, metas, embs = res.get("documents", [[]])[0], res.get("metadatas", [[]])[0], res.get("embeddings", [[]])[0]

        ranked_memories = []
        if docs:
            ranked_memories = rerank_results(q_emb, docs, metas, embs)[:3]

        context = "\n".join([f"- {doc}" for doc, meta, score in
                             ranked_memories]) if ranked_memories else "No relevant long-term memories found."

        prompt = f"""You are a helpful AI assistant with both short-term conversational memory and a permanent long-term memory.
        Your task is to answer the user's question by synthesizing information from BOTH memory sources.

        --- CONVERSATION HISTORY (Short-Term Memory) ---
        {history_str}
        --- END HISTORY ---

        --- LONG-TERM MEMORIES ---
        {context}
        --- END MEMORIES ---

        Based on all the information above, answer this question: {user_query}
        Answer:"""

        response = query_local_llm(prompt)
        return response, ranked_memories

    else:  # If not a question, just chat
        print("DEBUG: Intent is [chat].")
        prompt = f"""You are a helpful AI assistant. Continue the conversation naturally.
        --- CONVERSATION HISTORY ---\n{history_str}\n--- END HISTORY ---
        User: {user_query}\nAssistant:"""
        response = query_local_llm(prompt)
        return response, []


# --- 4. DATABASE SEEDING ---
def populate_db_if_empty(user_id: str):
    """
    Populates the database with sample memories only if the user is 'siddharth'
    and their memory store is empty.
    """
    # First, check if the current user is 'siddharth'
    if user_id.lower() == 'siddharth':
        # Then, check if their memory store is empty
        if len(collection.get(where={"user_id": "siddharth"})['ids']) == 0:
            print(f"No memories found for {user_id.lower()}. Seeding database with sample memories...")
            seed_entries = [
                ("Booked train tickets to Jaipur for next weekend.", "travel"),
                ("I had dinner with Rahul and we discussed career plans.", "social"),
                ("Watched F1 qualifying highlights on YouTube.", "media")
            ]
            for txt, topic in seed_entries:
                store_memory(txt, user_id=user_id, topic=topic)


# --- 5. DELETE ALL MEMORIES ---
def delete_all_memories():
    """Deletes all memories for all users from the collection."""
    print("🧹 Deleting all memories for all users...")
    try:
        count = collection.count()
        if count == 0:
            print("✅ Memory store is already empty.")
            return 0

        # To delete all, we first get all IDs and then delete by ID.
        all_ids = collection.get(include=[])['ids']
        if all_ids:
            collection.delete(ids=all_ids)
            print(f"✅ Successfully deleted {len(all_ids)} memories.")
            return len(all_ids)
        else:
            print("✅ No memories found to delete.")
            return 0
    except Exception as e:
        print(f"An error occurred during deletion: {e}")
        return 0

# --- 6. SHOW ALL MEMORIES ---
def get_all_memories():
    """Retrieves all memories for all users, sorted by timestamp."""
    print("🔎 Retrieving all memories...")
    try:
        results = collection.get(include=["metadatas", "documents"])
        # Combine the lists into a single list of dictionaries
        all_memories = [
            {"id": i, "document": d, "metadata": m}
            for i, d, m in zip(results['ids'], results['documents'], results['metadatas'])
        ]
        # Sort memories by timestamp, newest first
        all_memories.sort(key=lambda x: x['metadata']['timestamp'], reverse=True)
        return all_memories
    except Exception as e:
        print(f"An error occurred while retrieving memories: {e}")
        return []