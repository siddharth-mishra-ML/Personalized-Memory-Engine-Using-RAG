# app.py

import streamlit as st
from engine import (
    rag_chatbot,
    populate_db_if_empty,
    prune_old_memories,
    store_memory,
    delete_all_memories,
    get_all_memories
)
import datetime as dt

st.set_page_config(page_title="Personalized Memory Engine", page_icon="🧠")
st.title("🧠 Personalized Memory Engine")

# --- USER LOGIN AND SESSION STATE ---
if "user_id" not in st.session_state:
    st.session_state.user_id = None
    st.session_state.messages = []
    st.session_state.view_all_memories = False

if st.session_state.user_id is None:
    st.info("Welcome! This is a chatbot with a persistent, personalized memory.")
    with st.form("login_form"):
        username = st.text_input("Enter your username to begin").lower()
        submitted = st.form_submit_button("Start Chatting")
        if submitted and username:
            st.session_state.user_id = username
            prune_old_memories(username)
            populate_db_if_empty(username)
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Hello, {username}! How can I help you remember?"})
            st.rerun()

# --- MAIN APP LOGIC ---
else:
    # --- SIDEBAR ---
    st.sidebar.header(f"Logged in as: **{st.session_state.user_id}**")
    if st.sidebar.button("Logout"):
        st.session_state.user_id = None
        st.session_state.messages = []
        st.session_state.view_all_memories = False
        st.rerun()

    st.sidebar.divider()
    st.sidebar.header("📖 Memory Controls")
    if st.sidebar.button("View All Memories"):
        st.session_state.view_all_memories = True
        st.rerun()

    st.sidebar.divider()
    st.sidebar.header("⚠️ Admin Controls")
    if st.sidebar.button("🔥 Delete All Memories"):
        st.session_state.confirm_delete = True

    if st.session_state.get("confirm_delete", False):
        st.sidebar.warning("Are you absolutely sure? This will delete all memories for all users.")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Yes, Delete"):
                deleted_count = delete_all_memories()
                st.success(f"✅ Successfully deleted {deleted_count} memories.")
                st.session_state.confirm_delete = False
                st.rerun()
        with col2:
            if st.button("Cancel"):
                st.session_state.confirm_delete = False
                st.rerun()

    # --- MAIN WINDOW DISPLAY ---
    if st.session_state.view_all_memories:
        st.header("All Stored Memories")
        if st.button("⬅️ Back to Chat"):
            st.session_state.view_all_memories = False
            st.rerun()

        all_memories = get_all_memories()
        if not all_memories:
            st.info("The memory store is currently empty.")
        else:
            with st.expander(f"Found {len(all_memories)} memories", expanded=True):
                for mem in all_memories:
                    ts = dt.datetime.fromtimestamp(mem['metadata']['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                    st.markdown(
                        f"**User:** `{mem['metadata']['user_id']}`\n\n**Memory:** `{mem['document']}`\n\n*Saved on: {ts}*")
                    st.divider()
    else:
        # --- CHAT INTERFACE ---
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "user":
                    button_key = f"save_{i}"
                    if st.button("💾 Save to Memory", key=button_key, help="Save this message to your permanent memory"):
                        with st.spinner("Saving..."):
                            store_memory(message["content"], user_id=st.session_state.user_id)
                            st.success(f"Saved '{message['content'][:30]}...' to your long-term memory!")

        if prompt := st.chat_input("Ask a question or make a statement..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response, ranked_memories = rag_chatbot(prompt, user_id=st.session_state.user_id,
                                                            chat_history=st.session_state.messages)
                    if ranked_memories:
                        with st.expander("Show Retrieved Long-Term Memories"):
                            for j, (doc, meta, score) in enumerate(ranked_memories):
                                st.write(f"**Memory {j + 1} (Score: {score:.2f}):** {doc}")
                    st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()