# app.py (Streamlit + LangGraph + Human-in-the-Loop)

import streamlit as st
import re
import os
from typing import TypedDict, List, Union
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# === Embeddings + Vector DB ===
embedding_model_name = "all-mpnet-base-v2"
k = 10
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
vectordb = Chroma(persist_directory="egyptAI_rag_db", embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": k})

# === OpenRouter Key ===
os.environ["OPENROUTER_API_KEY"] = ""
llm = ChatOpenAI(
    model="tngtech/deepseek-r1t-chimera:free",
    openai_api_key=os.environ["OPENROUTER_API_KEY"],
    openai_api_base="https://openrouter.ai/api/v1"
)

# === Helper functions ===
def call_llm(prompt):
    messages = [
        {"role": "system", "content": """You are a policy assistant specializing in drafting legally enforceable clauses for regulating the development and use of AI in Egypt, 
 aligned with Egyptian data protection law, Telecommunications Regulation Law and Anti-Cyber, Information Technology Crimes Law and other law mentioned in the context provided to you.

"""},
        {"role": "user", "content": user_prompt}
    ]
    return re.sub(r"\s+", " ", str(llm.invoke(messages))).strip()

def chroma_retrieve(query):
    return [re.sub(r"\s+", " ", str(doc.page_content)).strip() for doc in retriever.invoke(query)[:k]]

# === LangGraph nodes ===
class ClauseState(TypedDict, total=False):
    user_prompt: str
    draft_clause: str
    final_clause: str
    retrieved_clauses: List[str]
    ethics_flags: Union[str, List[str]]
    review_decision: str

def clause_generator(state):
    clause = call_llm(f"Write a legally enforceable clause: {state['user_prompt']}. Respond with only the clause.")
    return {"draft_clause": clause}

def precedent_retriever(state):
    return {"retrieved_clauses": chroma_retrieve(state["draft_clause"])}

def ethics_guardian(state):
    clause = state["draft_clause"]
    refs = state["retrieved_clauses"]
    flags = call_llm(f"Flag vague/unethical issues in this clause.\nClause: {clause}\nReferences: {refs}")
    return {"ethics_flags": flags}

def legal_stylist(state):
    revised = call_llm(f"Rephrase this clause to fix issues:\nClause: {state['draft_clause']}\nFlags: {state['ethics_flags']}\nReferences: {state['retrieved_clauses']}. Output only the final clause.")
    return {"final_clause": revised.strip()}

def final_signoff(state):
    return {"final_clause": state.get("final_clause", state["draft_clause"])}

# === Streamlit UI ===
st.set_page_config(page_title="AI Clause Assistant", layout="wide")
st.title("🧾 AI Clause Assistant - Interactive LangGraph Demo")

if "state" not in st.session_state:
    st.session_state.state = {}

if "phase" not in st.session_state:
    st.session_state.phase = "input"

# === PHASE 1: Prompt input and partial graph ===
if st.session_state.phase == "input":
    user_input = st.text_area("💬 Enter your clause idea:", "I want to prevent algorithmic bias in public services.")
    if st.button("Generate Draft Clause"):
        # Run partial graph
        partial_state = {"user_prompt": user_input}
        partial_state.update(clause_generator(partial_state))
        partial_state.update(precedent_retriever(partial_state))
        partial_state.update(ethics_guardian(partial_state))
        st.session_state.state = partial_state
        st.session_state.phase = "review"
        st.experimental_rerun()

# === PHASE 2: Human Review ===
elif st.session_state.phase == "review":
    state = st.session_state.state
    st.subheader("Draft Clause")
    st.code(state["draft_clause"])

    with st.expander("Retrieved Precedents"):
        for i, clause in enumerate(state["retrieved_clauses"], 1):
            st.success(f"Precedent {i}: {clause}")

    st.subheader(" Ethics & Enforceability Flags")
    st.warning(state["ethics_flags"])

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Approve Clause"):
            state["review_decision"] = "approve"
            state.update(final_signoff(state))
            st.session_state.phase = "final"
            st.experimental_rerun()

    with col2:
        if st.button("🔁 Revise Clause"):
            state["review_decision"] = "revise"
            state.update(legal_stylist(state))
            state.update(final_signoff(state))
            st.session_state.phase = "final"
            st.experimental_rerun()

# === PHASE 3: Final Clause ===
elif st.session_state.phase == "final":
    st.success("Clause workflow complete.")
    st.subheader("🖋Final Clause")
    st.code(st.session_state.state["final_clause"])

    st.download_button("Download Clause as Text", st.session_state.state["final_clause"], "final_clause.txt")

    if st.button("Start Over"):
        st.session_state.phase = "input"
        st.session_state.state = {}

