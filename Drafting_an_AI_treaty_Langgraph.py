import os
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import re

# === API key for OpenRouter ===
os.environ["OPENROUTER_API_KEY"] = "insert key here"

# === LLM setup ===
llm = ChatOpenAI(
    model="tngtech/deepseek-r1t-chimera:free",
    openai_api_key=os.environ["OPENROUTER_API_KEY"],
    openai_api_base="https://openrouter.ai/api/v1"
)

# ===Embedding model + Chroma vectorstore ===
embedding_model_name = "all-mpnet-base-v2"
k = 10

embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
vectordb = Chroma(
    persist_directory="egyptAI_rag_db",
    embedding_function=embeddings
)

#fetch relevant legal precedents when drafting clauses, grounding the model’s output
retriever = vectordb.as_retriever(search_kwargs={"k": k})

def chroma_retrieve(query):
    retrieved = retriever.invoke(query)

    cleaned_clauses = [
        re.sub(r"\s+", " ", str(doc.page_content)).strip()
        for doc in retrieved[:k]
    ]
    return cleaned_clauses


def call_llm(user_prompt):
    messages = [
        {"role": "system", "content": """You are a policy assistant specializing in drafting legally enforceable clauses for regulating the development and use of AI in Egypt, 
 aligned with Egyptian data protection law, Telecommunications Regulation Law and Anti-Cyber, Information Technology Crimes Law and other law mentioned in the context provided to you.

"""},
        {"role": "user", "content": user_prompt}
    ]
   # Call LLM
    response = llm.invoke(messages)

    # Convert to string, remove line breaks and extra whitespace
    cleaned = re.sub(r"\s+", " ", str(response)).strip()

    # Optional: trim preamble if model outputs things like "Clause: ..."
    if "Clause:" in cleaned:
        cleaned = cleaned.split("Clause:", 1)[-1].strip()

    return cleaned

from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, END

# Define what the shared state looks like
#optional fields (total=False)
class ClauseState(TypedDict, total=False):
    user_prompt: str
    draft_clause: str
    final_clause: str
    retrieved_clauses: List[str]
    ethics_flags: Union[str, List[str]] #can return a single text message or a structured list of issues depending on the clause
    review_decision: str




#Input: user_prompt
#Output: draft_clause
def clause_generator(state):
    prompt = state["user_prompt"]
    raw_clause = call_llm(f"Write a legally enforceable clause based on this idea: {prompt}. Respond with only the final clause. Make it short, clear, and enforceable.")
    clause = str(raw_clause).strip()
    return {"draft_clause": clause}

#Input: draft_clause
#Output: retrieved_clauses

def precedent_retriever(state):
    query = state["draft_clause"]
    results = chroma_retrieve(query)  # already embedded Charter + Masaar
    return {"retrieved_clauses": results}


#Input: draft_clause, retrieved_clauses
#Output: ethics_flags

def ethics_guardian(state):
    clause = state["draft_clause"]
    refs = state["retrieved_clauses"]
    flags = call_llm(
        f"""You are an international human rights advisor. Evaluate this clause and return any vague, ambiguous, non-binding, or problematic elements. Your job is to 
flag any risk of unenforceability or ethical loopholes\nClause: {clause}\nReferences: {refs}""")
    return {"ethics_flags": flags}

def get_user_decision():
    while True:
        decision = input("Do you want to approve or revise the clause? (approve/revise): ").strip().lower()
        if decision in ["approve", "revise"]:
            return decision
        print("Invalid input. Please type 'approve' or 'revise'.")


#showcase output so far to the human reviewer
def show_to_user(clause, flags, references):
    print("\nDraft Clause:")
    print(clause)
    print("\nEthics Flags:")
    if isinstance(flags, list):
        for f in flags:
            print(f"-", f)
    else:
        print(flags)
    print("\n Retrieved Precedents:")
    for ref in references:   #the references coming back from retriever may be in one of two forms: LangChain Document objects, Cleaned strings
        if hasattr(ref, "page_content"):
            print("- ", ref.page_content.strip())
        else:
            print("- ", str(ref).strip())

#Input: draft_clause, ethics_flags, retrieved_clauses
#Output: review_decision → either "approve" or "revise"
def human_review(state):
    show_to_user(state["draft_clause"], state["ethics_flags"], state["retrieved_clauses"])
    decision = get_user_decision()  # UI button click simulation
    return {"review_decision": decision}



#Condition: Only if review_decision == "revise"
#Input: draft_clause, ethics_flags, retrieved_clauses
#Output: final_clause

def legal_stylist(state):
    revision = call_llm(
        f"""You are a treaty editor trained on formal law legistlations. Rephrase this clause to address ethical/legal flags and to be to be specific, enforceable, 
and reflect international legal standards:\nClause: {state['draft_clause']}\nFlags: {state['ethics_flags']}\nReferences: {state['retrieved_clauses']}. Do not include commentary, justification, or headers. Output only the final polished clause."""
    )
    clause = str(revision).strip()

    # Optional: remove hallucinated Markdown or commentary
    if "Clause:" in clause:
        clause = clause.split("Clause:", 1)[-1].strip()
    
    return {"final_clause": clause}

def final_signoff(state):
    clause = state.get("final_clause", state["draft_clause"])  #if final_clause is not present (e.g. user clicked “Approve” without revision), it defaults to draft_clause.
    #log_to_storage({
        #"final_clause": clause,
        #"ethics_flags": state["ethics_flags"],
        #"retrieved_clauses": state["retrieved_clauses"],
        #"review_decision": state["review_decision"]
    #})
    return {"final_clause": clause}


builder = StateGraph(ClauseState)

from langchain_core.runnables import RunnableLambda

builder.add_node("ClauseGenerator", RunnableLambda(clause_generator))
builder.add_node("PrecedentRetriever", RunnableLambda(precedent_retriever))
builder.add_node("EthicsGuardian", RunnableLambda(ethics_guardian))
builder.add_node("HumanReview", RunnableLambda(human_review))
builder.add_node("LegalStylist", RunnableLambda(legal_stylist))
builder.add_node("FinalSignoff", RunnableLambda(final_signoff))

# Logic
builder.set_entry_point("ClauseGenerator")
builder.add_edge("ClauseGenerator", "PrecedentRetriever")
builder.add_edge("PrecedentRetriever", "EthicsGuardian")
builder.add_edge("EthicsGuardian", "HumanReview")


#  Routing logic after HumanReview
def check_if_revision_needed(state):
    decision = state.get("review_decision", "").lower()
    if decision == "revise":
        return "LegalStylist"
    else:
        return "FinalSignoff"

builder.add_conditional_edges("HumanReview", check_if_revision_needed)

# Paths
builder.add_edge("LegalStylist", "FinalSignoff")
builder.set_finish_point("FinalSignoff")
initial_state = {
    "user_prompt": input("Enter your clause idea: ")
}

graph = builder.compile()
def print_state(state):
    def clean(s):
        return re.sub(r"\s+", " ", str(s)).strip()

    print("\n USER PROMPT:")
    print(clean(state.get("user_prompt", "—")))

    print("\n DRAFT CLAUSE:")
    print(clean(state.get("draft_clause", "—")))

    print("\nRETRIEVED CLAUSES:")
    clauses = state.get("retrieved_clauses", [])
    if clauses:
        for i, clause in enumerate(clauses, 1):
            print(f"{i}. {clean(clause)}")
    else:
        print("—")

    print("\n ETHICS FLAGS:")
    flags = state.get("ethics_flags", [])
    if isinstance(flags, list):
        for flag in flags:
            print(f"- {clean(flag)}")
    else:
        print(clean(flags or "—"))

    print("\n REVIEW DECISION:")
    print(clean(state.get("review_decision", "—")))

    print("\nFINAL CLAUSE:")
    print(clean(state.get("final_clause", state.get("draft_clause", "—"))))

result = graph.invoke(initial_state)
print_state(result)

