from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import re

# === Load PDFs ===
charter_loader = PyPDFLoader("EgyptianCharterForResponsibleAIEnglish-v1.0.pdf")
masaar_loader = PyPDFLoader("Regulating Artificial Intelligence in Egypt_ Proposed Standards and Principles.pdf")

charter_docs = charter_loader.load()
masaar_docs = masaar_loader.load()

# === Chunking function for masaar ===

#Chunk the masaar PDF based on section titles

#merge the PDF into a single searchable text string
full_text_masaar = "\n".join([doc.page_content for doc in masaar_docs])


#encode the hierarchy structure into the system
masaar_section_titles = [ 
"Introduction",
"The Legislative Environment",
 "The Legislative Environment>The Relevant Local Laws in Force", 
"The Legislative Environment>Shortcomings and Deficiencies",  
"Regulatory Environment>Existing Institutions and Authorities", 
"Regulatory Environment>Knowledge Environment", 
"Regulatory Environment>International Legislative Experiences", 
"Regulatory Environment>Contributions of International Institutions",
 "Regulatory Environment>Civil Society Contributions",
 "Philosophy and Approaches of the Law>Philosophy and Vision", 
"Philosophy and Approaches of the Law>Objectives",
 "Philosophy and Approaches of the Law>Approaches",
 "Philosophy and Approaches of the Law>Risk-Based Approach: Graduated Requirements and Obligations",
 "Philosophy and Approaches of the Law>Conditional Flexibility in Implementation", 
 "Philosophy and Approaches of the Law>The Adaptability Approach",
 "General Standards and Fundamental Principles>General Standards",
  "General Standards and Fundamental Principles>Language",
  "General Standards and Fundamental Principles >Comprehensiveness",
 "General Standards and Fundamental Principles>Limits of Referral to the Executive Regulations",
  "General Standards and Fundamental Principles>Limits on Power Delegation to Executive Entities",
  "General Standards and Fundamental Principles>Key Principles for Ethical AI Governance",
 "General Standards and Fundamental Principles >Criteria for Evaluating the Provisions of the Law",
  "General Provisions>Legal Jurisdiction and Scope of Application",
 "General Provisions>Key Definitions",
 "General Provisions>Classification of AI Practices and Systems",
 "General Provisions>Classification Criteria",
 "General Provisions>Prohibited Practices",
 "General Provisions>High-Risk Systems",
 "Requirements for Approving AI Systems>Compliance Safeguards",
 "Requirements for Approving AI Systems>Crisis Management Systems",
 "Requirements for Approving AI Systems>Data Governance",
 "Requirements for Approving AI Systems>Documentation and Record-Keeping",
 "Requirements for Approving AI Systems>Transparency Requirements",
 "Requirements for Approving AI Systems>Human Oversight",
 "Requirements for Approving AI Systems>Accuracy, Quality, and Cybersecurity",
 "Requirements for Approving AI Systems>Obligations of High-Risk System Service Providers",
 "Requirements for Approving AI Systems>Impact Assessment on Fundamental Rights",
 "Requirements for Approving AI Systems>Requirements for Systems with Special Characteristics",
 "Regulatory Procedures>Requirements for Systems with Special Characteristics",
 "Regulatory Procedures>Procedures of Oversight, Supervision, and Compliance with Law Requirements Assessment",
 "Regulatory Procedures>Investigation Procedures and Administrative Penalties",
 "Regulatory Procedures>Criminalization Cases and Regulation of Criminal Penalties",
 "Regulatory Procedures>Harm Assessment, Redress, and Compensation Mechanisms",
 "Regulatory Procedures>Procedures to Support Development, Innovation, and Testing Capabilities",
 "Regulatory Procedures>Procedures for Supporting International Cooperation",
"Regulatory Procedures>Procedures for Regulating the Use of AI Systems by Law Enforcement and Security Agencies",
 "Regulatory, Oversight, and Advisory Bodies>The Competent Authority: Purpose and Role",
 "Regulatory, Oversight, and Advisory Bodies>Functions, Powers, and Scope of the Competent Authority",
 "Regulatory, Oversight, and Advisory Bodies>Structure and Composition of the Competent Authority",
 "Regulatory, Oversight, and Advisory Bodies>Requirements for Independence, Transparency, and Accountability",
 "Regulatory, Oversight, and Advisory Bodies>Role of Advisory Bodies and Procedures for Accreditation and Registration"
]



def build_chunks_by_section(full_text, masaar_section_titles):
    section_docs = []

    # === Step 1: Extract start positions of each title in the text then store them ordered ===
    positions = []
    for title in masaar_section_titles:
        pattern = re.escape(title)
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            positions.append((match.start(), title))  # (position, raw_title)

    positions.sort()  # Ensure section order by start position in text

    # === Step 2: Slice between positions and build Document objects ===
    #Each section begins at its heading & Ends right before the next heading to preserve narrative coherence

    for i in range(len(positions)):
        start_pos, raw_title = positions[i]
        end_pos = positions[i+1][0] if i+1 < len(positions) else len(full_text)
        section_text = full_text[start_pos:end_pos].strip()

        if not section_text:
            continue  # Skip empty content

        # === Step 3: Clean and parse title hierarchy ===
        #preserved section hierarchy in metadata to support context-aware retrieval later
        title_parts = [t.strip() for t in raw_title.split(">")]
        section_title = title_parts[-1]  # Last part is the specific title
        parent_section = title_parts[0] if len(title_parts) > 1 else None
        full_path = " > ".join(title_parts)

        # === Step 4: Create Document with expanded metadata ===
        metadata = {
            "section_title": section_title,
            "parent_section": parent_section,
            "full_path": full_path,
            "source": "masaar"
        }

        doc = Document(page_content=section_text, metadata=metadata)
        section_docs.append(doc)

    return section_docs

masaar_docs_final= build_chunks_by_section(full_text_masaar, masaar_section_titles)

# === Chunking function for charter ===

#step 1:merge the PDF into a single searchable text string

full_text_charter = "\n".join([doc.page_content for doc in charter_docs])

charter_section_titles = [
    "Introduction and Background",
    "Document Scope",
    "General Guidelines",
    "Implementation Guidelines"
]

principles = [
    "Fairness",
    "Transparency and Explainability",
    "Accountability",
    "Security and Safety",
    "Human-Centeredness"
]


def tag_principle_in_chunks(docs):
    principle_pattern = re.compile(r"\[(Accountability|Fairness|Transparency and Explainability|Security and Safety|Human-Centeredness)\]", re.IGNORECASE)

    for doc in docs:
        match = principle_pattern.search(doc.page_content)
        if match:
            principle = match.group(1).capitalize()
            doc.metadata["principle"] = principle
        else:
            doc.metadata["principle"] = "Unspecified"  # fallback tag if no match found

    return docs

def build_charter_chunks_with_metadata(full_text, section_titles):
    section_docs = []

    # Step 1: Find all section title positions in text
    positions = []
    for title in section_titles:
        pattern = re.escape(title)
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            positions.append((match.start(), title))

    # Step 2: Sort by appearance in document
    positions.sort()

    # Step 3: Extract text for each section
    for i in range(len(positions)):
        start_pos, title = positions[i]
        end_pos = positions[i+1][0] if i+1 < len(positions) else len(full_text)
        section_text = full_text[start_pos:end_pos].strip()

        # Metadata enrichment
        if section_text:
            doc = Document(
                page_content=section_text,
                metadata={
                    "section_title": title,
                    "parent_section": title,         # No hierarchy in this case, same as title
                    "full_path": title,              # Full path = single level
                    "source": "charter",
                    "title": "Egyptian Charter for Responsible AI"
                }
            )
            section_docs.append(doc)

    return section_docs


charter_docs_1=build_charter_chunks_with_metadata(full_text_charter, charter_section_titles)

charter_docs_final= tag_principle_in_chunks (charter_docs_1)

#Created a unified corpus with consistent metadata schemas

all_docs = charter_docs_final + masaar_docs_final


# === Chunking ===

#custom semantic chunks might be too long
#secondary chunking to normalise length
#preserved semantic integrity first, then applied size-normalisation for embedding quality.

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = splitter.split_documents(all_docs)
#To check chunk size
for doc in all_docs[:3]: 
    print("---")
    print("Content:", doc.page_content[:200], "...")
    print("Metadata:", doc.metadata)



# === Choose your embedding model ===

#Best general-purpose semantic mode

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


# === Create the vector store ===
db = Chroma.from_documents(split_docs, embedding_model, persist_directory="egyptAI_rag_db")
db.persist()

print("Data embedded and saved to ChromaDB!")
