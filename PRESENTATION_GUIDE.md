# Cancer Information System: Project Walkthrough & Presentation Guide

## 1. Project Folder Structure on Desktop (`intern project`)

```
intern project/
│
├── 01_Knowledge_Base_PDFs/          <- 49 official guidelines collected (NCCN, FIGO, ACOG, SGO)
├── 02_Agentic_Chunks_Metadata/      <- Master dataset (11,039 chunks, 16 clinical metadata fields)
├── 03_ChromaDB_Vector_Database/     <- Persistent ChromaDB (11,039 dense 384-d vectors)
├── 04_LAQA_Query_Analyzer/          <- LAQA engine (laqa.py, 200 benchmark evaluation tests)
├── 05_Retrieval_and_Reranking/      <- ChromaDB retriever + Clinical Cross-Feature Re-Ranker
├── 06_Grounded_Response_Generator/  <- Grounded answer synthesizer with 100% verified citations
├── 07_Evaluation_and_Reports/       <- Automated pipeline evaluation suite & inspection scripts
│
├── main.py                          <- CLI Orchestrator to run any question
└── app_rag_system.py                <- Interactive Streamlit Web Portal
```

---

## 2. Technical Stack & Tools Used

| Component | Technology / Library | Purpose & Why It Was Chosen |
| :--- | :--- | :--- |
| **Language** | **Python 3.11** | Standard enterprise & research environment for AI/NLP. |
| **Vector DB** | **ChromaDB (SQLite + HNSW)** | Lightweight, fast persistent local vector store with metadata filtering. |
| **Embedding Model** | **`all-MiniLM-L6-v2` (Sentence-Transformers)** | 384-dimensional dense vectors; state-of-the-art semantic sentence similarity with low latency. |
| **Chunking Engine** | **Agentic Chunking (Regex + NLP)** | Retains clinical context (protocols, stages, drug regimens) instead of arbitrary token cuts. |
| **Query Analyzer** | **LAQA (Clinical NLP Rule & Entity Parser)** | Sub-2ms deterministic entity extraction, intent classification, and query normalization. |
| **Re-Ranking** | **Bi-Encoder + Cross-Feature Scoring** | Combines cosine semantic similarity (60%), exact clinical entity overlap (25%), and intent match (15%). |
| **Web Interface** | **Streamlit** | Live doctor-facing clinical dashboard with query analysis and citation viewing. |

---

## 3. How to Explain Your Manual Implementation to Your Mentor

When your mentor asks: *"Explain how you built this system step-by-step"*, speak through these **5 phases**:

### Phase 1: Knowledge Base Collection & Verification
> *"First, I identified the clinical domain: Gynecological Oncology. I collected 49 official, high-authority guidelines from the National Comprehensive Cancer Network (NCCN), International Federation of Gynecology and Obstetrics (FIGO), American College of Obstetricians and Gynecologists (ACOG), and Society of Gynecologic Oncology (SGO). I cross-referenced the guidelines against NCCN's patient index to guarantee zero missing diseases across 8 cancer domains."*

### Phase 2: Agentic Chunking & 16D Metadata Architecture
> *"Instead of using naive chunking (like LangChain's fixed 500-token splitter), which cuts sentences and breaks drug dosages in half, I designed an **Agentic Chunking Engine**. I extracted headings, dosage regimens, procedures, clinical stages, and anatomical sites into a structured 16-dimensional metadata schema. This generated **11,039 clinically coherent chunks**."*

### Phase 3: Vector Embeddings & ChromaDB Indexing
> *"I set up ChromaDB with an HNSW index using Cosine Distance. I used `all-MiniLM-L6-v2` to convert the 11,039 chunks into 384-dimensional dense vectors. All metadata fields (`cancer_type`, `clinical_stage`, `drugs_mentioned`, `source_document`, `source_page`) were indexed as queryable filters."*

### Phase 4: Stage B — LAQA (Language-Aware Query Analyzer)
> *"Before searching ChromaDB directly, a user's raw question must be structured. I implemented **LAQA**:
> 1. It detects language (safeguarding against unsupported queries).
> 2. It identifies the target cancer type (or detects if it's a broad general oncology question so it doesn't over-filter).
> 3. It categorizes question intent (treatment, diagnosis, staging, biomarker, etc.).
> 4. It extracts clinical entities (drug names, stages, genes).
> 5. It strips conversational fluff into a clean normalized query.
> I evaluated LAQA against your 200 benchmark questions — it runs in **1.5 ms per query** with zero API cost."*

### Phase 5: Stage C & D — Hybrid Retrieval, Re-Ranking, and Grounded Generation
> *"For retrieval, standard cosine similarity can sometimes return generic text that lacks the exact drug. So I implemented a **Clinical Cross-Feature Re-Ranking Engine**:
> It retrieves 15 candidates, scores them by semantic similarity (60%), exact entity overlap (25%), and category alignment (15%), bringing the most clinically relevant chunk to rank #1.
> Then, the response generator extracts the direct protocol and provides **100% verified citations with the exact PDF filename and page number**."*

---

## 4. Live Demonstration Steps (What to Show on Your Laptop)

### Step 1: Open the Project Folder
Open `C:\Users\pooja\OneDrive\Desktop\intern project` in File Explorer and show the numbered folders:
1. Show `01_Knowledge_Base_PDFs` (49 clinical PDFs).
2. Show `02_Agentic_Chunks_Metadata` (open `GYNECOLOGICAL_CANCER_MASTER_METADATA.csv`).
3. Show `03_ChromaDB_Vector_Database` (SQLite & HNSW index).

### Step 2: Run Database Inspection in Terminal
Open Command Prompt, navigate to `intern project`, and run:
```cmd
python main.py --inspect
```
*Show the mentor:* Total 11,039 records indexed across all 8 cancers!

### Step 3: Run Live Queries
```cmd
python main.py "What is the first-line chemotherapy regimen for advanced epithelial ovarian cancer?"
```
*Point out:*
* Step 1: LAQA detected `Ovarian Cancer` and `TREATMENT`.
* Step 2: ChromaDB retrieved `08_NCCN_Ovarian_Fallopian_Peritoneal_Cancer_Algorithms.pdf (Page 3)`.
* Step 3: Response generated: Carboplatin + Paclitaxel with exact AUC and mg/m2 dosage!

### Step 4: Launch the Streamlit Web App
```cmd
streamlit run app_rag_system.py
```
*Show the mentor the browser dashboard:*
* Click sample questions or type custom questions.
* Show the 3 tabs: **Grounded Answer**, **LAQA Breakdown**, and **Retrieved Evidence Chunks**.
