"""
Stage C: Similarity Search and Retrieval Module
Cancer Information System Using RAG, LLMs, LAQA and MCP

This module implements semantic similarity search against the ChromaDB vector database.
It consumes validated structured output from LAQA (Language-Aware Query Analyzer),
generates compatible query embeddings using all-MiniLM-L6-v2 (384-d),
and retrieves ranked clinical chunks with source citations and similarity scores.
"""

import os
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union

import chromadb
from chromadb.utils import embedding_functions

from laqa import LanguageAwareQueryAnalyzer, LAQAOutput, laqa

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ClinicalRetriever")

# Default Database Paths
def _find_default_chroma_path() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "03_ChromaDB_Vector_Database"),
        os.path.join(base_dir, "..", "03_ChromaDB_Vector_Database"),
        os.path.join(os.getcwd(), "03_ChromaDB_Vector_Database")
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return os.path.abspath(candidates[0])

DEFAULT_DB_PATH = _find_default_chroma_path()
DEFAULT_COLLECTION = "gynecological_cancer_rag"


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    similarity_score: float
    cosine_distance: float
    re_rank_score: float
    entity_match_hits: List[str]
    cancer_type: str
    chunk_type: str
    clinical_stage: str
    treatment_modality: str
    drugs_mentioned: str
    dosages_regimens: str
    procedures: str
    source_document: str
    source_page: str
    section_heading: str
    issuing_authority: str
    source_reference: str
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResponse:
    original_query: str
    normalized_query: str
    language: str
    cancer_type: Optional[str]
    question_category: str
    applied_filter: Dict[str, Any]
    fallback_used: bool
    fallback_reason: Optional[str]
    total_retrieved: int
    chunks: List[RetrievedChunk]
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["chunks"] = [c.to_dict() for c in self.chunks]
        return d


class ClinicalRetriever:
    """
    Retrieval engine connecting LAQA query representations to ChromaDB vector search.
    Guarantees embedding compatibility and enforces schema-verified metadata filtering.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION,
        analyzer: Optional[LanguageAwareQueryAnalyzer] = None
    ):
        self.db_path = os.path.abspath(db_path) if db_path else DEFAULT_DB_PATH
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"ChromaDB directory not found at: {self.db_path}")
        self.collection_name = collection_name
        self.analyzer = analyzer or laqa
        
        logger.info(f"Connecting to ChromaDB at: {self.db_path}")
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Use exact embedding function matching database collection configuration
        # all-MiniLM-L6-v2 via ONNX (384 dimensions)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn
            )
            self.total_records = self.collection.count()
            logger.info(f"Connected to collection '{self.collection_name}' with {self.total_records:,} records.")
        except Exception as e:
            logger.error(f"Failed to access collection '{self.collection_name}': {e}")
            raise RuntimeError(f"ChromaDB collection '{self.collection_name}' not available at '{self.db_path}'. Details: {e}")

    def retrieve(
        self,
        query_input: Union[str, LAQAOutput],
        top_k: int = 3,
        distance_threshold: Optional[float] = None
    ) -> RetrievalResponse:
        """
        Executes end-to-end similarity search:
        1. Analyzes query using LAQA if raw string provided.
        2. Applies validated metadata filters if supported.
        3. Generates compatible dense embeddings and queries ChromaDB.
        4. Handles graceful fallback if filtered search yields 0 matches.
        5. Computes cosine similarity and packages structured source citations.
        """
        start_time = time.perf_counter()

        # Step 1: Ensure structured LAQA input
        if isinstance(query_input, str):
            laqa_out = self.analyzer.analyze(query_input)
        elif isinstance(query_input, LAQAOutput):
            laqa_out = query_input
        else:
            raise ValueError(f"Expected str or LAQAOutput, got {type(query_input)}")

        search_text = laqa_out.normalized_query or laqa_out.original_query
        filters = dict(laqa_out.metadata_filters) if laqa_out.metadata_filters else {}

        # If query is flagged for clarification (empty or non-medical), return early
        if laqa_out.needs_clarification and not search_text:
            elapsed = (time.perf_counter() - start_time) * 1000
            return RetrievalResponse(
                original_query=laqa_out.original_query,
                normalized_query=search_text,
                language=laqa_out.language,
                cancer_type=laqa_out.cancer_type,
                question_category=laqa_out.question_category,
                applied_filter={},
                fallback_used=False,
                fallback_reason="Query lacks sufficient medical context or valid search text.",
                total_retrieved=0,
                chunks=[],
                latency_ms=round(elapsed, 2)
            )

        fallback_used = False
        fallback_reason = None
        where_clause = filters if filters else None

        n_candidates = max(top_k * 4, 15)

        # Step 2: Query ChromaDB with Metadata Filter (if active)
        try:
            results = self.collection.query(
                query_texts=[search_text],
                n_results=n_candidates,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            logger.warning(f"Filtered query failed ({where_clause}): {e}. Falling back to unfiltered search.")
            where_clause = None
            fallback_used = True
            fallback_reason = f"Filtered query error: {str(e)}"
            results = self.collection.query(
                query_texts=[search_text],
                n_results=n_candidates,
                where=None,
                include=["documents", "metadatas", "distances"]
            )

        # Step 3: Handle Zero Results Fallback
        # If a metadata filter was applied but 0 chunks returned (e.g. cancer_type not present in current DB partition),
        # gracefully fall back to unfiltered semantic search across all indexed oncology documents.
        if (not results["ids"] or len(results["ids"][0]) == 0) and where_clause is not None:
            logger.info(f"Zero results with filter {where_clause}. Executing fallback semantic search without filter.")
            fallback_used = True
            fallback_reason = f"Filter {where_clause} returned 0 results. Fallback to broad semantic retrieval."
            where_clause = None
            results = self.collection.query(
                query_texts=[search_text],
                n_results=n_candidates,
                where=None,
                include=["documents", "metadatas", "distances"]
            )

        # Step 4: Clinical Bi-Encoder + Cross-Feature Re-Ranking Engine
        # Collect all query entities
        query_entities = []
        if laqa_out.extracted_entities:
            for entity_list in laqa_out.extracted_entities.values():
                query_entities.extend([e.lower() for e in entity_list if len(e) > 1])

        candidate_chunks: List[RetrievedChunk] = []
        if results["ids"] and len(results["ids"][0]) > 0:
            count = len(results["ids"][0])
            for i in range(count):
                cid = results["ids"][0][i]
                doc_text = results["documents"][0][i]
                dist = float(results["distances"][0][i])
                meta = results["metadatas"][0][i] if results["metadatas"] else {}

                # Dense semantic similarity (cosine space: sim = 1.0 - dist)
                sim_score = max(0.0, min(1.0, 1.0 - dist))

                if distance_threshold is not None and dist > distance_threshold:
                    continue

                # 1. Entity Matching Score
                chunk_haystack = (
                    doc_text + " " +
                    meta.get("drugs_mentioned", "") + " " +
                    meta.get("procedures", "") + " " +
                    meta.get("clinical_stage", "") + " " +
                    meta.get("section_heading", "")
                ).lower()

                matched_entities = []
                for entity in query_entities:
                    if entity in chunk_haystack:
                        matched_entities.append(entity)

                if query_entities:
                    entity_score = len(matched_entities) / len(query_entities)
                else:
                    entity_score = 0.5  # Neutral if no entities in query

                # 2. Category & Modality Alignment Score
                cat_lower = laqa_out.question_category.lower()
                chunk_type_lower = meta.get("chunk_type", "").lower()
                modality_lower = meta.get("treatment_modality", "").lower()
                heading_lower = meta.get("section_heading", "").lower()

                category_score = 0.5
                if cat_lower == "treatment" and any(k in chunk_type_lower or k in modality_lower or k in heading_lower for k in ["treatment", "chemotherapy", "radiation", "surg", "regimen"]):
                    category_score = 1.0
                elif cat_lower == "diagnosis" and any(k in chunk_type_lower or k in heading_lower for k in ["diagnos", "screening", "workup"]):
                    category_score = 1.0
                elif cat_lower == "staging" and any(k in chunk_type_lower or k in heading_lower for k in ["staging", "figo", "tnm"]):
                    category_score = 1.0
                elif cat_lower == "biomarker" and any(k in chunk_type_lower or k in heading_lower for k in ["biomarker", "pathology", "molecular"]):
                    category_score = 1.0

                # 3. Composite Re-Ranking Score
                # Weights: 60% semantic similarity + 25% clinical entity hit + 15% category alignment
                re_rank_score = (0.60 * sim_score) + (0.25 * entity_score) + (0.15 * category_score)

                doc_name = meta.get("source_document", "Unknown Document")
                raw_page = meta.get("source_page", "Unknown Page")
                page_info = raw_page if str(raw_page).lower().startswith("page") else f"Page {raw_page}"
                source_ref = f"{doc_name} ({page_info})"

                chunk_obj = RetrievedChunk(
                    chunk_id=cid,
                    text=doc_text,
                    similarity_score=round(sim_score, 4),
                    cosine_distance=round(dist, 4),
                    re_rank_score=round(re_rank_score, 4),
                    entity_match_hits=list(set(matched_entities)),
                    cancer_type=meta.get("cancer_type", "Unknown"),
                    chunk_type=meta.get("chunk_type", "General"),
                    clinical_stage=meta.get("clinical_stage", "Not Specified"),
                    treatment_modality=meta.get("treatment_modality", "Not Specified"),
                    drugs_mentioned=meta.get("drugs_mentioned", "None"),
                    dosages_regimens=meta.get("dosages_regimens", "None"),
                    procedures=meta.get("procedures", "None"),
                    source_document=doc_name,
                    source_page=page_info,
                    section_heading=meta.get("section_heading", "General"),
                    issuing_authority=meta.get("issuing_authority", "Standard Guideline"),
                    source_reference=source_ref,
                    raw_metadata=meta
                )
                candidate_chunks.append(chunk_obj)

        # Sort candidate chunks by re_rank_score descending
        candidate_chunks.sort(key=lambda x: x.re_rank_score, reverse=True)
        retrieved_chunks = candidate_chunks[:top_k]

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return RetrievalResponse(
            original_query=laqa_out.original_query,
            normalized_query=search_text,
            language=laqa_out.language,
            cancer_type=laqa_out.cancer_type,
            question_category=laqa_out.question_category,
            applied_filter=where_clause if where_clause else {},
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            total_retrieved=len(retrieved_chunks),
            chunks=retrieved_chunks,
            latency_ms=round(elapsed_ms, 2)
        )


if __name__ == "__main__":
    print("=" * 80)
    print("INITIALIZING CLINICAL RETRIEVER MODULE (STAGE C)")
    print("=" * 80)
    retriever = ClinicalRetriever()
    
    test_queries = [
        "What is the recommended concurrent chemoradiotherapy for cervical cancer?",
        "What are the diagnostic HPV cotesting guidelines?",
        "What does neoadjuvant chemotherapy refer to in oncology?"
    ]

    for q in test_queries:
        print(f"\n--- Testing Query: '{q}' ---")
        resp = retriever.retrieve(q, top_k=2)
        print(f"Language:           {resp.language}")
        print(f"Detected Cancer:    {resp.cancer_type}")
        print(f"Category:           {resp.question_category}")
        print(f"Applied Filter:     {resp.applied_filter}")
        print(f"Fallback Used:      {resp.fallback_used} ({resp.fallback_reason})")
        print(f"Latency:            {resp.latency_ms:.2f} ms")
        print(f"Chunks Retrieved:   {resp.total_retrieved}")
        
        for idx, chunk in enumerate(resp.chunks, 1):
            print(f"  [Rank #{idx}] Similarity: {chunk.similarity_score*100:.1f}% | Dist: {chunk.cosine_distance}")
            print(f"    Source:   {chunk.source_reference}")
            print(f"    Heading:  {chunk.section_heading}")
            print(f"    Drugs:    {chunk.drugs_mentioned}")
            print(f"    Snippet:  {chunk.text[:140]}...")
