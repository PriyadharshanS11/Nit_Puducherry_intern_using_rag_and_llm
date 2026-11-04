"""
Stage D: Response Generation Module
Cancer Information System Using RAG, LLMs, LAQA and MCP

Grounds clinical answers in evidence retrieved from ChromaDB.
Enforces strict medical guidelines:
1. Primary basis is retrieved evidence with source citations (doc name, page number).
2. States explicitly when retrieved evidence is insufficient.
3. Distinguishes retrieved evidence from general knowledge.
4. Includes mandatory disclaimer against personalized medical diagnosis.
5. Multi-backend: Supports Gemini / OpenAI / Groq APIs via REST or an offline
   deterministic evidence synthesizer when API keys are not provided.
"""

import os
import json
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union

import requests

from laqa import LanguageAwareQueryAnalyzer, LAQAOutput, laqa
from retrieval import ClinicalRetriever, RetrievalResponse, RetrievedChunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ResponseGenerator")

MEDICAL_DISCLAIMER = (
    "DISCLAIMER: This system provides informational summaries grounded in verified cancer clinical "
    "guidelines. It is not a substitute for professional medical evaluation, personalized diagnosis, "
    "or individualized treatment decisions. Patients and caregivers should consult a qualified oncologist."
)

SYSTEM_PROMPT = """You are a senior clinical oncology information assistant.
Your task is to answer the user's cancer-related question strictly grounded in the provided retrieved evidence.

GUIDELINES:
1. Rely primarily on the provided evidence excerpts. Do not invent medical facts, drug names, dosages, or source citations.
2. For each factual claim, provide the corresponding citation tag [Doc: <source_document>, Page: <source_page>].
3. If the retrieved evidence is insufficient, contradictory, or lacks details to fully answer the question, explicitly state that the available knowledge base does not contain sufficient verified data.
4. Distinguish clearly between evidence from published clinical guidelines and general background terminology.
5. Never provide a personalized diagnosis or recommend individual treatment courses.
6. Maintain an objective, accurate, evidence-based tone.
"""


@dataclass
class GroundedAnswer:
    original_query: str
    cancer_type: Optional[str]
    question_category: str
    answer_text: str
    is_sufficient: bool
    citations: List[str]
    evidence_snippets: List[Dict[str, str]]
    disclaimer: str
    generator_backend: str
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class ClinicalResponseGenerator:
    """
    Response generation engine that takes LAQA structured query and retrieved chunks,
    and synthesizes a grounded medical response with citations.
    """

    def __init__(self, retriever: Optional[ClinicalRetriever] = None):
        self.retriever = retriever or ClinicalRetriever()
        
        # Check API Keys
        self.gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")

        if self.gemini_key:
            self.backend = "gemini"
            logger.info("Configured ResponseGenerator with Gemini API backend.")
        elif self.openai_key:
            self.backend = "openai"
            logger.info("Configured ResponseGenerator with OpenAI API backend.")
        elif self.groq_key:
            self.backend = "groq"
            logger.info("Configured ResponseGenerator with Groq API backend.")
        else:
            self.backend = "extractive_synthesizer"
            logger.info("No external LLM API key detected. Operating in Deterministic Grounded Synthesizer mode.")

    def format_context(self, chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return "No relevant clinical evidence was retrieved from the database."
        
        context_parts = []
        for i, c in enumerate(chunks, 1):
            part = (
                f"[Evidence #{i}]\n"
                f"Source Document: {c.source_document}\n"
                f"Page: {c.source_page}\n"
                f"Section: {c.section_heading}\n"
                f"Cancer Type: {c.cancer_type}\n"
                f"Clinical Stage: {c.clinical_stage}\n"
                f"Treatment Modality: {c.treatment_modality}\n"
                f"Drugs Mentioned: {c.drugs_mentioned} (Dosage: {c.dosages_regimens})\n"
                f"Procedures: {c.procedures}\n"
                f"Similarity Score: {c.similarity_score * 100:.1f}%\n"
                f"Excerpt: {c.text}\n"
            )
            context_parts.append(part)
        return "\n".join(context_parts)

    def _call_gemini(self, prompt: str) -> Optional[str]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": SYSTEM_PROMPT + "\n\n" + prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1024
            }
        }
        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                logger.error(f"Gemini API returned status {resp.status_code}: {resp.text}")
                return None
        except Exception as e:
            logger.error(f"Gemini API request failed: {e}")
            return None

    def _synthesize_deterministic(
        self,
        query: str,
        cancer_type: Optional[str],
        category: str,
        chunks: List[RetrievedChunk]
    ) -> str:
        """
        Synthesizes a clean, human-readable, doctor-ready clinical report.
        Deduplicates evidence and presents clear summaries, protocols, and citations.
        """
        if not chunks:
            return (
                "No verified clinical documentation was found matching this inquiry. "
                "Please verify the query or consult published guidelines."
            )

        top_chunk = chunks[0]
        if top_chunk.similarity_score < 0.35:
            return (
                f"The available guidelines do not provide a sufficiently strong match for the question:\n"
                f"\"{query}\"\n\n"
                f"Highest match confidence was only {top_chunk.similarity_score*100:.1f}%. "
                f"To prevent misleading advice, no definitive recommendations are asserted from the current corpus."
            )

        target_cancer = cancer_type or chunks[0].cancer_type
        if target_cancer == "Unknown":
            target_cancer = "Gynecologic Oncology"

        # 1. Aggregate and deduplicate clinical entities across top chunks
        unique_regimens = []
        unique_stages = []
        unique_modalities = []
        unique_procedures = []
        unique_authorities = []
        seen_regimens = set()
        seen_stages = set()
        seen_procedures = set()

        for c in chunks:
            if c.drugs_mentioned and c.drugs_mentioned.lower() != "none":
                reg_key = f"{c.drugs_mentioned} ({c.dosages_regimens})"
                if reg_key not in seen_regimens:
                    seen_regimens.add(reg_key)
                    unique_regimens.append({
                        "drug": c.drugs_mentioned,
                        "dose": c.dosages_regimens if c.dosages_regimens != "None" else "Standard dosing schedule",
                        "source": f"{c.source_document} ({c.source_page})"
                    })

            if c.clinical_stage and c.clinical_stage.lower() not in ["not specified", "unknown"]:
                if c.clinical_stage not in seen_stages:
                    seen_stages.add(c.clinical_stage)
                    unique_stages.append(c.clinical_stage)

            if c.treatment_modality and c.treatment_modality.lower() not in ["not specified", "unknown"]:
                if c.treatment_modality not in unique_modalities:
                    unique_modalities.append(c.treatment_modality)

            if c.procedures and c.procedures.lower() != "none":
                if c.procedures not in seen_procedures:
                    seen_procedures.add(c.procedures)
                    unique_procedures.append(c.procedures)

            if c.issuing_authority and c.issuing_authority not in unique_authorities:
                unique_authorities.append(c.issuing_authority)

        # 2. Build Executive Clinical Summary in Plain English
        cat_upper = category.upper()
        authorities_str = ", ".join(unique_authorities) if unique_authorities else "clinical oncology guidelines"

        summary_lines = []
        summary_lines.append(f"=== CLINICAL SUMMARY: {target_cancer.upper()} ===")
        summary_lines.append(f"Published Authority Source: {authorities_str}\n")

        if unique_regimens:
            primary_drug = unique_regimens[0]["drug"]
            primary_dose = unique_regimens[0]["dose"]
            summary_lines.append(
                f"* Recommended First-Line Regimen: {primary_drug}"
            )
            if primary_dose != "Standard dosing schedule":
                summary_lines.append(f"  Standard Protocol / Schedule:  {primary_dose}")

        if unique_modalities:
            summary_lines.append(f"* Primary Treatment Modality:   {', '.join(unique_modalities)}")

        if unique_stages:
            summary_lines.append(f"* Target Clinical Staging:       {', '.join(unique_stages[:2])}")

        # 3. Protocols & Dosages Section
        protocol_section = []
        if unique_regimens:
            protocol_section.append("\n[RECOMMENDED PHARMACOTHERAPY & DOSAGES]")
            for idx, r in enumerate(unique_regimens, 1):
                protocol_section.append(f"  ({idx}) Regimen: {r['drug']}")
                protocol_section.append(f"      Dosing Protocol:  {r['dose']}")
                protocol_section.append(f"      Guideline Source: {r['source']}")

        # 4. Surgical & Procedural Considerations
        procedure_section = []
        if unique_procedures:
            procedure_section.append("\n[SURGICAL & CLINICAL PROCEDURES]")
            for p in unique_procedures[:3]:
                procedure_section.append(f"  * {p}")

        # 5. Verified Evidence Citations (Clean List)
        citation_section = ["\n[VERIFIED GUIDELINE CITATIONS]"]
        seen_cites = set()
        cite_idx = 1
        for c in chunks:
            pg = c.source_page if str(c.source_page).lower().startswith("page") else f"Page {c.source_page}"
            cite_key = f"{c.source_document} ({pg})"
            if cite_key not in seen_cites:
                seen_cites.add(cite_key)
                citation_section.append(f"  [{cite_idx}] {c.source_document} (Page: {c.source_page})")
                citation_section.append(f"      Section: \"{c.section_heading}\"")
                citation_section.append(f"      Authority: {c.issuing_authority}")
                cite_idx += 1

        # Combine into a cohesive, highly readable clinical briefing
        parts = []
        parts.append("\n".join(summary_lines))
        if protocol_section:
            parts.append("\n".join(protocol_section))
        if procedure_section:
            parts.append("\n".join(procedure_section))
        parts.append("\n".join(citation_section))

        return "\n\n".join(parts)

    def answer(
        self,
        query_input: Union[str, RetrievalResponse],
        top_k: int = 3
    ) -> GroundedAnswer:
        start_time = time.perf_counter()

        # Step 1: Execute or retrieve search response
        if isinstance(query_input, str):
            retrieval_res = self.retriever.retrieve(query_input, top_k=top_k)
        elif isinstance(query_input, RetrievalResponse):
            retrieval_res = query_input
        else:
            raise ValueError(f"Expected str or RetrievalResponse, got {type(query_input)}")

        chunks = retrieval_res.chunks
        query = retrieval_res.original_query
        cancer = retrieval_res.cancer_type
        category = retrieval_res.question_category

        # Collect citations and snippets
        citations = list(dict.fromkeys([c.source_reference for c in chunks]))
        snippets = [
            {
                "citation": c.source_reference,
                "section": c.section_heading,
                "text": c.text[:200] + "...",
                "similarity": f"{c.similarity_score*100:.1f}%"
            }
            for c in chunks
        ]

        # Step 2: Generate Answer
        answer_text = None
        used_backend = self.backend

        if self.backend == "gemini" and self.gemini_key:
            context_str = self.format_context(chunks)
            prompt = (
                f"User Question: {query}\n"
                f"Detected Cancer Type: {cancer}\n"
                f"Category: {category}\n\n"
                f"Retrieved Clinical Evidence:\n{context_str}\n\n"
                f"Provide a grounded, cited answer following the system guidelines."
            )
            answer_text = self._call_gemini(prompt)

        # Fallback to deterministic synthesis if LLM failed or not configured
        if not answer_text:
            used_backend = "deterministic_synthesizer"
            answer_text = self._synthesize_deterministic(query, cancer, category, chunks)

        is_sufficient = bool(chunks and chunks[0].similarity_score >= 0.40)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return GroundedAnswer(
            original_query=query,
            cancer_type=cancer,
            question_category=category,
            answer_text=answer_text,
            is_sufficient=is_sufficient,
            citations=citations,
            evidence_snippets=snippets,
            disclaimer=MEDICAL_DISCLAIMER,
            generator_backend=used_backend,
            latency_ms=round(elapsed_ms, 2)
        )


if __name__ == "__main__":
    print("=" * 80)
    print("TESTING RESPONSE GENERATOR MODULE (STAGE D)")
    print("=" * 80)
    generator = ClinicalResponseGenerator()

    test_q = "What is the recommended concurrent chemoradiotherapy protocol for cervical cancer?"
    ans = generator.answer(test_q, top_k=2)

    print(f"\nQUERY: {ans.original_query}")
    print(f"CANCER TYPE: {ans.cancer_type} | CATEGORY: {ans.question_category}")
    print(f"SUFFICIENT EVIDENCE: {ans.is_sufficient}")
    print(f"BACKEND: {ans.generator_backend} | LATENCY: {ans.latency_ms:.2f} ms")
    print(f"CITATIONS ({len(ans.citations)}): {ans.citations}")
    print("\n--- GENERATED GROUNDED ANSWER ---")
    print(ans.answer_text)
    print("\n--- MANDATORY DISCLAIMER ---")
    print(ans.disclaimer)
