"""
Stage E: Multi-Agent System
Cancer Information System Using RAG, LLMs, LAQA and MCP

Architecture:
  User Question
      │
      ▼
  ┌─────────────┐
  │ QueryAgent  │  ← Language-Aware Query Understanding (wraps LAQA)
  └──────┬──────┘
         │ structured LAQAOutput
         ▼
  ┌──────────────────┐
  │ RetrievalAgent   │  ← Dense Retrieval + Clinical Re-Ranking (wraps ClinicalRetriever)
  └────────┬─────────┘
           │ ranked evidence chunks
           ▼
  ┌─────────────────────┐
  │  FactCheckAgent     │  ← Citation Verification + Grounding Confidence Score
  └──────────┬──────────┘
             │ verified, scored evidence
             ▼
  ┌──────────────────────┐
  │   ResponseAgent      │  ← Grounded Answer Synthesis with verified citations
  └──────────┬───────────┘
             │ GroundedAnswer
             ▼
        Final Output

Each agent communicates via a typed AgentMessage (structured JSON payload).
No external orchestration framework is required — this is pure Python.
"""

import os
import sys
import json
import time
import logging
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

# ── Project root on sys.path so subfolders can also run this ──────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from laqa import LanguageAwareQueryAnalyzer, LAQAOutput, laqa
from retrieval import ClinicalRetriever, RetrievalResponse, RetrievedChunk
from response_generator import ClinicalResponseGenerator, GroundedAnswer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MultiAgentSystem")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  TYPED MESSAGE ENVELOPE
#     All inter-agent communication uses this typed envelope.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentMessage:
    """
    Typed envelope for inter-agent communication.

    Fields
    ------
    sender       : Name of the sending agent
    receiver     : Name of the receiving agent
    message_type : One of  QUERY | RETRIEVAL_REQUEST | RETRIEVAL_RESULT
                           FACT_CHECK_REQUEST | FACT_CHECK_RESULT
                           RESPONSE_REQUEST   | FINAL_RESPONSE | ERROR
    payload      : Arbitrary typed data (dict, dataclass, str …)
    timestamp_ms : Unix timestamp in milliseconds (auto-set)
    trace_id     : Shared request trace UUID (auto-set when empty)
    """
    sender: str
    receiver: str
    message_type: str
    payload: Any
    timestamp_ms: float = field(default_factory=lambda: time.time() * 1000)
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # payload may contain dataclass objects; convert them
        if hasattr(d["payload"], "to_dict"):
            d["payload"] = d["payload"].to_dict()
        return d

    def to_json(self, indent: int = 2) -> str:
        try:
            return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)
        except Exception:
            return json.dumps({"sender": self.sender, "receiver": self.receiver,
                               "message_type": self.message_type, "trace_id": self.trace_id},
                              indent=indent)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FACT-CHECK RESULT  (custom dataclass for FactCheckAgent output)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FactCheckResult:
    """
    Per-chunk verification output produced by FactCheckAgent.

    grounding_confidence  : 0.0–1.0 composite score
                            = 0.6 × similarity + 0.25 × citation_completeness
                              + 0.15 × source_authority_score
    citation_complete     : True if both source_document AND source_page are present
    authority_verified    : True if issuing_authority is a known oncology body
    flag                  : One of  VERIFIED | LOW_CONFIDENCE | MISSING_CITATION
    """
    chunk_id: str
    source_document: str
    source_page: str
    section_heading: str
    similarity_score: float
    citation_complete: bool
    authority_verified: bool
    authority_name: str
    grounding_confidence: float   # 0.0 – 1.0
    flag: str                     # VERIFIED | LOW_CONFIDENCE | MISSING_CITATION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FactCheckBundle:
    """
    Aggregated output of FactCheckAgent for a single query.
    """
    trace_id: str
    total_chunks_checked: int
    verified_count: int
    low_confidence_count: int
    missing_citation_count: int
    overall_confidence: float          # mean grounding_confidence across all chunks
    verified_chunks: List[RetrievedChunk]
    fact_check_results: List[FactCheckResult]
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["verified_chunks"] = [c.to_dict() if hasattr(c, "to_dict") else str(c)
                                 for c in self.verified_chunks]
        d["fact_check_results"] = [r.to_dict() for r in self.fact_check_results]
        return d


# ══════════════════════════════════════════════════════════════════════════════
# 3.  BASE AGENT
# ══════════════════════════════════════════════════════════════════════════════

class BaseAgent:
    """
    Abstract base for all agents.
    Subclasses must implement `handle(message: AgentMessage) -> AgentMessage`.
    """

    KNOWN_AUTHORITIES = {
        "NCCN", "FIGO", "ACOG", "SGO", "ESMO", "ASCO", "WHO",
        "National Comprehensive Cancer Network",
        "Society of Gynecologic Oncology",
        "American College of Obstetricians and Gynecologists",
        "International Federation of Gynecology and Obstetrics",
        "European Society for Medical Oncology",
        "American Society of Clinical Oncology",
    }

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"Agent.{name}")

    def handle(self, message: AgentMessage) -> AgentMessage:
        raise NotImplementedError

    def _make_error(self, trace_id: str, receiver: str, reason: str) -> AgentMessage:
        return AgentMessage(
            sender=self.name,
            receiver=receiver,
            message_type="ERROR",
            payload={"error": reason},
            trace_id=trace_id
        )


# ══════════════════════════════════════════════════════════════════════════════
# 4.  QUERY AGENT
#     Wraps LAQA. Accepts a raw string; emits a structured LAQAOutput.
# ══════════════════════════════════════════════════════════════════════════════

class QueryAgent(BaseAgent):
    """
    Agent 1 — Query Understanding

    Responsibility
    --------------
    • Accept a raw natural-language question (AgentMessage.payload = str)
    • Run LAQA analysis to detect: language, cancer type, intent category,
      extracted entities (drugs, stages, procedures), metadata filters
    • Emit an AgentMessage with LAQAOutput as payload

    Why a separate agent?
    ---------------------
    Separating query understanding from retrieval allows independent
    evaluation, fine-tuning, and language expansion without touching
    retrieval or generation logic.
    """

    def __init__(self):
        super().__init__("QueryAgent")
        self.analyzer = LanguageAwareQueryAnalyzer()

    def handle(self, message: AgentMessage) -> AgentMessage:
        self.logger.info(f"[{message.trace_id}] Received query: {str(message.payload)[:80]}")
        t0 = time.perf_counter()

        raw_query = message.payload
        if not isinstance(raw_query, str) or not raw_query.strip():
            return self._make_error(message.trace_id, "RetrievalAgent",
                                    "QueryAgent requires a non-empty string query.")

        laqa_out: LAQAOutput = self.analyzer.analyze(raw_query)
        elapsed = (time.perf_counter() - t0) * 1000

        self.logger.info(
            f"[{message.trace_id}] LAQA done in {elapsed:.1f}ms | "
            f"cancer={laqa_out.cancer_type} | category={laqa_out.question_category}"
        )

        if laqa_out.needs_clarification:
            self.logger.warning(f"[{message.trace_id}] Clarification needed: {laqa_out.clarification_reason}")

        return AgentMessage(
            sender=self.name,
            receiver="RetrievalAgent",
            message_type="RETRIEVAL_REQUEST",
            payload={
                "laqa_output": asdict(laqa_out),
                "laqa_latency_ms": round(elapsed, 2)
            },
            trace_id=message.trace_id
        )


# ══════════════════════════════════════════════════════════════════════════════
# 5.  RETRIEVAL AGENT
#     Wraps ClinicalRetriever. Accepts LAQAOutput; emits ranked chunks.
# ══════════════════════════════════════════════════════════════════════════════

class RetrievalAgent(BaseAgent):
    """
    Agent 2 — Evidence Retrieval

    Responsibility
    --------------
    • Accept a LAQAOutput (from QueryAgent)
    • Run dense ChromaDB similarity search (top-15 candidates)
    • Apply Clinical Cross-Feature Re-Ranking:
        60% cosine similarity + 25% entity overlap + 15% category alignment
    • Return top-K re-ranked chunks with source citations

    Why a separate agent?
    ---------------------
    Retrieval is I/O-bound (vector DB calls). Isolating it makes it easy
    to swap the vector store (Pinecone, Weaviate …) or the embedding model
    without breaking the rest of the pipeline.
    """

    def __init__(self, top_k: int = 5):
        super().__init__("RetrievalAgent")
        self.retriever = ClinicalRetriever(analyzer=laqa)
        self.top_k = top_k

    def handle(self, message: AgentMessage) -> AgentMessage:
        self.logger.info(f"[{message.trace_id}] Starting retrieval …")
        t0 = time.perf_counter()

        payload = message.payload
        laqa_dict = payload.get("laqa_output", {})
        laqa_latency = payload.get("laqa_latency_ms", 0.0)

        # Reconstruct LAQAOutput from the dict (sent over the message bus)
        try:
            from laqa import LAQAOutput
            laqa_out = LAQAOutput(**{
                k: v for k, v in laqa_dict.items() if k in LAQAOutput.__dataclass_fields__
            })
        except Exception as e:
            return self._make_error(message.trace_id, "FactCheckAgent",
                                    f"Failed to deserialise LAQAOutput: {e}")

        try:
            ret_res: RetrievalResponse = self.retriever.retrieve(laqa_out, top_k=self.top_k)
        except Exception as e:
            return self._make_error(message.trace_id, "FactCheckAgent",
                                    f"ChromaDB retrieval failed: {e}")

        elapsed = (time.perf_counter() - t0) * 1000
        self.logger.info(
            f"[{message.trace_id}] Retrieved {ret_res.total_retrieved} chunks "
            f"in {elapsed:.1f}ms | fallback={ret_res.fallback_used}"
        )

        return AgentMessage(
            sender=self.name,
            receiver="FactCheckAgent",
            message_type="FACT_CHECK_REQUEST",
            payload={
                "retrieval_response": ret_res,
                "laqa_latency_ms": laqa_latency,
                "retrieval_latency_ms": round(elapsed, 2)
            },
            trace_id=message.trace_id
        )


# ══════════════════════════════════════════════════════════════════════════════
# 6.  FACT-CHECK AGENT
#     Verifies citations, scores evidence, filters low-confidence chunks.
# ══════════════════════════════════════════════════════════════════════════════

class FactCheckAgent(BaseAgent):
    """
    Agent 3 — Citation Verification & Grounding Confidence Scoring

    Responsibility
    --------------
    For every retrieved chunk, FactCheckAgent computes a composite
    Grounding Confidence Score (GCS):

        GCS = 0.60 × cosine_similarity
            + 0.25 × citation_completeness      (1.0 if doc+page present, else 0.0)
            + 0.15 × source_authority_score     (1.0 if NCCN/FIGO/ACOG/SGO, else 0.5)

    Chunks with GCS < CONFIDENCE_THRESHOLD are tagged LOW_CONFIDENCE and
    excluded from the final context passed to ResponseAgent.

    Flags
    -----
    VERIFIED         : GCS ≥ 0.70
    LOW_CONFIDENCE   : 0.35 ≤ GCS < 0.70
    MISSING_CITATION : doc or page absent → automatic LOW_CONFIDENCE

    Why a separate agent?
    ---------------------
    Keeps hallucination risk low. Even if the vector store returns a chunk
    with high cosine similarity, a missing page number or unknown issuing
    authority reduces the GCS — protecting clinical integrity.
    """

    CONFIDENCE_THRESHOLD = 0.35   # chunks below this GCS are removed

    def __init__(self):
        super().__init__("FactCheckAgent")

    def _score_chunk(self, chunk: RetrievedChunk) -> FactCheckResult:
        # Citation completeness
        has_doc = bool(chunk.source_document and chunk.source_document.lower() not in ("", "unknown"))
        has_page = bool(chunk.source_page and str(chunk.source_page).lower() not in ("", "unknown", "none"))
        citation_complete = has_doc and has_page
        citation_score = 1.0 if citation_complete else 0.0

        # Source authority
        auth = chunk.issuing_authority or ""
        authority_verified = any(a.lower() in auth.lower() for a in self.KNOWN_AUTHORITIES)
        authority_score = 1.0 if authority_verified else 0.5

        # Composite GCS
        gcs = (
            0.60 * chunk.similarity_score +
            0.25 * citation_score +
            0.15 * authority_score
        )
        gcs = round(min(max(gcs, 0.0), 1.0), 4)

        if not citation_complete:
            flag = "MISSING_CITATION"
        elif gcs >= 0.70:
            flag = "VERIFIED"
        else:
            flag = "LOW_CONFIDENCE"

        return FactCheckResult(
            chunk_id=chunk.chunk_id,
            source_document=chunk.source_document,
            source_page=str(chunk.source_page),
            section_heading=chunk.section_heading,
            similarity_score=round(chunk.similarity_score, 4),
            citation_complete=citation_complete,
            authority_verified=authority_verified,
            authority_name=auth,
            grounding_confidence=gcs,
            flag=flag
        )

    def handle(self, message: AgentMessage) -> AgentMessage:
        self.logger.info(f"[{message.trace_id}] Starting fact-check …")
        t0 = time.perf_counter()

        payload = message.payload
        ret_res: RetrievalResponse = payload.get("retrieval_response")
        laqa_lat = payload.get("laqa_latency_ms", 0.0)
        retr_lat = payload.get("retrieval_latency_ms", 0.0)

        if ret_res is None or not isinstance(ret_res, RetrievalResponse):
            return self._make_error(message.trace_id, "ResponseAgent",
                                    "FactCheckAgent received no valid RetrievalResponse.")

        results: List[FactCheckResult] = []
        verified_chunks: List[RetrievedChunk] = []

        for chunk in ret_res.chunks:
            fc = self._score_chunk(chunk)
            results.append(fc)
            if fc.grounding_confidence >= self.CONFIDENCE_THRESHOLD:
                verified_chunks.append(chunk)
            else:
                self.logger.warning(
                    f"[{message.trace_id}] Chunk {chunk.chunk_id} REMOVED: "
                    f"GCS={fc.grounding_confidence:.3f} (flag={fc.flag})"
                )

        verified_count = sum(1 for r in results if r.flag == "VERIFIED")
        low_conf_count = sum(1 for r in results if r.flag == "LOW_CONFIDENCE")
        missing_cit_count = sum(1 for r in results if r.flag == "MISSING_CITATION")
        overall_conf = (
            round(sum(r.grounding_confidence for r in results) / len(results), 4)
            if results else 0.0
        )

        elapsed = (time.perf_counter() - t0) * 1000

        self.logger.info(
            f"[{message.trace_id}] Fact-check: {len(results)} total | "
            f"VERIFIED={verified_count} | LOW={low_conf_count} | "
            f"MISSING_CIT={missing_cit_count} | overall_GCS={overall_conf:.3f}"
        )

        bundle = FactCheckBundle(
            trace_id=message.trace_id,
            total_chunks_checked=len(results),
            verified_count=verified_count,
            low_confidence_count=low_conf_count,
            missing_citation_count=missing_cit_count,
            overall_confidence=overall_conf,
            verified_chunks=verified_chunks,
            fact_check_results=results,
            latency_ms=round(elapsed, 2)
        )

        # Reconstruct a RetrievalResponse with only verified chunks
        verified_ret_res = RetrievalResponse(
            original_query=ret_res.original_query,
            normalized_query=ret_res.normalized_query,
            language=ret_res.language,
            cancer_type=ret_res.cancer_type,
            question_category=ret_res.question_category,
            applied_filter=ret_res.applied_filter,
            fallback_used=ret_res.fallback_used,
            fallback_reason=ret_res.fallback_reason,
            total_retrieved=len(verified_chunks),
            chunks=verified_chunks,
            latency_ms=ret_res.latency_ms
        )

        return AgentMessage(
            sender=self.name,
            receiver="ResponseAgent",
            message_type="RESPONSE_REQUEST",
            payload={
                "verified_retrieval_response": verified_ret_res,
                "fact_check_bundle": bundle,
                "laqa_latency_ms": laqa_lat,
                "retrieval_latency_ms": retr_lat,
                "fact_check_latency_ms": round(elapsed, 2)
            },
            trace_id=message.trace_id
        )


# ══════════════════════════════════════════════════════════════════════════════
# 7.  RESPONSE AGENT
#     Accepts verified chunks; produces the final GroundedAnswer.
# ══════════════════════════════════════════════════════════════════════════════

class ResponseAgent(BaseAgent):
    """
    Agent 4 — Grounded Answer Synthesis

    Responsibility
    --------------
    • Accept verified evidence chunks (from FactCheckAgent)
    • Pass them to ClinicalResponseGenerator (Gemini / GPT / deterministic)
    • Attach the FactCheckBundle summary to the final answer
    • Return a complete AgentPipelineResult

    Why a separate agent?
    ---------------------
    Response generation is the only network-dependent step (LLM API calls).
    Isolating it makes it trivial to swap or A/B test LLM backends without
    changing any upstream retrieval or fact-checking logic.
    """

    def __init__(self):
        super().__init__("ResponseAgent")
        self.generator = ClinicalResponseGenerator()

    def handle(self, message: AgentMessage) -> AgentMessage:
        self.logger.info(f"[{message.trace_id}] Starting response synthesis …")
        t0 = time.perf_counter()

        payload = message.payload
        verified_ret_res: RetrievalResponse = payload.get("verified_retrieval_response")
        fc_bundle: FactCheckBundle = payload.get("fact_check_bundle")
        laqa_lat = payload.get("laqa_latency_ms", 0.0)
        retr_lat = payload.get("retrieval_latency_ms", 0.0)
        fc_lat = payload.get("fact_check_latency_ms", 0.0)

        if verified_ret_res is None:
            return self._make_error(message.trace_id, "ORCHESTRATOR",
                                    "ResponseAgent received no verified RetrievalResponse.")

        try:
            answer: GroundedAnswer = self.generator.answer(verified_ret_res)
        except Exception as e:
            return self._make_error(message.trace_id, "ORCHESTRATOR",
                                    f"ClinicalResponseGenerator failed: {e}")

        elapsed = (time.perf_counter() - t0) * 1000
        total_lat = round(laqa_lat + retr_lat + fc_lat + elapsed, 2)

        self.logger.info(
            f"[{message.trace_id}] Response generated in {elapsed:.1f}ms | "
            f"backend={answer.generator_backend} | total_pipeline={total_lat}ms"
        )

        return AgentMessage(
            sender=self.name,
            receiver="ORCHESTRATOR",
            message_type="FINAL_RESPONSE",
            payload={
                "answer": answer,
                "fact_check_bundle": fc_bundle,
                "latency_breakdown": {
                    "laqa_ms": laqa_lat,
                    "retrieval_ms": retr_lat,
                    "fact_check_ms": fc_lat,
                    "generation_ms": round(elapsed, 2),
                    "total_pipeline_ms": total_lat
                }
            },
            trace_id=message.trace_id
        )


# ══════════════════════════════════════════════════════════════════════════════
# 8.  ORCHESTRATOR  (Pipeline Coordinator — not an agent itself)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentPipelineResult:
    """
    Final structured output from the full 4-agent pipeline.
    """
    trace_id: str
    original_query: str
    answer: GroundedAnswer
    fact_check_bundle: FactCheckBundle
    latency_breakdown: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "original_query": self.original_query,
            "answer": asdict(self.answer),
            "fact_check_bundle": self.fact_check_bundle.to_dict(),
            "latency_breakdown": self.latency_breakdown
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)


class MultiAgentOrchestrator:
    """
    Coordinates the 4-agent pipeline in sequential message-passing order:

        QueryAgent → RetrievalAgent → FactCheckAgent → ResponseAgent

    Usage
    -----
    orchestrator = MultiAgentOrchestrator()
    result = orchestrator.run("What is first-line chemotherapy for ovarian cancer?")
    print(result.answer.answer_text)
    print(result.fact_check_bundle.overall_confidence)
    """

    def __init__(self, top_k: int = 5):
        self.query_agent = QueryAgent()
        self.retrieval_agent = RetrievalAgent(top_k=top_k)
        self.fact_check_agent = FactCheckAgent()
        self.response_agent = ResponseAgent()
        logger.info("MultiAgentOrchestrator initialised with 4 agents.")

    def run(self, query: str, top_k: Optional[int] = None) -> AgentPipelineResult:
        """
        Execute the full pipeline for a single natural-language query.

        Parameters
        ----------
        query  : Raw clinical question (any supported language)
        top_k  : Override number of chunks to retrieve (default: instance default)

        Returns
        -------
        AgentPipelineResult with answer, citations, GCS scores, and latency breakdown
        """
        trace_id = str(uuid.uuid4())[:8]
        logger.info(f"[{trace_id}] Pipeline start: '{query[:80]}'")

        # Override top_k if provided
        if top_k is not None:
            self.retrieval_agent.top_k = top_k

        # Step 1: QueryAgent
        msg1 = AgentMessage(
            sender="ORCHESTRATOR",
            receiver="QueryAgent",
            message_type="QUERY",
            payload=query,
            trace_id=trace_id
        )
        msg2 = self.query_agent.handle(msg1)
        if msg2.message_type == "ERROR":
            raise RuntimeError(f"QueryAgent error: {msg2.payload}")

        # Step 2: RetrievalAgent
        msg3 = self.retrieval_agent.handle(msg2)
        if msg3.message_type == "ERROR":
            raise RuntimeError(f"RetrievalAgent error: {msg3.payload}")

        # Step 3: FactCheckAgent
        msg4 = self.fact_check_agent.handle(msg3)
        if msg4.message_type == "ERROR":
            raise RuntimeError(f"FactCheckAgent error: {msg4.payload}")

        # Step 4: ResponseAgent
        msg5 = self.response_agent.handle(msg4)
        if msg5.message_type == "ERROR":
            raise RuntimeError(f"ResponseAgent error: {msg5.payload}")

        payload = msg5.payload
        return AgentPipelineResult(
            trace_id=trace_id,
            original_query=query,
            answer=payload["answer"],
            fact_check_bundle=payload["fact_check_bundle"],
            latency_breakdown=payload["latency_breakdown"]
        )


# ══════════════════════════════════════════════════════════════════════════════
# 9.  CLI  —  python multi_agent_system.py "your question here"
# ══════════════════════════════════════════════════════════════════════════════

def _print_separator(char: str = "=", width: int = 80):
    print(char * width)


def _print_pipeline_result(result: AgentPipelineResult):
    _print_separator()
    print("       MULTI-AGENT CANCER INFORMATION SYSTEM — PIPELINE RESULT")
    _print_separator()
    print(f"  Trace ID     : {result.trace_id}")
    print(f"  Query        : {result.original_query}")
    print()

    # ── Latency breakdown
    lb = result.latency_breakdown
    print("  +- Latency Breakdown -----------------------------------------------------------")
    print(f"  |  QueryAgent   (LAQA)        : {lb.get('laqa_ms', 0):.1f} ms")
    print(f"  |  RetrievalAgent (ChromaDB)  : {lb.get('retrieval_ms', 0):.1f} ms")
    print(f"  |  FactCheckAgent (GCS)       : {lb.get('fact_check_ms', 0):.1f} ms")
    print(f"  |  ResponseAgent (Synthesis)  : {lb.get('generation_ms', 0):.1f} ms")
    print(f"  |  -----------------------------------------------------------------")
    print(f"  |  TOTAL PIPELINE             : {lb.get('total_pipeline_ms', 0):.1f} ms")
    print("  +-------------------------------------------------------------------------------")
    print()

    # ── Fact-check summary
    fc = result.fact_check_bundle
    print("  +- FactCheckAgent Summary ------------------------------------------------------")
    print(f"  |  Chunks Checked   : {fc.total_chunks_checked}")
    print(f"  |  [VERIFIED]       : {fc.verified_count}")
    print(f"  |  [LOW_CONF]       : {fc.low_confidence_count}")
    print(f"  |  [MISSING_CIT]    : {fc.missing_citation_count}")
    print(f"  |  Overall GCS      : {fc.overall_confidence * 100:.1f}%")
    print("  +-------------------------------------------------------------------------------")
    print()

    # ── Individual chunk GCS scores
    if fc.fact_check_results:
        print("  Chunk-Level Grounding Confidence Scores:")
        for r in fc.fact_check_results:
            icon = "[OK]" if r.flag == "VERIFIED" else ("[WARN]" if r.flag == "LOW_CONFIDENCE" else "[MISS]")
            print(f"    {icon} [{r.flag:17s}] GCS={r.grounding_confidence * 100:.1f}%  "
                  f"| {r.source_document[:45]}  (Page {r.source_page})")
        print()

    # ── Answer
    answer = result.answer
    _print_separator("-")
    print(f"  Backend : {answer.generator_backend}  |  "
          f"Evidence Sufficient: {'YES [OK]' if answer.is_sufficient else 'NO [WARN]'}")
    print(f"  Citations ({len(answer.citations)}):")
    for c in answer.citations:
        print(f"    * {c}")
    _print_separator("-")
    print("\nCLINICAL ANSWER:")
    print(answer.answer_text)
    print()
    print(f"[DISCLAIMER] {answer.disclaimer}")
    _print_separator()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Agent Cancer Information System (Stage E)"
    )
    parser.add_argument(
        "query",
        nargs="?",
        type=str,
        default="What is the recommended first-line chemotherapy for advanced ovarian cancer?",
        help="Natural-language cancer clinical question"
    )
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of evidence chunks to retrieve (default: 5)")
    parser.add_argument("--json", action="store_true",
                        help="Output the full result as JSON instead of formatted text")
    args = parser.parse_args()

    orchestrator = MultiAgentOrchestrator(top_k=args.top_k)
    result = orchestrator.run(args.query, top_k=args.top_k)

    if args.json:
        print(result.to_json())
    else:
        _print_pipeline_result(result)
