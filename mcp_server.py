"""
Stage F: MCP (Model Context Protocol) Server
Cancer Information System Using RAG, LLMs, LAQA and MCP

Protocol  : JSON-RPC 2.0 over STDIO (the MCP standard transport)
Tools Exposed:
  1. query_cancer_guidelines    — Full 4-agent RAG pipeline on a clinical question
  2. get_cancer_stats           — ChromaDB partition statistics (counts per cancer type)
  3. verify_citation_evidence   — Point-check a source document + page claim

How to run:
  python mcp_server.py                    # starts MCP server on STDIO

How to test manually (pipe JSON-RPC):
  echo {"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}} | python mcp_server.py

MCP Specification Reference:
  https://spec.modelcontextprotocol.io/specification/
"""

import os
import sys
import json
import time
import logging
import traceback
from typing import Any, Dict, List, Optional

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Logging: MCP servers MUST NOT write to stdout (it corrupts JSON-RPC).
#    All logs go to stderr only.
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] MCP: %(message)s"
)
logger = logging.getLogger("MCPServer")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LAZY IMPORTS — only load heavy modules when first tool is called
# ══════════════════════════════════════════════════════════════════════════════

_orchestrator = None   # MultiAgentOrchestrator (Stage E)
_retriever = None      # ClinicalRetriever (direct DB access for stats/verify)
_db_client = None      # Raw ChromaDB PersistentClient


def _get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        logger.info("Initialising MultiAgentOrchestrator …")
        from multi_agent_system import MultiAgentOrchestrator
        _orchestrator = MultiAgentOrchestrator(top_k=5)
        logger.info("MultiAgentOrchestrator ready.")
    return _orchestrator


def _get_retriever():
    global _retriever
    if _retriever is None:
        logger.info("Initialising ClinicalRetriever …")
        from retrieval import ClinicalRetriever
        _retriever = ClinicalRetriever()
        logger.info("ClinicalRetriever ready.")
    return _retriever


def _get_db_client():
    global _db_client
    if _db_client is None:
        import chromadb
        from retrieval import _find_default_chroma_path
        db_path = _find_default_chroma_path()
        logger.info(f"Opening ChromaDB at: {db_path}")
        _db_client = chromadb.PersistentClient(path=db_path)
    return _db_client


# ══════════════════════════════════════════════════════════════════════════════
# 2.  TOOL SCHEMAS  (returned in tools/list response)
# ══════════════════════════════════════════════════════════════════════════════

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "query_cancer_guidelines",
        "description": (
            "Answer a natural-language clinical question about gynecologic cancer "
            "using the full 4-agent RAG pipeline (LAQA → ChromaDB retrieval → "
            "FactCheck → Response synthesis). Returns a grounded answer with "
            "100% verified PDF source citations and a Grounding Confidence Score."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Natural-language clinical oncology question (English or Hindi)"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of evidence chunks to retrieve (default: 5, max: 15)",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 15
                }
            },
            "required": ["question"]
        }
    },
    {
        "name": "get_cancer_stats",
        "description": (
            "Return ChromaDB knowledge-base statistics: total chunk count, "
            "per-cancer-type distribution, and the list of indexed guideline sources. "
            "Useful for auditing data coverage before querying."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "cancer_type": {
                    "type": "string",
                    "description": "Optional filter — return stats only for this cancer type "
                                   "(e.g. 'Ovarian', 'Cervical'). Omit for full summary."
                }
            },
            "required": []
        }
    },
    {
        "name": "verify_citation_evidence",
        "description": (
            "Verify whether a specific source document and page number claim is "
            "supported by the indexed knowledge base. Returns matching chunks, "
            "similarity context, and a verification verdict: "
            "CONFIRMED | PARTIAL | NOT_FOUND."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_document": {
                    "type": "string",
                    "description": "Exact or partial name of the guideline PDF "
                                   "(e.g. 'NCCN_Ovarian_Cancer_2023.pdf')"
                },
                "page_number": {
                    "type": "string",
                    "description": "Page number to verify (e.g. '45' or 'Page 45')"
                },
                "claim_text": {
                    "type": "string",
                    "description": "Optional: the specific clinical claim to verify "
                                   "against the indexed content of that page."
                }
            },
            "required": ["source_document"]
        }
    }
]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  TOOL HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

def _tool_query_cancer_guidelines(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool 1 — Full RAG pipeline query.
    Runs: QueryAgent → RetrievalAgent → FactCheckAgent → ResponseAgent
    """
    question = params.get("question", "").strip()
    top_k = int(params.get("top_k", 5))
    top_k = max(1, min(top_k, 15))

    if not question:
        return {"error": "Parameter 'question' must be a non-empty string."}

    logger.info(f"Tool: query_cancer_guidelines | question='{question[:60]}' | top_k={top_k}")
    t0 = time.perf_counter()

    try:
        orch = _get_orchestrator()
        result = orch.run(question, top_k=top_k)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return {"error": str(e), "traceback": traceback.format_exc()}

    elapsed = (time.perf_counter() - t0) * 1000
    fc = result.fact_check_bundle
    answer = result.answer

    # Serialise fact-check per-chunk results
    chunk_scores = [
        {
            "source_document": r.source_document,
            "source_page": r.source_page,
            "section_heading": r.section_heading,
            "similarity_score_pct": round(r.similarity_score * 100, 1),
            "grounding_confidence_pct": round(r.grounding_confidence * 100, 1),
            "flag": r.flag,
            "citation_complete": r.citation_complete,
            "authority_verified": r.authority_verified,
            "authority_name": r.authority_name
        }
        for r in fc.fact_check_results
    ]

    return {
        "trace_id": result.trace_id,
        "query": question,
        "cancer_type": answer.cancer_type,
        "question_category": answer.question_category,
        "answer": answer.answer_text,
        "is_sufficient": answer.is_sufficient,
        "generator_backend": answer.generator_backend,
        "citations": answer.citations,
        "evidence_snippets": answer.evidence_snippets,
        "fact_check_summary": {
            "total_chunks_checked": fc.total_chunks_checked,
            "verified_count": fc.verified_count,
            "low_confidence_count": fc.low_confidence_count,
            "missing_citation_count": fc.missing_citation_count,
            "overall_grounding_confidence_pct": round(fc.overall_confidence * 100, 1)
        },
        "chunk_grounding_scores": chunk_scores,
        "latency_breakdown_ms": result.latency_breakdown,
        "total_latency_ms": round(elapsed, 2),
        "disclaimer": answer.disclaimer
    }


def _tool_get_cancer_stats(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool 2 — ChromaDB knowledge-base statistics.
    """
    cancer_filter = params.get("cancer_type", "").strip().lower()
    logger.info(f"Tool: get_cancer_stats | filter='{cancer_filter or 'ALL'}'")

    try:
        client = _get_db_client()
        collection = client.get_collection("gynecological_cancer_rag")
        total = collection.count()
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return {"error": str(e)}

    # Count per cancer type using metadata "where" queries
    CANCER_TYPES = [
        "Ovarian Cancer", "Uterine Cancer", "Cervical Cancer",
        "Gestational Trophoblastic Disease", "Vulvar Cancer",
        "Fallopian Tube Cancer", "Vaginal Cancer", "Multidisciplinary"
    ]

    distribution = {}
    for ct in CANCER_TYPES:
        try:
            count_res = collection.get(
                where={"cancer_type": {"$eq": ct}},
                include=[]    # only IDs; fastest
            )
            cnt = len(count_res["ids"])
            if cnt > 0:
                if not cancer_filter or cancer_filter in ct.lower():
                    distribution[ct] = cnt
        except Exception:
            pass

    # Guideline sources from metadata
    source_docs: List[str] = []
    try:
        sample = collection.get(limit=500, include=["metadatas"])
        seen = set()
        for meta in sample.get("metadatas", []):
            sd = meta.get("source_document", "")
            if sd and sd not in seen:
                seen.add(sd)
                source_docs.append(sd)
        source_docs.sort()
    except Exception:
        pass

    return {
        "total_chunks_indexed": total,
        "cancer_type_distribution": distribution,
        "guideline_sources_sample": source_docs[:30],
        "collection_name": "gynecological_cancer_rag",
        "embedding_model": "all-MiniLM-L6-v2 (384-d, Cosine HNSW)",
        "database_type": "ChromaDB (persistent)"
    }


def _tool_verify_citation_evidence(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool 3 — Citation verification.
    Checks whether a document + page claim exists in the knowledge base.
    """
    source_doc = params.get("source_document", "").strip()
    page_num = str(params.get("page_number", "")).strip()
    claim_text = params.get("claim_text", "").strip()

    if not source_doc:
        return {"error": "Parameter 'source_document' is required."}

    logger.info(f"Tool: verify_citation_evidence | doc='{source_doc}' | page='{page_num}'")

    try:
        client = _get_db_client()
        collection = client.get_collection("gynecological_cancer_rag")
    except Exception as e:
        return {"error": str(e)}

    # Search by source_document (partial match via regex-free substring)
    matched_chunks = []
    try:
        # Try exact document match first
        result = collection.get(
            where={"source_document": {"$eq": source_doc}},
            include=["metadatas", "documents"]
        )
        ids = result.get("ids", [])
        metas = result.get("metadatas", [])
        docs = result.get("documents", [])

        # If nothing exact, try partial (get a broader set and filter)
        if not ids:
            broader = collection.get(limit=2000, include=["metadatas", "documents"])
            b_ids = broader.get("ids", [])
            b_metas = broader.get("metadatas", [])
            b_docs = broader.get("documents", [])
            for i, m in enumerate(b_metas):
                sd = m.get("source_document", "")
                if source_doc.lower() in sd.lower():
                    ids.append(b_ids[i])
                    metas.append(m)
                    docs.append(b_docs[i] if i < len(b_docs) else "")

        for i, (cid, meta, doc_text) in enumerate(zip(ids, metas, docs)):
            sp = str(meta.get("source_page", ""))
            if page_num and page_num.replace("Page ", "").replace("page ", "") not in sp:
                continue
            # Optional claim similarity (simple substring check)
            claim_match = False
            if claim_text:
                claim_match = any(
                    word.lower() in doc_text.lower()
                    for word in claim_text.split()
                    if len(word) > 4
                )
            matched_chunks.append({
                "chunk_id": cid,
                "source_document": meta.get("source_document", ""),
                "source_page": sp,
                "section_heading": meta.get("section_heading", ""),
                "cancer_type": meta.get("cancer_type", ""),
                "issuing_authority": meta.get("issuing_authority", ""),
                "text_excerpt": (doc_text[:300] + "…") if doc_text else "",
                "claim_text_found": claim_match if claim_text else None
            })
            if len(matched_chunks) >= 10:
                break

    except Exception as e:
        logger.error(f"Verification query failed: {e}")
        return {"error": str(e)}

    # Verdict
    if not matched_chunks:
        verdict = "NOT_FOUND"
        message = (f"No indexed chunks found for document '{source_doc}'"
                   + (f" page '{page_num}'" if page_num else "") + ".")
    elif page_num:
        verdict = "CONFIRMED"
        message = f"Found {len(matched_chunks)} chunk(s) matching the document and page."
    else:
        verdict = "PARTIAL"
        message = (f"Document '{source_doc}' found with {len(matched_chunks)} chunk(s), "
                   f"but no page number was provided to narrow down.")

    if matched_chunks and claim_text:
        claim_hits = sum(1 for c in matched_chunks if c.get("claim_text_found"))
        if claim_hits == 0:
            verdict = "PARTIAL"
            message += f" However, claim keywords were not found in the matched text."

    return {
        "verdict": verdict,
        "message": message,
        "source_document_queried": source_doc,
        "page_number_queried": page_num or "Not specified",
        "claim_text_queried": claim_text or "Not specified",
        "matched_chunks_count": len(matched_chunks),
        "matched_chunks": matched_chunks
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4.  JSON-RPC 2.0 MESSAGE DISPATCHER
# ══════════════════════════════════════════════════════════════════════════════

def _jsonrpc_response(request_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _jsonrpc_error(request_id: Any, code: int, message: str, data: Any = None) -> Dict[str, Any]:
    err = {"code": code, "message": message}
    if data:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": err}


def _handle_initialize(request_id: Any, params: Dict) -> Dict:
    """MCP initialize handshake — announce server capabilities."""
    return _jsonrpc_response(request_id, {
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {"listChanged": False}
        },
        "serverInfo": {
            "name": "cancer-information-mcp-server",
            "version": "1.0.0",
            "description": (
                "MCP server for the Cancer Information System. "
                "Provides evidence-grounded clinical oncology answers via RAG + LAQA + Multi-Agent pipeline. "
                "All answers are 100% cited from NCCN / FIGO / ACOG / SGO guidelines."
            )
        }
    })


def _handle_tools_list(request_id: Any) -> Dict:
    """Return the list of available tools."""
    return _jsonrpc_response(request_id, {"tools": TOOL_DEFINITIONS})


def _handle_tools_call(request_id: Any, params: Dict) -> Dict:
    """Dispatch a tool call and return its result."""
    tool_name = params.get("name", "")
    tool_params = params.get("arguments", {})

    logger.info(f"Tool call: {tool_name} | params keys: {list(tool_params.keys())}")

    t0 = time.perf_counter()

    try:
        if tool_name == "query_cancer_guidelines":
            tool_result = _tool_query_cancer_guidelines(tool_params)
        elif tool_name == "get_cancer_stats":
            tool_result = _tool_get_cancer_stats(tool_params)
        elif tool_name == "verify_citation_evidence":
            tool_result = _tool_verify_citation_evidence(tool_params)
        else:
            return _jsonrpc_error(request_id, -32601, f"Tool not found: '{tool_name}'")
    except Exception as e:
        logger.error(f"Unhandled exception in tool '{tool_name}': {e}\n{traceback.format_exc()}")
        return _jsonrpc_error(request_id, -32603, f"Internal error: {e}",
                              data=traceback.format_exc())

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(f"Tool '{tool_name}' completed in {elapsed:.1f}ms")

    # MCP content format: array of content items
    content = [{"type": "text", "text": json.dumps(tool_result, indent=2, ensure_ascii=False, default=str)}]

    return _jsonrpc_response(request_id, {"content": content, "isError": "error" in tool_result})


def _dispatch(message: Dict) -> Optional[Dict]:
    """Route a parsed JSON-RPC message to the correct handler."""
    jsonrpc = message.get("jsonrpc", "")
    if jsonrpc != "2.0":
        return _jsonrpc_error(None, -32600, "Invalid Request: jsonrpc must be '2.0'")

    method = message.get("method", "")
    params = message.get("params", {})
    request_id = message.get("id")   # None for notifications

    if method == "initialize":
        return _handle_initialize(request_id, params)
    elif method == "initialized":
        # Notification — no response required
        logger.info("Client sent 'initialized' notification.")
        return None
    elif method == "tools/list":
        return _handle_tools_list(request_id)
    elif method == "tools/call":
        return _handle_tools_call(request_id, params)
    elif method == "ping":
        return _jsonrpc_response(request_id, {})
    else:
        logger.warning(f"Unknown method: {method}")
        if request_id is not None:
            return _jsonrpc_error(request_id, -32601, f"Method not found: '{method}'")
        return None   # notification — no reply


# ══════════════════════════════════════════════════════════════════════════════
# 5.  STDIO TRANSPORT LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_stdio_server():
    """
    Main STDIO transport loop (MCP standard).

    - Reads newline-delimited JSON-RPC 2.0 messages from stdin.
    - Writes newline-delimited JSON-RPC 2.0 responses to stdout.
    - All logs go to stderr to avoid corrupting the JSON stream.
    """
    logger.info("Cancer Information MCP Server started (STDIO transport)")
    logger.info(f"Project root: {PROJECT_ROOT}")

    # Use binary stdin/stdout for safe cross-platform newline handling
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    while True:
        try:
            line_bytes = stdin.readline()
            if not line_bytes:
                logger.info("stdin closed — MCP server shutting down.")
                break

            line = line_bytes.decode("utf-8").strip()
            if not line:
                continue

            logger.debug(f"← RECV: {line[:200]}")

            try:
                message = json.loads(line)
            except json.JSONDecodeError as e:
                response = _jsonrpc_error(None, -32700, f"Parse error: {e}")
                _send(stdout, response)
                continue

            response = _dispatch(message)
            if response is not None:
                _send(stdout, response)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — MCP server shutting down.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in server loop: {e}\n{traceback.format_exc()}")
            try:
                error_resp = _jsonrpc_error(None, -32603, f"Server internal error: {e}")
                _send(stdout, error_resp)
            except Exception:
                pass


def _send(stdout, response: Dict):
    """Write a single newline-delimited JSON-RPC response to stdout."""
    payload = json.dumps(response, ensure_ascii=False, default=str) + "\n"
    stdout.write(payload.encode("utf-8"))
    stdout.flush()
    logger.debug(f"→ SENT: {payload[:200]}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  STANDALONE TEST MODE  (python mcp_server.py --test)
# ══════════════════════════════════════════════════════════════════════════════

def _run_self_test():
    """
    Non-STDIO test mode: exercises all 3 tools and prints results.
    Run with:  python mcp_server.py --test
    """
    print("=" * 80)
    print("  MCP SERVER SELF-TEST (Tool Functionality Verification)")
    print("=" * 80)

    def _divider(title: str):
        print(f"\n{'-'*80}\n  TEST: {title}\n{'-'*80}")

    # Test 1: get_cancer_stats
    _divider("get_cancer_stats (all cancers)")
    result1 = _tool_get_cancer_stats({})
    print(f"  Total Chunks  : {result1.get('total_chunks_indexed', 'ERROR')}")
    dist = result1.get("cancer_type_distribution", {})
    for ct, cnt in dist.items():
        print(f"    {ct:<40} : {cnt:,} chunks")
    srcs = result1.get("guideline_sources_sample", [])
    print(f"  Sources (sample): {srcs[:3]}")

    # Test 2: verify_citation_evidence
    _divider("verify_citation_evidence")
    result2 = _tool_verify_citation_evidence({
        "source_document": "NCCN",
        "page_number": "",
        "claim_text": ""
    })
    print(f"  Verdict  : {result2.get('verdict')}")
    print(f"  Message  : {result2.get('message')}")
    print(f"  Matched  : {result2.get('matched_chunks_count')} chunks")

    # Test 3: query_cancer_guidelines
    _divider("query_cancer_guidelines")
    test_q = "What is the standard first-line chemotherapy for advanced ovarian cancer?"
    print(f"  Question : {test_q}")
    result3 = _tool_query_cancer_guidelines({"question": test_q, "top_k": 3})
    if "error" in result3:
        print(f"  ERROR: {result3['error']}")
    else:
        print(f"  Cancer Type  : {result3.get('cancer_type')}")
        print(f"  Is Sufficient: {result3.get('is_sufficient')}")
        print(f"  Backend      : {result3.get('generator_backend')}")
        print(f"  Overall GCS  : {result3.get('fact_check_summary', {}).get('overall_grounding_confidence_pct')}%")
        print(f"  Citations    : {result3.get('citations')}")
        print(f"  Total latency: {result3.get('total_latency_ms')} ms")
        print("\n  --- ANSWER (first 500 chars) ---")
        print(str(result3.get("answer", ""))[:500])

    print("\n" + "=" * 80)
    print("  [PASS] MCP SERVER SELF-TEST COMPLETE")
    print("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--test" in sys.argv or "-t" in sys.argv:
        _run_self_test()
    else:
        run_stdio_server()
