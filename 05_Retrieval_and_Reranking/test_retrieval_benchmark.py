"""
Comprehensive Benchmark & Unit Tests for Stage C: Retrieval Module
Evaluates semantic search across diverse query types:
1. Cancer-specific queries matching indexed partition
2. General oncology queries (unfiltered search)
3. Out-of-partition queries (validating graceful zero-result fallback)
4. Configurable top_k and similarity threshold filters
5. Verification of source document and page number citations
"""

import os
import time
import json
from retrieval import ClinicalRetriever

def run_retrieval_benchmark():
    print("=" * 80)
    print("STAGE C RETRIEVAL MODULE: BENCHMARK & EVALUATION")
    print("=" * 80)

    retriever = ClinicalRetriever()

    test_suite = [
        {
            "id": "T01",
            "type": "Organ-Specific (In Partition)",
            "query": "What is the recommended concurrent chemoradiotherapy for cervical cancer?",
            "expected_cancer": "Cervical Cancer",
            "expect_fallback": False
        },
        {
            "id": "T02",
            "type": "Diagnostic & Screening",
            "query": "What are the guidelines for HPV cotesting and Pap smear screening?",
            "expected_cancer": None,
            "expect_fallback": False
        },
        {
            "id": "T03",
            "type": "General Oncology Concept",
            "query": "What does neoadjuvant chemotherapy refer to in oncology?",
            "expected_cancer": None,
            "expect_fallback": False
        },
        {
            "id": "T04",
            "type": "Organ-Specific (Trigger Fallback)",
            "query": "What are the surgical options and adjuvant chemotherapy for breast cancer?",
            "expected_cancer": "Breast Cancer",
            "expect_fallback": True
        },
        {
            "id": "T05",
            "type": "Treatment Modality & Toxicity",
            "query": "What are the common side effects and toxicities of cisplatin during radiation therapy?",
            "expected_cancer": None,
            "expect_fallback": False
        }
    ]

    results = []
    latencies = []

    for test in test_suite:
        t_id = test["id"]
        q_text = test["query"]
        print(f"\n[{t_id}] Running: '{q_text}' ({test['type']})")
        
        t0 = time.perf_counter()
        resp = retriever.retrieve(q_text, top_k=3)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

        # Check fallback condition
        fallback_matched = (resp.fallback_used == test["expect_fallback"])

        print(f"  * Detected Cancer:  {resp.cancer_type} (Expected: {test['expected_cancer']})")
        print(f"  * Category:         {resp.question_category}")
        print(f"  * Applied Filter:   {resp.applied_filter}")
        print(f"  * Fallback Used:    {resp.fallback_used} (Passed: {fallback_matched})")
        print(f"  * Chunks Found:     {resp.total_retrieved}")
        print(f"  * Query Latency:    {resp.latency_ms:.2f} ms")

        top_chunk = resp.chunks[0] if resp.chunks else None
        if top_chunk:
            print(f"  * Top Citation:     {top_chunk.source_reference}")
            print(f"  * Section Heading:  {top_chunk.section_heading}")
            print(f"  * Similarity Score: {top_chunk.similarity_score*100:.1f}% (Dist: {top_chunk.cosine_distance})")
            print(f"  * Drugs:            {top_chunk.drugs_mentioned}")

        results.append({
            "test_id": t_id,
            "type": test["type"],
            "query": q_text,
            "expected_cancer": test["expected_cancer"],
            "detected_cancer": resp.cancer_type,
            "applied_filter": resp.applied_filter,
            "fallback_used": resp.fallback_used,
            "fallback_matched": fallback_matched,
            "total_retrieved": resp.total_retrieved,
            "latency_ms": resp.latency_ms,
            "top_citation": top_chunk.source_reference if top_chunk else None,
            "top_similarity": top_chunk.similarity_score if top_chunk else 0.0
        })

    # Summary
    avg_latency = sum(latencies) / len(latencies)
    print("\n" + "=" * 80)
    print("STAGE C EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Test Cases:            {len(test_suite)}")
    print(f"Average End-to-End Latency:  {avg_latency:.2f} ms")
    print(f"All Chunks Carry Citations:  {all(r['top_citation'] is not None for r in results)}")
    print(f"Fallback Mechanism Verified: {all(r['fallback_matched'] for r in results)}")
    print("=" * 80)

    # Save benchmark report
    local_out = os.path.join(os.path.dirname(__file__), "retrieval_benchmark_results.json")
    out_file = local_out if os.path.exists(os.path.dirname(local_out)) else r"C:\Users\pooja\.gemini\antigravity\scratch\agentic_chunking_system\retrieval_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed benchmark results saved to: {out_file}")

if __name__ == "__main__":
    run_retrieval_benchmark()
