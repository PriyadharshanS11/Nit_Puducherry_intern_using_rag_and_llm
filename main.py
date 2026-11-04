"""
Main Pipeline Orchestrator
Cancer Information System Using RAG, LLMs, LAQA and MCP

Architecture Pipeline:
User Question -> LAQA -> ChromaDB Similarity Retrieval -> Response Generator -> Citations & Disclaimer
"""

import sys
import os
import argparse
import json

from laqa import LanguageAwareQueryAnalyzer, laqa
from retrieval import ClinicalRetriever
from response_generator import ClinicalResponseGenerator
from database_inspection import inspect_database
from evaluation import PipelineEvaluator

def print_banner():
    banner = """
================================================================================
          CANCER INFORMATION SYSTEM: RAG + LAQA + CHROMADB PIPELINE             
================================================================================
    """
    print(banner)

def run_pipeline(query: str, top_k: int = 3, verbose: bool = True):
    print_banner()
    print(f"USER QUERY: \"{query}\"\n")

    # Step 1: LAQA Analysis
    print("--- [Step 1] LAQA: Language-Aware Query Analyzer ---")
    analyzer = laqa
    laqa_out = analyzer.analyze(query)
    print(f"  • Detected Language:     {laqa_out.language}")
    print(f"  • Cancer Type Identified: {laqa_out.cancer_type or 'None (General Oncology)'}")
    print(f"  • Question Category:      {laqa_out.question_category.upper()}")
    print(f"  • Normalized Search Text: \"{laqa_out.normalized_query}\"")
    print(f"  • Extracted Entities:     {laqa_out.extracted_entities}")
    print(f"  • Applicable Filters:     {laqa_out.metadata_filters}")
    print(f"  • Needs Clarification:    {laqa_out.needs_clarification}")
    if laqa_out.needs_clarification:
        print(f"  • Reason:                 {laqa_out.clarification_reason}")
        return

    # Step 2: ChromaDB Retrieval
    print("\n--- [Step 2] ChromaDB Similarity Search & Retrieval ---")
    retriever = ClinicalRetriever(analyzer=analyzer)
    ret_res = retriever.retrieve(laqa_out, top_k=top_k)
    print(f"  • Total Chunks Retrieved: {ret_res.total_retrieved}")
    print(f"  • Applied Filter:         {ret_res.applied_filter}")
    print(f"  • Fallback Used:          {ret_res.fallback_used} ({ret_res.fallback_reason})")
    print(f"  • Retrieval Latency:      {ret_res.latency_ms:.2f} ms")

    if ret_res.chunks:
        print("\n  Top Retrieved Evidence Chunks (Clinical Bi-Encoder + Re-Ranker):")
        for rank, c in enumerate(ret_res.chunks, 1):
            re_rank_str = f"Re-Rank: {c.re_rank_score*100:.1f}% | " if hasattr(c, "re_rank_score") else ""
            print(f"    [Rank #{rank}] {re_rank_str}Cosine Sim: {c.similarity_score * 100:.1f}% (Dist: {c.cosine_distance:.4f})")
            print(f"      Source:   {c.source_reference}")
            print(f"      Heading:  {c.section_heading}")
            print(f"      Stage:    {c.clinical_stage} | Modality: {c.treatment_modality}")
            if hasattr(c, "entity_match_hits") and c.entity_match_hits:
                print(f"      Entities: {', '.join(c.entity_match_hits)}")
            if c.drugs_mentioned != "None":
                print(f"      Drugs:    {c.drugs_mentioned} ({c.dosages_regimens})")

    # Step 3: Response Generation
    print("\n--- [Step 3] Grounded Response Generation ---")
    generator = ClinicalResponseGenerator(retriever=retriever)
    answer = generator.answer(ret_res, top_k=top_k)
    print(f"  • Generator Backend:     {answer.generator_backend}")
    print(f"  • Sufficient Evidence:    {answer.is_sufficient}")
    print(f"  • Generation Latency:     {answer.latency_ms:.2f} ms")
    print(f"  • Source Citations:       {answer.citations}\n")
    print("ANSWER:")
    print("-" * 80)
    print(answer.answer_text)
    print("-" * 80)
    print(f"\n{answer.disclaimer}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Cancer Information System CLI")
    parser.add_argument("query", nargs="?", type=str, help="Natural-language cancer question")
    parser.add_argument("--top_k", type=int, default=3, help="Number of chunks to retrieve (default: 3)")
    parser.add_argument("--inspect", action="store_true", help="Inspect ChromaDB database schema and records")
    parser.add_argument("--eval", action="store_true", help="Run benchmark evaluation across representative questions")
    
    args = parser.parse_args()

    if args.inspect:
        data = inspect_database()
        print(json.dumps(data, indent=2))
        return

    if args.eval:
        evaluator = PipelineEvaluator()
        evaluator.run_evaluation()
        return

    if args.query:
        run_pipeline(args.query, top_k=args.top_k)
    else:
        # Default sample question if none provided
        sample_q = "What is the recommended concurrent chemoradiotherapy protocol for cervical cancer?"
        run_pipeline(sample_q, top_k=args.top_k)

if __name__ == "__main__":
    main()
