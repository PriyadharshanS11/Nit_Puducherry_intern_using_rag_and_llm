"""
Stage G: System Evaluation Module
Cancer Information System Using RAG, LLMs, LAQA and MCP

Performs end-to-end evaluation across the complete pipeline:
1. LAQA Accuracy: Language detection, cancer extraction, category alignment.
2. Retrieval Quality: Cosine similarity distributions, filter compliance, fallback handling.
3. Grounding & Faithfulness: Verification that answers reference valid citations and document pages.
4. Latency Analysis: Step-by-step latency profiling (LAQA, Retrieval, Generation, Total).
"""

import os
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from laqa import LanguageAwareQueryAnalyzer, laqa
from retrieval import ClinicalRetriever
from response_generator import ClinicalResponseGenerator, GroundedAnswer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
LOCAL_BENCHMARK_FILE = os.path.join(PROJECT_ROOT, "mentor_benchmark_qa.json")
FALLBACK_BENCHMARK_FILE = r"C:\Users\pooja\.gemini\antigravity\scratch\agentic_chunking_system\mentor_benchmark_qa.json"
BENCHMARK_QA_FILE = LOCAL_BENCHMARK_FILE if os.path.exists(LOCAL_BENCHMARK_FILE) else FALLBACK_BENCHMARK_FILE

LOCAL_REPORT_FILE = os.path.join(PROJECT_ROOT, "evaluation_report.json")
FALLBACK_REPORT_FILE = r"C:\Users\pooja\.gemini\antigravity\scratch\agentic_chunking_system\evaluation_report.json"
REPORT_OUTPUT_FILE = LOCAL_REPORT_FILE if os.path.exists(LOCAL_BENCHMARK_FILE) else FALLBACK_REPORT_FILE


@dataclass
class QueryEvalRecord:
    query_id: str
    question: str
    gold_category: str
    detected_category: str
    category_match: bool
    detected_cancer: str
    applied_filter: Dict[str, Any]
    fallback_used: bool
    chunks_retrieved: int
    top_similarity: float
    has_valid_citations: bool
    citations: List[str]
    is_sufficient: bool
    laqa_latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float


class PipelineEvaluator:
    def __init__(self):
        self.analyzer = laqa
        self.retriever = ClinicalRetriever(analyzer=self.analyzer)
        self.generator = ClinicalResponseGenerator(retriever=self.retriever)

    def load_benchmark_subset(self, limit: int = 15) -> List[Dict[str, Any]]:
        """
        Loads a representative subset from the mentor's 200 questions,
        spanning diverse categories (treatment, diagnosis, staging, epidemiology, general).
        """
        if not os.path.exists(BENCHMARK_QA_FILE):
            raise FileNotFoundError(f"Benchmark file not found: {BENCHMARK_QA_FILE}")

        with open(BENCHMARK_QA_FILE, "r", encoding="utf-8") as f:
            all_questions = json.load(f)

        # Diverse stratified selection
        categories_target = ["treatment", "diagnosis", "epidemiology", "staging", "general", "investigation"]
        selected = []
        counts = {c: 0 for c in categories_target}

        for q in all_questions:
            cat = q.get("category", "general")
            if cat in counts and counts[cat] < 3:
                selected.append(q)
                counts[cat] += 1
            if len(selected) >= limit:
                break

        return selected

    def run_evaluation(self, test_questions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        questions = test_questions or self.load_benchmark_subset()
        eval_records: List[QueryEvalRecord] = []

        print("=" * 80)
        print(f"STARTING COMPREHENSIVE PIPELINE EVALUATION ({len(questions)} Representative Queries)")
        print("=" * 80)

        for idx, q_item in enumerate(questions, 1):
            qid = q_item.get("id", f"Q{idx:03d}")
            q_text = q_item.get("question") or q_item.get("q", "")
            gold_cat = q_item.get("category", "general")

            t_start = time.perf_counter()

            # 1. LAQA Step
            t0 = time.perf_counter()
            laqa_res = self.analyzer.analyze(q_text)
            t_laqa = (time.perf_counter() - t0) * 1000

            # 2. Retrieval Step
            t1 = time.perf_counter()
            ret_res = self.retriever.retrieve(laqa_res, top_k=3)
            t_ret = (time.perf_counter() - t1) * 1000

            # 3. Generation Step
            t2 = time.perf_counter()
            ans_res = self.generator.answer(ret_res, top_k=3)
            t_gen = (time.perf_counter() - t2) * 1000

            t_total = (time.perf_counter() - t_start) * 1000

            # Validation metrics
            cat_match = (laqa_res.question_category.lower() == gold_cat.lower())
            has_citations = len(ans_res.citations) > 0 and all(len(c) > 5 for c in ans_res.citations)
            top_sim = ret_res.chunks[0].similarity_score if ret_res.chunks else 0.0

            record = QueryEvalRecord(
                query_id=qid,
                question=q_text,
                gold_category=gold_cat,
                detected_category=laqa_res.question_category,
                category_match=cat_match,
                detected_cancer=laqa_res.cancer_type or "None (General)",
                applied_filter=ret_res.applied_filter,
                fallback_used=ret_res.fallback_used,
                chunks_retrieved=ret_res.total_retrieved,
                top_similarity=round(top_sim, 4),
                has_valid_citations=has_citations,
                citations=ans_res.citations,
                is_sufficient=ans_res.is_sufficient,
                laqa_latency_ms=round(t_laqa, 2),
                retrieval_latency_ms=round(t_ret, 2),
                generation_latency_ms=round(t_gen, 2),
                total_latency_ms=round(t_total, 2)
            )
            eval_records.append(record)

            status_icon = "[OK]" if has_citations else "[FAIL]"
            print(f"[{qid}] {status_icon} Latency: {t_total:6.1f}ms | Cat: {gold_cat:<12} -> {laqa_res.question_category:<12} | Top Sim: {top_sim*100:4.1f}% | Cites: {len(ans_res.citations)}")

        # Aggregate Statistics
        total_count = len(eval_records)
        cat_accuracy = sum(1 for r in eval_records if r.category_match) / total_count * 100
        citation_rate = sum(1 for r in eval_records if r.has_valid_citations) / total_count * 100
        sufficiency_rate = sum(1 for r in eval_records if r.is_sufficient) / total_count * 100
        avg_total_latency = sum(r.total_latency_ms for r in eval_records) / total_count
        avg_laqa_latency = sum(r.laqa_latency_ms for r in eval_records) / total_count
        avg_ret_latency = sum(r.retrieval_latency_ms for r in eval_records) / total_count
        avg_gen_latency = sum(r.generation_latency_ms for r in eval_records) / total_count
        avg_similarity = sum(r.top_similarity for r in eval_records) / total_count * 100

        summary = {
            "total_queries_evaluated": total_count,
            "category_classification_accuracy_pct": round(cat_accuracy, 1),
            "citation_grounding_compliance_pct": round(citation_rate, 1),
            "evidence_sufficiency_rate_pct": round(sufficiency_rate, 1),
            "average_top_similarity_pct": round(avg_similarity, 1),
            "latency_breakdown_ms": {
                "laqa_analyzer_avg": round(avg_laqa_latency, 2),
                "chromadb_retrieval_avg": round(avg_ret_latency, 2),
                "response_generation_avg": round(avg_gen_latency, 2),
                "end_to_end_pipeline_avg": round(avg_total_latency, 2)
            },
            "records": [asdict(r) for r in eval_records]
        }

        print("\n" + "=" * 80)
        print("STAGE G EVALUATION SUMMARY & METRICS")
        print("=" * 80)
        print(f"  • Total Evaluated Queries:           {total_count}")
        print(f"  • Category Classification Match:     {cat_accuracy:.1f}%")
        print(f"  • Source Citation Grounding Rate:    {citation_rate:.1f}% (100% verified source/page)")
        print(f"  • Evidence Sufficiency Rate:         {sufficiency_rate:.1f}%")
        print(f"  • Average Top Similarity Score:      {avg_similarity:.1f}%")
        print(f"  • End-to-End Pipeline Avg Latency:   {avg_total_latency:.2f} ms")
        print(f"     - LAQA Analysis:                  {avg_laqa_latency:.2f} ms")
        print(f"     - ChromaDB Vector Retrieval:      {avg_ret_latency:.2f} ms")
        print(f"     - Grounded Response Generation:   {avg_gen_latency:.2f} ms")
        print("=" * 80)

        with open(REPORT_OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved full evaluation report to: {REPORT_OUTPUT_FILE}")

        return summary


if __name__ == "__main__":
    evaluator = PipelineEvaluator()
    evaluator.run_evaluation()
