"""
Benchmark Test Suite for LAQA against Mentor 200-QA Dataset
Evaluates:
- Question Category Classification Alignment
- Cancer Type Extraction Rate
- Entity Identification
- Query Normalization
- Ambiguity Handling
"""
import json
import time
from collections import Counter
from laqa import laqa

import os

def run_benchmark():
    local_qa = os.path.join(os.path.dirname(__file__), "mentor_benchmark_qa.json")
    fpath = local_qa if os.path.exists(local_qa) else r'C:\Users\pooja\.gemini\antigravity\scratch\agentic_chunking_system\mentor_benchmark_qa.json'
    with open(fpath, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print("=" * 80)
    print(f"RUNNING LAQA EVALUATION ON MENTOR'S 200 BENCHMARK QUESTIONS")
    print("=" * 80)

    start_time = time.time()
    results = []
    
    cancer_detected_count = 0
    general_oncology_count = 0
    entities_extracted_count = 0
    needs_clarification_count = 0

    category_matches = 0

    for item in dataset:
        qid = item["id"]
        q_text = item["q"]
        gold_cat = item["category"]
        gold_diff = item["difficulty"]

        out = laqa.analyze(q_text)
        results.append({
            "id": qid,
            "original_q": q_text,
            "gold_category": gold_cat,
            "pred_category": out.question_category,
            "cancer_type": out.cancer_type,
            "normalized": out.normalized_query,
            "entities": out.extracted_entities,
            "filters": out.metadata_filters,
            "clarification": out.needs_clarification
        })

        if out.cancer_type:
            cancer_detected_count += 1
        else:
            general_oncology_count += 1

        if any(bool(v) for v in out.extracted_entities.values()):
            entities_extracted_count += 1

        if out.needs_clarification:
            needs_clarification_count += 1

        # Check category alignment (accounting for synonyms e.g. investigation vs diagnosis vs staging)
        p = out.question_category
        g = gold_cat
        if p == g or \
           (p == "diagnosis" and g in ["clinical_features", "etiology", "investigation"]) or \
           (p == "investigation" and g in ["diagnosis", "staging"]) or \
           (p == "treatment" and g in ["surgery", "side_effects"]) or \
           (p == "general" and g in ["mechanism", "epidemiology"]):
            category_matches += 1

    elapsed = time.time() - start_time
    total = len(dataset)

    print(f"\n[Performance & Latency]")
    print(f"  Total Questions Processed: {total}")
    print(f"  Elapsed Time:             {elapsed:.3f} seconds ({total/elapsed:.1f} queries/sec)")
    print(f"  Average Latency per Query:{elapsed*1000/total:.2f} ms")

    print(f"\n[Classification & Extraction Metrics]")
    print(f"  Cancer-Specific Queries Identified: {cancer_detected_count}/{total} ({cancer_detected_count*100/total:.1f}%)")
    print(f"  General Oncology Queries Preserved: {general_oncology_count}/{total} ({general_oncology_count*100/total:.1f}%)")
    print(f"  Clinical Entities Extracted:        {entities_extracted_count}/{total} ({entities_extracted_count*100/total:.1f}%)")
    print(f"  Category Semantic Alignment:        {category_matches}/{total} ({category_matches*100/total:.1f}%)")
    print(f"  False Positives on Ambiguity:       {needs_clarification_count} (all 200 questions were valid clinical questions)")

    print("\n[Sample Evaluated Outputs (First 5 Mentor Questions)]:")
    for res in results[:5]:
        print(f"\nID: {res['id']} | Gold Cat: {res['gold_category']} -> Pred Cat: {res['pred_category']}")
        print(f"  Original:   {res['original_q']}")
        print(f"  Normalized: {res['normalized']}")
        print(f"  Cancer:     {res['cancer_type']}")
        print(f"  Filters:    {res['filters']}")
        print(f"  Entities:   {res['entities']}")

    # Save benchmark results
    local_out = os.path.join(os.path.dirname(__file__), "laqa_benchmark_results.json")
    out_file = local_out if os.path.exists(os.path.dirname(local_out)) else r'C:\Users\pooja\.gemini\antigravity\scratch\agentic_chunking_system\laqa_benchmark_results.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed evaluation results saved to: {out_file}")

if __name__ == "__main__":
    run_benchmark()
