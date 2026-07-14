"""
Evaluation Script for Legal RAG System.
Tests metadata extraction, chunking, and retrieval with filters.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


def load_test_queries() -> List[Dict[str, Any]]:
    """Load predefined test queries covering different user intents."""
    return [
        {
            "id": "Q1",
            "intent": "specific_precedent",
            "query": "How did Google handle IP Ownership in their vendor contracts?",
            "expected_filters": {"party_1": "Google", "clause_type": "intellectual_property"},
            "expected_corpus": ["cuad"],
            "description": "Find specific company precedent"
        },
        {
            "id": "Q2",
            "intent": "market_standard",
            "query": "What is the typical Liability Cap language in Cloud Services agreements?",
            "expected_filters": {"contract_type": "Services Agreement", "has_liability_cap": True},
            "expected_corpus": ["cuad"],
            "description": "Find market standard clause"
        },
        {
            "id": "Q3",
            "intent": "comparative_analysis",
            "query": "Compare the Confidentiality terms in NDAs vs. Merger Agreements.",
            "expected_filters": {"clause_type": "confidentiality"},
            "expected_corpus": ["contractnli", "maud"],
            "description": "Cross-corpus comparison"
        },
        {
            "id": "Q4",
            "intent": "content_search",
            "query": "Find me a Force Majeure clause that specifically mentions pandemics.",
            "expected_filters": {"clause_type": "force_majeure"},
            "expected_corpus": ["cuad", "maud"],
            "description": "Content-based search without company filter"
        },
        {
            "id": "Q5",
            "intent": "jurisdiction_search",
            "query": "Show me examples of Governing Law clauses set to New York in SaaS agreements.",
            "expected_filters": {"governing_law": "New York", "contract_subtype": "SaaS"},
            "expected_corpus": ["cuad"],
            "description": "Jurisdiction-specific search"
        },
        {
            "id": "Q6",
            "intent": "clause_retrieval",
            "query": "Retrieve any Termination clause with a 30-day notice period.",
            "expected_filters": {"clause_type": "termination"},
            "expected_corpus": ["cuad", "contractnli", "maud"],
            "description": "Specific clause type retrieval"
        },
        {
            "id": "Q7",
            "intent": "company_specific",
            "query": "Show me all Non-Compete clauses in agreements signed by Tesla.",
            "expected_filters": {"party_1": "Tesla", "clause_type": "non_compete"},
            "expected_corpus": ["cuad"],
            "description": "Company + clause type filter"
        },
        {
            "id": "Q8",
            "intent": "temporal_company",
            "query": "How did Microsoft structure the Indemnification cap in their 2023 licensing deals?",
            "expected_filters": {"party_1": "Microsoft", "clause_type": "indemnification", "contract_type": "License Agreement"},
            "expected_corpus": ["cuad"],
            "description": "Company + year + clause type"
        },
        {
            "id": "Q9",
            "intent": "maud_sector",
            "query": "Find all Merger Agreements where Company A acquired a target in the Healthcare sector.",
            "expected_filters": {"target_industry": "Healthcare", "deal_type": "Merger"},
            "expected_corpus": ["maud"],
            "description": "M&A sector-specific search"
        },
        {
            "id": "Q10",
            "intent": "nda_specific",
            "query": "Find NDAs with injunctive relief provisions.",
            "expected_filters": {"has_injunctive_relief": True},
            "expected_corpus": ["contractnli"],
            "description": "NDA-specific feature search"
        }
    ]


def evaluate_metadata_extraction(corpus_dir: Path, output_file: Path) -> Dict[str, Any]:
    """
    Evaluate metadata extraction quality on sample documents.
    
    Args:
        corpus_dir: Directory containing corpus folders
        output_file: Path to save evaluation results
    
    Returns:
        Evaluation results dictionary
    """
    from src.metadata_extractor import MetadataExtractor, ContractMetadata
    
    extractor = MetadataExtractor()
    results = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "metadata_extraction",
        "documents_tested": [],
        "summary": {}
    }
    
    # Test one document from each corpus type
    corpus_types = ["cuad", "contractnli", "maud"]
    
    for corpus_type in corpus_types:
        corpus_path = corpus_dir / corpus_type
        if not corpus_path.exists():
            print(f"Skipping {corpus_type}: directory not found")
            continue
        
        # Get first document
        docs = list(corpus_path.glob("*.txt"))[:1]
        if not docs:
            print(f"No documents found in {corpus_type}")
            continue
        
        doc_path = docs[0]
        with open(doc_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Extract metadata
        metadata = extractor.extract_metadata(
            text=text,
            file_path=doc_path,
            corpus_type=corpus_type,
            document_id=f"test_{corpus_type}"
        )
        
        doc_result = {
            "file": doc_path.name,
            "corpus_type": corpus_type,
            "metadata": metadata.to_dict(),
            "extraction_success": True
        }
        
        # Check key fields
        checks = {
            "has_parties": metadata.party_1 is not None,
            "has_governing_law": metadata.governing_law is not None,
            "has_contract_type": metadata.contract_type is not None,
            "has_clause_types": len(metadata.clause_types_present) > 0,
            "has_industry": metadata.industry_sector is not None
        }
        
        doc_result["field_checks"] = checks
        doc_result["fields_extracted"] = sum(1 for v in checks.values() if v)
        
        results["documents_tested"].append(doc_result)
        print(f"✓ Tested {doc_path.name}: {doc_result['fields_extracted']}/5 fields extracted")
    
    # Calculate summary
    total_docs = len(results["documents_tested"])
    if total_docs > 0:
        avg_fields = sum(d["fields_extracted"] for d in results["documents_tested"]) / total_docs
        results["summary"] = {
            "total_documents": total_docs,
            "average_fields_extracted": round(avg_fields, 2),
            "success_rate": 1.0  # All documents processed
        }
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    return results


def evaluate_retrieval_with_filters(index_dir: Path, output_file: Path) -> Dict[str, Any]:
    """
    Evaluate retrieval system with metadata filters.
    
    Args:
        index_dir: Directory containing indexes
        output_file: Path to save evaluation results
    
    Returns:
        Evaluation results dictionary
    """
    from src.retrieval_service import RetrievalService
    
    service = RetrievalService(index_dir=index_dir)
    queries = load_test_queries()
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "retrieval_with_filters",
        "query_results": [],
        "summary": {}
    }
    
    available_indexes = service.discover_indexes()
    print(f"Available indexes: {available_indexes}")
    
    if not available_indexes:
        print("No indexes found. Please run ingestion first.")
        results["summary"]["error"] = "No indexes available"
        return results
    
    for query_spec in queries:
        print(f"\nTesting Query {query_spec['id']}: {query_spec['description']}")
        
        # Determine which categories to search
        search_categories = [
            cat for cat in available_indexes 
            if cat in query_spec["expected_corpus"]
        ]
        
        if not search_categories:
            print(f"  ⚠ Skipping: No matching indexes for {query_spec['expected_corpus']}")
            continue
        
        # Test 1: Search without filters
        results_no_filter = service.search(
            query=query_spec["query"],
            categories=search_categories,
            top_k=5
        )
        
        # Test 2: Search with filters
        results_with_filter = service.search_with_filters(
            query=query_spec["query"],
            filters=query_spec["expected_filters"],
            categories=search_categories,
            top_k=5
        )
        
        query_result = {
            "query_id": query_spec["id"],
            "query": query_spec["query"],
            "intent": query_spec["intent"],
            "expected_filters": query_spec["expected_filters"],
            "expected_corpus": query_spec["expected_corpus"],
            "results_without_filter": len(results_no_filter),
            "results_with_filter": len(results_with_filter),
            "filter_effectiveness": None,
            "sample_results": []
        }
        
        # Calculate filter effectiveness
        if len(results_no_filter) > 0:
            reduction = (len(results_no_filter) - len(results_with_filter)) / len(results_no_filter)
            query_result["filter_effectiveness"] = round(reduction, 2)
        
        # Add sample results
        if results_with_filter:
            query_result["sample_results"] = [
                {
                    "chunk_id": r["chunk_id"],
                    "score": round(r["score"], 4),
                    "matched_filters": r.get("matched_filters", {}),
                    "text_preview": r["text"][:200]
                }
                for r in results_with_filter[:2]
            ]
        
        results["query_results"].append(query_result)
        print(f"  ✓ Results: {len(results_no_filter)} (no filter) → {len(results_with_filter)} (with filter)")
    
    # Calculate summary
    total_queries = len(results["query_results"])
    if total_queries > 0:
        successful_filters = sum(
            1 for q in results["query_results"]
            if q["filter_effectiveness"] is not None and q["filter_effectiveness"] >= 0
        )
        avg_reduction = sum(
            q["filter_effectiveness"] for q in results["query_results"]
            if q["filter_effectiveness"] is not None
        ) / max(1, successful_filters)
        
        results["summary"] = {
            "total_queries_tested": total_queries,
            "queries_with_filter_improvement": successful_filters,
            "average_result_reduction": round(avg_reduction, 2),
            "filter_success_rate": round(successful_filters / total_queries, 2)
        }
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    return results


def evaluate_end_to_end(corpus_dir: Path, index_dir: Path, output_file: Path) -> Dict[str, Any]:
    """
    End-to-end evaluation: ingestion → retrieval with real-world queries.
    
    Args:
        corpus_dir: Directory containing corpus
        index_dir: Directory containing indexes
        output_file: Path to save evaluation results
    
    Returns:
        Evaluation results dictionary
    """
    from src.retrieval_service import RetrievalService
    
    service = RetrievalService(index_dir=index_dir)
    queries = load_test_queries()
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "end_to_end",
        "scenario_results": [],
        "summary": {}
    }
    
    available_indexes = service.discover_indexes()
    
    # Define evaluation scenarios
    scenarios = [
        {
            "name": "specific_company_precedent",
            "query": "Show me all Non-Compete clauses in agreements signed by Tesla.",
            "filters": {"party_1": "Tesla", "clause_type": "non_compete"},
            "success_criteria": ["results_contain_party_name", "results_contain_clause_type"]
        },
        {
            "name": "cross_corpus_comparison",
            "query": "Compare confidentiality provisions in NDAs and Merger Agreements",
            "filters": {"clause_type": "confidentiality"},
            "success_criteria": ["results_from_multiple_corpus"]
        },
        {
            "name": "industry_specific_ma",
            "query": "Find merger agreements in healthcare sector",
            "filters": {"target_industry": "Healthcare", "deal_type": "Merger"},
            "success_criteria": ["results_match_industry"]
        }
    ]
    
    for scenario in scenarios:
        print(f"\nEvaluating scenario: {scenario['name']}")
        
        # Determine relevant categories
        if "ma" in scenario["name"] or "merger" in scenario["name"].lower():
            categories = ["maud"] if "maud" in available_indexes else available_indexes
        elif "nda" in scenario["name"].lower():
            categories = ["contractnli"] if "contractnli" in available_indexes else available_indexes
        else:
            categories = available_indexes
        
        # Execute search
        results_list = service.search_with_filters(
            query=scenario["query"],
            filters=scenario["filters"],
            categories=categories,
            top_k=10
        )
        
        # Evaluate success criteria
        criteria_met = {}
        
        if "results_contain_party_name" in scenario["success_criteria"]:
            criteria_met["results_contain_party_name"] = any(
                scenario["filters"].get("party_1", "").lower() in str(r.get("metadata", {})).lower()
                for r in results_list
            )
        
        if "results_contain_clause_type" in scenario["success_criteria"]:
            criteria_met["results_contain_clause_type"] = any(
                r.get("clause_type") == scenario["filters"].get("clause_type")
                for r in results_list
            )
        
        if "results_from_multiple_corpus" in scenario["success_criteria"]:
            corpus_types = set(r.get("category") for r in results_list)
            criteria_met["results_from_multiple_corpus"] = len(corpus_types) > 1
        
        if "results_match_industry" in scenario["success_criteria"]:
            criteria_met["results_match_industry"] = any(
                str(r.get("metadata", {}).get("target_industry", "")).lower() == 
                scenario["filters"].get("target_industry", "").lower()
                for r in results_list
            )
        
        scenario_result = {
            "scenario_name": scenario["name"],
            "query": scenario["query"],
            "filters_applied": scenario["filters"],
            "num_results": len(results_list),
            "criteria_met": criteria_met,
            "overall_success": all(criteria_met.values()) if criteria_met else False,
            "sample_result": results_list[0] if results_list else None
        }
        
        results["scenario_results"].append(scenario_result)
        
        success_count = sum(1 for v in criteria_met.values() if v)
        total_criteria = len(criteria_met)
        print(f"  ✓ Success: {success_count}/{total_criteria} criteria met")
    
    # Calculate summary
    total_scenarios = len(results["scenario_results"])
    successful_scenarios = sum(
        1 for s in results["scenario_results"] if s["overall_success"]
    )
    
    results["summary"] = {
        "total_scenarios": total_scenarios,
        "successful_scenarios": successful_scenarios,
        "success_rate": round(successful_scenarios / max(1, total_scenarios), 2)
    }
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Legal RAG System")
    parser.add_argument(
        "--test-type",
        choices=["metadata", "retrieval", "end_to_end", "all"],
        default="all",
        help="Type of evaluation to run"
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/corpus"),
        help="Directory containing corpus files"
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/index"),
        help="Directory containing indexes"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_results"),
        help="Directory to save evaluation results"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Legal RAG System Evaluation")
    print("=" * 80)
    
    if args.test_type in ["metadata", "all"]:
        print("\n[1/3] Evaluating Metadata Extraction...")
        metadata_output = args.output_dir / "metadata_evaluation.json"
        metadata_results = evaluate_metadata_extraction(args.corpus_dir, metadata_output)
        print(f"\nMetadata extraction results saved to: {metadata_output}")
        if metadata_results.get("summary"):
            print(f"Summary: {json.dumps(metadata_results['summary'], indent=2)}")
    
    if args.test_type in ["retrieval", "all"]:
        print("\n[2/3] Evaluating Retrieval with Filters...")
        retrieval_output = args.output_dir / "retrieval_evaluation.json"
        retrieval_results = evaluate_retrieval_with_filters(args.index_dir, retrieval_output)
        print(f"\nRetrieval evaluation results saved to: {retrieval_output}")
        if retrieval_results.get("summary"):
            print(f"Summary: {json.dumps(retrieval_results['summary'], indent=2)}")
    
    if args.test_type in ["end_to_end", "all"]:
        print("\n[3/3] Running End-to-End Evaluation...")
        e2e_output = args.output_dir / "end_to_end_evaluation.json"
        e2e_results = evaluate_end_to_end(args.corpus_dir, args.index_dir, e2e_output)
        print(f"\nEnd-to-end evaluation results saved to: {e2e_output}")
        if e2e_results.get("summary"):
            print(f"Summary: {json.dumps(e2e_results['summary'], indent=2)}")
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
