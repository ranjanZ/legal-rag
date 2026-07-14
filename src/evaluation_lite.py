
"""
Legal RAG Evaluation Script
============================
Evaluates retrieval performance on corpus_lite documents using Recall@K metrics.

This script:
1. Defines 50 ground-truth questions based on actual content in corpus_lite
2. Runs retrieval for each question
3. Computes Recall@1, @3, @5, @10 based on whether relevant chunks are retrieved
4. Generates a detailed report

Usage:
    python src/evaluation_corpus_lite.py [--top-k 10] [--verbose]
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from retrieval_service import RetrievalService


# ============================================================================
# GROUND TRUTH QUESTIONS FOR CORPUS_LITE
# ============================================================================
# Each entry contains:
# - question: The query to ask
# - expected_sources: List of file basenames that should contain the answer
# - clause_types: Expected clause types (for metadata filtering validation)
# - corpus_type: Which corpus this belongs to

GROUND_TRUTH_QUESTIONS = [
    # ========================================================================
    # CUAD - Co-Branding Agreement (2TheMart + i-Escrow)
    # ========================================================================
    {
        "question": "What advertising fees does i-Escrow pay to 2TheMart per Transaction Inquiry?",
        "expected_sources": ["2ThemartComInc_19990826_10-12G_EX-10.10_6700288_EX-10.10_Co-Branding Agreement_ Agency Agreement.txt"],
        "clause_types": ["payment", "fee_structure"],
        "corpus_type": "cuad",
        "difficulty": "easy"
    },
    {
        "question": "How can 2TheMart terminate the Co-Branding Agreement if development is delayed?",
        "expected_sources": ["2ThemartComInc_19990826_10-12G_EX-10.10_6700288_EX-10.10_Co-Branding Agreement_ Agency Agreement.txt"],
        "clause_types": ["termination", "timeline"],
        "corpus_type": "cuad",
        "difficulty": "medium"
    },
    {
        "question": "What restrictions apply to banner advertising on the Co-Branded Site?",
        "expected_sources": ["2ThemartComInc_19990826_10-12G_EX-10.10_6700288_EX-10.10_Co-Branding Agreement_ Agency Agreement.txt"],
        "clause_types": ["advertising", "restrictions"],
        "corpus_type": "cuad",
        "difficulty": "medium"
    },
    {
        "question": "What audit rights does 2TheMart have regarding i-Escrow's payment records?",
        "expected_sources": ["2ThemartComInc_19990826_10-12G_EX-10.10_6700288_EX-10.10_Co-Branding Agreement_ Agency Agreement.txt"],
        "clause_types": ["audit_rights", "reporting"],
        "corpus_type": "cuad",
        "difficulty": "hard"
    },
    {
        "question": "What intellectual property rights does i-Escrow grant to 2TheMart?",
        "expected_sources": ["2ThemartComInc_19990826_10-12G_EX-10.10_6700288_EX-10.10_Co-Branding Agreement_ Agency Agreement.txt"],
        "clause_types": ["ip_license", "content_ownership"],
        "corpus_type": "cuad",
        "difficulty": "medium"
    },
    {
        "question": "What trademark restrictions apply to the use of Domain Name in the agreement?",
        "expected_sources": ["2ThemartComInc_19990826_10-12G_EX-10.10_6700288_EX-10.10_Co-Branding Agreement_ Agency Agreement.txt"],
        "clause_types": ["trademark", "restrictions"],
        "corpus_type": "cuad",
        "difficulty": "hard"
    },

    # ========================================================================
    # CUAD - Services Agreement (ABILITY INC)
    # ========================================================================
    {
        "question": "What services are being provided under the ABILITY INC services agreement?",
        "expected_sources": ["ABILITYINC_06_15_2020-EX-4.25-SERVICES AGREEMENT.txt"],
        "clause_types": ["service_scope", "deliverables"],
        "corpus_type": "cuad",
        "difficulty": "easy"
    },
    {
        "question": "What are the payment terms and invoicing requirements in the ABILITY INC agreement?",
        "expected_sources": ["ABILITYINC_06_15_2020-EX-4.25-SERVICES AGREEMENT.txt"],
        "clause_types": ["payment_terms", "invoicing"],
        "corpus_type": "cuad",
        "difficulty": "medium"
    },
    {
        "question": "What indemnification obligations exist in the ABILITY INC services agreement?",
        "expected_sources": ["ABILITYINC_06_15_2020-EX-4.25-SERVICES AGREEMENT.txt"],
        "clause_types": ["indemnification"],
        "corpus_type": "cuad",
        "difficulty": "hard"
    },
    {
        "question": "How can either party terminate the ABILITY INC services agreement?",
        "expected_sources": ["ABILITYINC_06_15_2020-EX-4.25-SERVICES AGREEMENT.txt"],
        "clause_types": ["termination"],
        "corpus_type": "cuad",
        "difficulty": "medium"
    },
    {
        "question": "What confidentiality obligations are specified in the ABILITY INC agreement?",
        "expected_sources": ["ABILITYINC_06_15_2020-EX-4.25-SERVICES AGREEMENT.txt"],
        "clause_types": ["confidentiality"],
        "corpus_type": "cuad",
        "difficulty": "medium"
    },
    {
        "question": "What liability caps or limitations exist in the ABILITY INC services agreement?",
        "expected_sources": ["ABILITYINC_06_15_2020-EX-4.25-SERVICES AGREEMENT.txt"],
        "clause_types": ["liability_cap", "limitation_of_liability"],
        "corpus_type": "cuad",
        "difficulty": "hard"
    },

    # ========================================================================
    # CUAD - Joint Venture Agreement (ACCELERATED TECHNOLOGIES)
    # ========================================================================
    {
        "question": "What is the purpose and scope of the joint venture in the ACCELERATED TECHNOLOGIES agreement?",
        "expected_sources": ["ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT VENTURE AGREEMENT.txt"],
        "clause_types": ["purpose", "scope"],
        "corpus_type": "cuad",
        "difficulty": "easy"
    },
    {
        "question": "How are profits and losses shared in the ACCELERATED TECHNOLOGIES joint venture?",
        "expected_sources": ["ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT VENTURE AGREEMENT.txt"],
        "clause_types": ["profit_sharing", "financial_terms"],
        "corpus_type": "cuad",
        "difficulty": "medium"
    },
    {
        "question": "What governance structure is established for the joint venture?",
        "expected_sources": ["ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT VENTURE AGREEMENT.txt"],
        "clause_types": ["governance", "management"],
        "corpus_type": "cuad",
        "difficulty": "hard"
    },
    {
        "question": "What intellectual property contributions are required from each party?",
        "expected_sources": ["ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT VENTURE AGREEMENT.txt"],
        "clause_types": ["ip_contribution", "ip_ownership"],
        "corpus_type": "cuad",
        "difficulty": "hard"
    },
    {
        "question": "How can the joint venture be dissolved or terminated?",
        "expected_sources": ["ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT VENTURE AGREEMENT.txt"],
        "clause_types": ["termination", "dissolution"],
        "corpus_type": "cuad",
        "difficulty": "medium"
    },
    {
        "question": "What non-compete restrictions apply to the parties in the joint venture?",
        "expected_sources": ["ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT VENTURE AGREEMENT.txt"],
        "clause_types": ["non_compete", "restrictions"],
        "corpus_type": "cuad",
        "difficulty": "hard"
    },

    # ========================================================================
    # CONTRACTNLI - Bosch NDA
    # ========================================================================
    {
        "question": "What is the definition of Confidential Information in the Bosch NDA?",
        "expected_sources": ["01_Bosch-Automotive-Service-Solutions-Mutual-Non-Disclosure-Agreement-7-12-17.txt"],
        "clause_types": ["confidentiality", "definitions"],
        "corpus_type": "contractnli",
        "difficulty": "easy"
    },
    {
        "question": "How long must confidentiality obligations survive after termination of the Bosch NDA?",
        "expected_sources": ["01_Bosch-Automotive-Service-Solutions-Mutual-Non-Disclosure-Agreement-7-12-17.txt"],
        "clause_types": ["term", "survival"],
        "corpus_type": "contractnli",
        "difficulty": "medium"
    },
    {
        "question": "What exceptions allow disclosure without breaching the Bosch NDA?",
        "expected_sources": ["01_Bosch-Automotive-Service-Solutions-Mutual-Non-Disclosure-Agreement-7-12-17.txt"],
        "clause_types": ["exclusions", "exceptions"],
        "corpus_type": "contractnli",
        "difficulty": "medium"
    },
    {
        "question": "What injunctive relief provisions exist in the Bosch NDA?",
        "expected_sources": ["01_Bosch-Automotive-Service-Solutions-Mutual-Non-Disclosure-Agreement-7-12-17.txt"],
        "clause_types": ["injunctive_relief", "remedies"],
        "corpus_type": "contractnli",
        "difficulty": "hard"
    },
    {
        "question": "Which state law governs the Bosch NDA?",
        "expected_sources": ["01_Bosch-Automotive-Service-Solutions-Mutual-Non-Disclosure-Agreement-7-12-17.txt"],
        "clause_types": ["governing_law", "jurisdiction"],
        "corpus_type": "contractnli",
        "difficulty": "easy"
    },
    {
        "question": "What are the return or destruction requirements for confidential information?",
        "expected_sources": ["01_Bosch-Automotive-Service-Solutions-Mutual-Non-Disclosure-Agreement-7-12-17.txt"],
        "clause_types": ["return_of_materials", "destruction"],
        "corpus_type": "contractnli",
        "difficulty": "medium"
    },
    {
        "question": "Can the Recipient reverse engineer prototypes under the Bosch NDA?",
        "expected_sources": ["01_Bosch-Automotive-Service-Solutions-Mutual-Non-Disclosure-Agreement-7-12-17.txt"],
        "clause_types": ["reverse_engineering", "restrictions"],
        "corpus_type": "contractnli",
        "difficulty": "hard"
    },
    {
        "question": "What notice requirements exist for unauthorized use of confidential information?",
        "expected_sources": ["01_Bosch-Automotive-Service-Solutions-Mutual-Non-Disclosure-Agreement-7-12-17.txt"],
        "clause_types": ["notice", "breach_notification"],
        "corpus_type": "contractnli",
        "difficulty": "medium"
    },

    # ========================================================================
    # CONTRACTNLI - Munt NDA
    # ========================================================================
    {
        "question": "What is the subject matter of the NDA with The Munt?",
        "expected_sources": ["12032018_NDA_The%20Munt_EN.txt"],
        "clause_types": ["subject_matter", "purpose"],
        "corpus_type": "contractnli",
        "difficulty": "easy"
    },
    {
        "question": "What is the period of confidentiality in The Munt NDA?",
        "expected_sources": ["12032018_NDA_The%20Munt_EN.txt"],
        "clause_types": ["term", "confidentiality_period"],
        "corpus_type": "contractnli",
        "difficulty": "medium"
    },
    {
        "question": "Are there any export control restrictions in The Munt NDA?",
        "expected_sources": ["12032018_NDA_The%20Munt_EN.txt"],
        "clause_types": ["export_control", "compliance"],
        "corpus_type": "contractnli",
        "difficulty": "hard"
    },
    {
        "question": "What independent development rights exist in The Munt NDA?",
        "expected_sources": ["12032018_NDA_The%20Munt_EN.txt"],
        "clause_types": ["independent_development", "rights"],
        "corpus_type": "contractnli",
        "difficulty": "hard"
    },
    {
        "question": "How are affiliates treated under The Munt NDA?",
        "expected_sources": ["12032018_NDA_The%20Munt_EN.txt"],
        "clause_types": ["affiliates", "scope"],
        "corpus_type": "contractnli",
        "difficulty": "medium"
    },

    # ========================================================================
    # CONTRACTNLI - NSK NDA for Suppliers
    # ========================================================================
    {
        "question": "What specific obligations do suppliers have under the NSK confidentiality agreement?",
        "expected_sources": ["5-NSK-Confidentiality-Agreement-for-Suppliers.txt"],
        "clause_types": ["supplier_obligations", "confidentiality"],
        "corpus_type": "contractnli",
        "difficulty": "easy"
    },
    {
        "question": "What is the term of the NSK supplier confidentiality agreement?",
        "expected_sources": ["5-NSK-Confidentiality-Agreement-for-Suppliers.txt"],
        "clause_types": ["term", "duration"],
        "corpus_type": "contractnli",
        "difficulty": "medium"
    },
    {
        "question": "Are there any warranty disclaimers in the NSK supplier agreement?",
        "expected_sources": ["5-NSK-Confidentiality-Agreement-for-Suppliers.txt"],
        "clause_types": ["warranty_disclaimer", "liability"],
        "corpus_type": "contractnli",
        "difficulty": "hard"
    },
    {
        "question": "What remedies are available for breach of the NSK confidentiality agreement?",
        "expected_sources": ["5-NSK-Confidentiality-Agreement-for-Suppliers.txt"],
        "clause_types": ["remedies", "breach"],
        "corpus_type": "contractnli",
        "difficulty": "medium"
    },

    # ========================================================================
    # MAUD - Acacia Communications + Cisco Systems Merger
    # ========================================================================
    {
        "question": "Who are the parties involved in the Acacia-Cisco merger agreement?",
        "expected_sources": ["Acacia_Communications_Cisco_Systems.txt"],
        "clause_types": ["parties", "merger_parties"],
        "corpus_type": "maud",
        "difficulty": "easy"
    },
    {
        "question": "What is the effective date of the Acacia-Cisco merger agreement?",
        "expected_sources": ["Acacia_Communications_Cisco_Systems.txt"],
        "clause_types": ["effective_date", "timing"],
        "corpus_type": "maud",
        "difficulty": "easy"
    },
    {
        "question": "What happens to Company Capital Stock upon merger completion?",
        "expected_sources": ["Acacia_Communications_Cisco_Systems.txt"],
        "clause_types": ["capital_stock", "merger_effect"],
        "corpus_type": "maud",
        "difficulty": "hard"
    },
    {
        "question": "What representations does the Company make about its capital structure?",
        "expected_sources": ["Acacia_Communications_Cisco_Systems.txt"],
        "clause_types": ["representations", "capital_structure"],
        "corpus_type": "maud",
        "difficulty": "hard"
    },
    {
        "question": "What are the closing conditions for the Acacia-Cisco merger?",
        "expected_sources": ["Acacia_Communications_Cisco_Systems.txt"],
        "clause_types": ["closing_conditions", "requirements"],
        "corpus_type": "maud",
        "difficulty": "hard"
    },
    {
        "question": "How are Company Options and RSUs treated in the merger?",
        "expected_sources": ["Acacia_Communications_Cisco_Systems.txt"],
        "clause_types": ["equity_treatment", "employee_benefits"],
        "corpus_type": "maud",
        "difficulty": "hard"
    },

    # ========================================================================
    # MAUD - Acceleron Pharma + Merck Merger
    # ========================================================================
    {
        "question": "Who is acquiring Acceleron Pharma in this merger agreement?",
        "expected_sources": ["Acceleron_Pharma_Inc_Merck_Co.txt"],
        "clause_types": ["acquirer", "parties"],
        "corpus_type": "maud",
        "difficulty": "easy"
    },
    {
        "question": "What regulatory approvals are required for the Acceleron-Merck merger?",
        "expected_sources": ["Acceleron_Pharma_Inc_Merck_Co.txt"],
        "clause_types": ["regulatory_approval", "conditions"],
        "corpus_type": "maud",
        "difficulty": "hard"
    },
    {
        "question": "What termination fees apply if the Acceleron-Merck merger fails?",
        "expected_sources": ["Acceleron_Pharma_Inc_Merck_Co.txt"],
        "clause_types": ["termination_fee", "breakup_fee"],
        "corpus_type": "maud",
        "difficulty": "hard"
    },
    {
        "question": "What representations does Acceleron make about its SEC filings?",
        "expected_sources": ["Acceleron_Pharma_Inc_Merck_Co.txt"],
        "clause_types": ["representations", "sec_filings"],
        "corpus_type": "maud",
        "difficulty": "medium"
    },

    # ========================================================================
    # MAUD - Adamas Pharmaceuticals + Supernus Pharmaceuticals Merger
    # ========================================================================
    {
        "question": "What is the merger consideration for Adamas shareholders?",
        "expected_sources": ["Adamas_Pharmaceuticals_Supernus_Pharmaceuticals.txt"],
        "clause_types": ["merger_consideration", "payment"],
        "corpus_type": "maud",
        "difficulty": "medium"
    },
    {
        "question": "Who are the parties in the Adamas-Supernus merger agreement?",
        "expected_sources": ["Adamas_Pharmaceuticals_Supernus_Pharmaceuticals.txt"],
        "clause_types": ["parties", "merger_parties"],
        "corpus_type": "maud",
        "difficulty": "easy"
    },
    {
        "question": "What covenants restrict Adamas's business operations before closing?",
        "expected_sources": ["Adamas_Pharmaceuticals_Supernus_Pharmaceuticals.txt"],
        "clause_types": ["covenants", "operating_restrictions"],
        "corpus_type": "maud",
        "difficulty": "hard"
    },
    {
        "question": "What employee benefit arrangements are specified in the Adamas merger?",
        "expected_sources": ["Adamas_Pharmaceuticals_Supernus_Pharmaceuticals.txt"],
        "clause_types": ["employee_benefits", "compensation"],
        "corpus_type": "maud",
        "difficulty": "hard"
    },

    # ========================================================================
    # CROSS-CORPUS COMPARISON QUESTIONS
    # ========================================================================
    {
        "question": "Compare confidentiality periods across all NDAs in the dataset.",
        "expected_sources": [
            "01_Bosch-Automotive-Service-Solutions-Mutual-Non-Disclosure-Agreement-7-12-17.txt",
            "12032018_NDA_The%20Munt_EN.txt",
            "5-NSK-Confidentiality-Agreement-for-Suppliers.txt"
        ],
        "clause_types": ["confidentiality_period", "term"],
        "corpus_type": "all",
        "difficulty": "hard"
    },
    {
        "question": "Find all governing law clauses across different contract types.",
        "expected_sources": [
            "01_Bosch-Automotive-Service-Solutions-Mutual-Non-Disclosure-Agreement-7-12-17.txt",
            "2ThemartComInc_19990826_10-12G_EX-10.10_6700288_EX-10.10_Co-Branding Agreement_ Agency Agreement.txt",
            "ABILITYINC_06_15_2020-EX-4.25-SERVICES AGREEMENT.txt"
        ],
        "clause_types": ["governing_law", "jurisdiction"],
        "corpus_type": "all",
        "difficulty": "hard"
    },
    {
        "question": "What termination rights exist across commercial agreements vs merger agreements?",
        "expected_sources": [
            "2ThemartComInc_19990826_10-12G_EX-10.10_6700288_EX-10.10_Co-Branding Agreement_ Agency Agreement.txt",
            "ABILITYINC_06_15_2020-EX-4.25-SERVICES AGREEMENT.txt",
            "Acacia_Communications_Cisco_Systems.txt",
            "Acceleron_Pharma_Inc_Merck_Co.txt"
        ],
        "clause_types": ["termination"],
        "corpus_type": "all",
        "difficulty": "hard"
    },
    {
        "question": "Compare indemnification provisions in services agreements vs joint ventures.",
        "expected_sources": [
            "ABILITYINC_06_15_2020-EX-4.25-SERVICES AGREEMENT.txt",
            "ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT VENTURE AGREEMENT.txt"
        ],
        "clause_types": ["indemnification"],
        "corpus_type": "cuad",
        "difficulty": "hard"
    },
    {
        "question": "Find all intellectual property license grants across different contract types.",
        "expected_sources": [
            "2ThemartComInc_19990826_10-12G_EX-10.10_6700288_EX-10.10_Co-Branding Agreement_ Agency Agreement.txt",
            "ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT VENTURE AGREEMENT.txt"
        ],
        "clause_types": ["ip_license", "ip_ownership"],
        "corpus_type": "cuad",
        "difficulty": "hard"
    },
]


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def load_chunk_metadata() -> Dict[str, Dict]:
    """Load chunk metadata to map chunks to source documents."""
    metadata_map = {}
    chunks_dir = Path("data/chunks")

    for corpus_type in ["cuad", "contractnli", "maud"]:
        corpus_path = chunks_dir / corpus_type
        if corpus_path.exists():
            for chunk_file in corpus_path.glob("*.json"):
                try:
                    with open(chunk_file, 'r') as f:
                        chunk_data = json.load(f)

                    # Extract document ID and source file
                    doc_id = chunk_file.stem.replace("_chunks", "")
                    metadata = chunk_data.get("metadata", {})
                    source_file = metadata.get("source_file", chunk_file.name)

                    # Store mapping: chunk_id -> metadata
                    metadata_map[chunk_file.stem] = {
                        "source_file": source_file,
                        "corpus_type": corpus_type,
                        "doc_id": doc_id,
                        "metadata": metadata
                    }
                except Exception as e:
                    print(f"Warning: Could not load {chunk_file}: {e}")

    return metadata_map


def check_relevance(retrieved_chunks: List[Dict], expected_sources: List[str]) -> bool:
    """Check if any retrieved chunk comes from an expected source document."""
    for chunk in retrieved_chunks:
        chunk_metadata = chunk.get("metadata", {})
        source_file = chunk_metadata.get("source_file", "")

        # Check if source file matches any expected source
        for expected in expected_sources:
            if expected in source_file or source_file in expected:
                return True

    return False


def compute_recall_at_k(retrieved_chunks: List[Dict], expected_sources: List[str], k: int) -> float:
    """
    Compute Recall@K: proportion of expected sources found in top-k results.

    Returns 1.0 if at least one expected source is in top-k, 0.0 otherwise.
    For multi-document questions, returns fraction of expected sources found.
    """
    top_k_chunks = retrieved_chunks[:k]

    found_sources = set()
    for chunk in top_k_chunks:
        chunk_metadata = chunk.get("metadata", {})
        source_file = chunk_metadata.get("source_file", "")

        for expected in expected_sources:
            if expected in source_file or source_file in expected:
                found_sources.add(expected)

    if len(expected_sources) == 0:
        return 0.0

    return len(found_sources) / len(expected_sources)


def run_evaluation(rs: RetrievalService, questions: List[Dict],
                   top_ks: List[int] = [1, 3, 5, 10],
                   verbose: bool = False) -> Dict:
    """Run evaluation on all questions and compute metrics."""

    results = {
        "questions": [],
        "aggregate_metrics": {k: {"total_recall": 0.0, "count": 0} for k in top_ks},
        "by_corpus": defaultdict(lambda: {k: {"total_recall": 0.0, "count": 0} for k in top_ks}),
        "by_difficulty": defaultdict(lambda: {k: {"total_recall": 0.0, "count": 0} for k in top_ks}),
    }

    print(f"\n{'='*80}")
    print(f"EVALUATING {len(questions)} QUESTIONS")
    print(f"{'='*80}\n")

    for idx, q in enumerate(questions, 1):
        question = q["question"]
        expected_sources = q["expected_sources"]
        corpus_type = q.get("corpus_type", "all")
        difficulty = q.get("difficulty", "medium")

        if verbose:
            print(f"\n[{idx}/{len(questions)}] Question: {question}")
            print(f"  Expected sources: {expected_sources}")
            print(f"  Corpus: {corpus_type}, Difficulty: {difficulty}")

        # Run retrieval
        try:
            retrieved = rs.search(query=question, top_k=max(top_ks))

            # Compute recall at each k
            recalls = {}
            for k in top_ks:
                recall = compute_recall_at_k(retrieved, expected_sources, k)
                recalls[k] = recall

                # Update aggregate metrics
                results["aggregate_metrics"][k]["total_recall"] += recall
                results["aggregate_metrics"][k]["count"] += 1

                # Update by-corpus metrics
                results["by_corpus"][corpus_type][k]["total_recall"] += recall
                results["by_corpus"][corpus_type][k]["count"] += 1

                # Update by-difficulty metrics
                results["by_difficulty"][difficulty][k]["total_recall"] += recall
                results["by_difficulty"][difficulty][k]["count"] += 1

            # Store individual result
            result_entry = {
                "question": question,
                "expected_sources": expected_sources,
                "corpus_type": corpus_type,
                "difficulty": difficulty,
                "recalls": recalls,
                "retrieved_count": len(retrieved),
                "success": any(recalls.values())
            }
            results["questions"].append(result_entry)

            if verbose:
                print(f"  Recall@1: {recalls[1]:.2f}, @3: {recalls[3]:.2f}, @5: {recalls[5]:.2f}, @10: {recalls[10]:.2f}")
                if retrieved:
                    top_source = retrieved[0].get("metadata", {}).get("source_file", "Unknown")
                    print(f"  Top result from: {top_source}")
                else:
                    print(f"  ⚠ No results returned!")

        except Exception as e:
            print(f"  ERROR processing question: {e}")
            result_entry = {
                "question": question,
                "expected_sources": expected_sources,
                "error": str(e),
                "recalls": {k: 0.0 for k in top_ks},
                "success": False
            }
            results["questions"].append(result_entry)

    # Compute averages
    for k in top_ks:
        agg = results["aggregate_metrics"][k]
        if agg["count"] > 0:
            agg["average_recall"] = agg["total_recall"] / agg["count"]
        else:
            agg["average_recall"] = 0.0

    for corpus_type in results["by_corpus"]:
        for k in top_ks:
            agg = results["by_corpus"][corpus_type][k]
            if agg["count"] > 0:
                agg["average_recall"] = agg["total_recall"] / agg["count"]

    for difficulty in results["by_difficulty"]:
        for k in top_ks:
            agg = results["by_difficulty"][difficulty][k]
            if agg["count"] > 0:
                agg["average_recall"] = agg["total_recall"] / agg["count"]

    return results


def print_report(results: Dict, top_ks: List[int]):
    """Print detailed evaluation report."""

    print("\n" + "="*80)
    print("EVALUATION REPORT")
    print("="*80)

    # Overall metrics
    print("\n📊 OVERALL METRICS")
    print("-" * 80)
    print(f"{'Metric':<20} {'Value':>15}")
    print("-" * 80)

    total_questions = len(results["questions"])
    successful = sum(1 for q in results["questions"] if q.get("success", False))
    print(f"{'Total Questions':<20} {total_questions:>15}")
    print(f"{'Successful (>0 recall)':<20} {successful:>15}")
    print(f"{'Success Rate':<20} {(successful/total_questions*100):>14.1f}%")

    print(f"\n{'Recall@K':<20} {'Average':>15}")
    print("-" * 80)
    for k in top_ks:
        avg_recall = results["aggregate_metrics"][k]["average_recall"]
        print(f"{'Recall@' + str(k):<20} {avg_recall:>15.3f}")

    # By corpus
    print("\n📁 METRICS BY CORPUS")
    print("-" * 80)
    print(f"{'Corpus':<20} {'Count':>8} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8}")
    print("-" * 80)

    for corpus_type, metrics in sorted(results["by_corpus"].items()):
        count = metrics[1]["count"]
        r1 = metrics[1]["average_recall"]
        r3 = metrics[3]["average_recall"]
        r5 = metrics[5]["average_recall"]
        r10 = metrics[10]["average_recall"]
        print(f"{corpus_type:<20} {count:>8} {r1:>8.3f} {r3:>8.3f} {r5:>8.3f} {r10:>8.3f}")

    # By difficulty
    print("\n📈 METRICS BY DIFFICULTY")
    print("-" * 80)
    print(f"{'Difficulty':<20} {'Count':>8} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8}")
    print("-" * 80)

    for difficulty, metrics in sorted(results["by_difficulty"].items()):
        count = metrics[1]["count"]
        r1 = metrics[1]["average_recall"]
        r3 = metrics[3]["average_recall"]
        r5 = metrics[5]["average_recall"]
        r10 = metrics[10]["average_recall"]
        print(f"{difficulty:<20} {count:>8} {r1:>8.3f} {r3:>8.3f} {r5:>8.3f} {r10:>8.3f}")

    # Failed questions
    failed = [q for q in results["questions"] if not q.get("success", False)]
    if failed:
        print(f"\n⚠️  FAILED QUESTIONS ({len(failed)})")
        print("-" * 80)
        for i, q in enumerate(failed[:10], 1):  # Show first 10
            print(f"{i}. {q['question'][:80]}...")
            print(f"   Expected: {q['expected_sources']}")
        if len(failed) > 10:
            print(f"   ... and {len(failed) - 10} more")

    print("\n" + "="*80)


def save_results(results: Dict, output_path: str = "evaluation_results.json"):
    """Save evaluation results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Results saved to: {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Legal RAG on corpus_lite")
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5, 10],
                        help="K values for Recall@K (default: 1 3 5 10)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output for each question")
    parser.add_argument("--output", "-o", type=str, default="evaluation_results.json",
                        help="Output file path for results")

    args = parser.parse_args()

    # Initialize retrieval service
    print("🔍 Initializing Retrieval Service...")
    try:
        rs = RetrievalService()
        print("✓ Retrieval Service initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Retrieval Service: {e}")
        print("\nMake sure you have run ingestion first:")
        print("  python src/ingestion_service.py --corpus all --force")
        sys.exit(1)

    # Run evaluation
    results = run_evaluation(
        rs=rs,
        questions=GROUND_TRUTH_QUESTIONS,
        top_ks=args.top_k,
        verbose=args.verbose
    )

    # Print report
    print_report(results, args.top_k)

    # Save results
    save_results(results, args.output)

    # Return exit code based on success rate
    total = len(results["questions"])
    successful = sum(1 for q in results["questions"] if q.get("success", False))
    success_rate = successful / total if total > 0 else 0

    if success_rate < 0.5:
        print(f"\n⚠️  Warning: Success rate ({success_rate:.1%}) is below 50%")
        print("Consider re-indexing with better chunking or metadata extraction.")

    return 0 if success_rate > 0.7 else 1


if __name__ == "__main__":
    sys.exit(main())