"""
Enhanced Metadata Extraction Module for Legal RAG System.
Extracts structured metadata from legal documents to support advanced filtering queries.
Supports CUAD, ContractNLI, and MAUD corpus types.
"""

import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ContractMetadata:
    """Standardized metadata structure for all contract types."""
    # Core identification
    document_id: str
    file_name: str
    corpus_type: str
    
    # Party information
    party_1: Optional[str] = None
    party_2: Optional[str] = None
    party_1_type: Optional[str] = None  # e.g., "licensor", "acquirer", "disclosing_party"
    party_2_type: Optional[str] = None  # e.g., "licensee", "target", "receiving_party"
    
    # Contract classification
    contract_type: Optional[str] = None  # e.g., "License Agreement", "NDA", "Merger Agreement"
    contract_subtype: Optional[str] = None  # e.g., "SaaS", "Distribution", "Asset Purchase"
    
    # Temporal information
    effective_date: Optional[str] = None
    expiration_date: Optional[str] = None
    execution_date: Optional[str] = None
    
    # Legal jurisdiction
    governing_law: Optional[str] = None
    jurisdiction: Optional[str] = None
    
    # Industry/sector classification
    industry_sector: Optional[str] = None
    target_industry: Optional[str] = None  # For M&A deals
    acquirer_industry: Optional[str] = None  # For M&A deals
    
    # Financial terms (extracted mentions)
    has_liability_cap: bool = False
    liability_cap_amount: Optional[str] = None
    has_termination_fee: bool = False
    termination_fee_amount: Optional[str] = None
    
    # Key clause indicators (multi-label)
    clause_types_present: List[str] = None
    
    # Special fields for MAUD
    acquirer: Optional[str] = None
    target: Optional[str] = None
    deal_type: Optional[str] = None  # e.g., "Merger", "Asset Purchase", "Stock Purchase"
    
    # Special fields for ContractNLI
    has_injunctive_relief: bool = False
    confidentiality_period: Optional[str] = None
    mutual_nondisclosure: bool = False
    
    # Additional metadata
    word_count: int = 0
    section_count: int = 0
    
    def __post_init__(self):
        if self.clause_types_present is None:
            self.clause_types_present = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def to_filter_dict(self) -> Dict[str, Any]:
        """
        Convert to a filter-friendly dictionary for query processing.
        Only includes non-None values for efficient filtering.
        """
        result = {}
        for key, value in self.to_dict().items():
            if value is not None and value != []:
                result[key] = value
        return result


class MetadataExtractor:
    """
    Extracts structured metadata from legal documents.
    Uses regex patterns and rule-based extraction for reliability.
    """
    
    # Common party name patterns
    PARTY_PATTERNS = [
        r'between\s+([A-Z][A-Za-z0-9\s&,\.]+(?:Inc\.|LLC|Ltd\.|Corporation|Corp\.|Company))',
        r'by\s+and\s+between\s+([A-Z][A-Za-z0-9\s&,\.]+(?:Inc\.|LLC|Ltd\.|Corporation|Corp\.))',
        r'Party:\s*([A-Z][A-Za-z0-9\s&,\.]+(?:Inc\.|LLC|Ltd\.|Corp\.))',
        r'^([A-Z][A-Za-z0-9\s&]+(?:Inc\.|LLC|Ltd\.|Corporation|Corp\.|Company)),?\s+a\s+(?:corporation|company|LLC)',
    ]
    
    # Governing law patterns
    GOVERNING_LAW_PATTERNS = [
        r'governed\s+by\s+the\s+laws\s+of\s+([A-Z][a-zA-Z\s]+?)(?:\s|$|,|\.)',
        r'governing\s+law[:\s]+([A-Z][a-zA-Z\s]+?)(?:\s|$|,|\.)',
        r'shall\s+be\s+governed\s+by\s+and\s+construed\s+in\s+accordance\s+with\s+the\s+laws\s+of\s+([A-Z][a-zA-Z\s]+?)(?:\s|$|,|\.)',
        r'jurisdiction\s+of\s+([A-Z][a-zA-Z\s]+?)(?:\s|$|,|\.)',
    ]
    
    # Date patterns
    DATE_PATTERNS = [
        r'(?:effective|executed|dated)\s+(?:as\s+of\s+)?(\w+\s+\d{1,2},?\s+\d{4})',
        r'(\w+\s+\d{1,2},?\s+\d{4})',
        r'(\d{1,2}/\d{1,2}/\d{4})',
    ]
    
    # Contract type keywords
    CONTRACT_TYPE_KEYWORDS = {
        'License Agreement': ['license', 'licensed', 'licensing'],
        'Non-Disclosure Agreement': ['non-disclosure', 'confidentiality', 'NDA', 'proprietary information'],
        'Merger Agreement': ['merger', 'merge', 'combination'],
        'Purchase Agreement': ['purchase', 'sale', 'acquisition'],
        'Services Agreement': ['services', 'service provider', 'consulting'],
        'Distribution Agreement': ['distribution', 'distributor', 'reseller'],
        'Employment Agreement': ['employment', 'employee', 'hire'],
        'Lease Agreement': ['lease', 'leased', 'lessor', 'lessee'],
        'Settlement Agreement': ['settlement', 'release', 'dispute'],
    }
    
    # Clause type patterns
    CLAUSE_PATTERNS = {
        'non_compete': [r'non[- ]compete', r'covenant\s+not\s+to\s+compete', r'restricted\s+activities'],
        'non_solicit': [r'non[- ]solicitation', r'solicitation\s+of\s+employees', r'no[- ]hire'],
        'confidentiality': [r'confidentialit', r'non[- ]disclosure', r'proprietary\s+information'],
        'indemnification': [r'indemnif', r'hold\s+harmless', r'indemnity'],
        'termination': [r'termination', r'terminate', 'expiration'],
        'governing_law': [r'governing\s+law', r'choice\s+of\s+law', r'applicable\s+law'],
        'liability_cap': [r'limitation\s+of\s+liability', r'liability\s+cap', r'maximum\s+liability'],
        'force_majeure': [r'force\s+majeure', r'act\s+of\s+god', r'unforeseeable'],
        'assignment': [r'assignment', r'assign', r'transfer\s+of\s+rights'],
        'intellectual_property': [r'intellectual\s+property', r'IP\s+rights', r'patent', r'copyright', r'trademark'],
        'warranties': [r'warrant', r'representation', r'warranty'],
        'dispute_resolution': [r'arbitration', r'dispute\s+resolution', r'mediation'],
        'insurance': [r'insurance', r'coverage', r'policy'],
        'payment_terms': [r'payment', r'fee', r'compensation', r'consideration'],
    }
    
    # Industry keywords
    INDUSTRY_KEYWORDS = {
        'Technology': ['software', 'technology', 'IT', 'computer', 'digital', 'SaaS', 'cloud'],
        'Healthcare': ['healthcare', 'medical', 'pharmaceutical', 'biotech', 'hospital', 'clinical'],
        'Financial Services': ['financial', 'bank', 'insurance', 'investment', 'securities', 'fintech'],
        'Manufacturing': ['manufacturing', 'production', 'factory', 'industrial'],
        'Retail': ['retail', 'consumer', 'e-commerce', 'store', 'merchant'],
        'Energy': ['energy', 'oil', 'gas', 'renewable', 'utility', 'power'],
        'Telecommunications': ['telecom', 'communication', 'wireless', 'network', 'broadband'],
        'Real Estate': ['real estate', 'property', 'leasing', 'construction', 'development'],
        'Transportation': ['transportation', 'logistics', 'shipping', 'airline', 'automotive'],
        'Media & Entertainment': ['media', 'entertainment', 'broadcasting', 'publishing', 'content'],
    }
    
    def __init__(self):
        """Initialize the metadata extractor."""
        self.compiled_patterns = {
            'party': [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in self.PARTY_PATTERNS],
            'governing_law': [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in self.GOVERNING_LAW_PATTERNS],
            'date': [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in self.DATE_PATTERNS],
        }
        self.clause_compiled = {
            clause_type: [re.compile(p, re.IGNORECASE) for p in patterns]
            for clause_type, patterns in self.CLAUSE_PATTERNS.items()
        }
    
    def extract_parties(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract party names from the document.
        
        Returns:
            Tuple of (party_1, party_2) or (None, None) if not found
        """
        parties = []
        
        for pattern in self.compiled_patterns['party']:
            matches = pattern.findall(text[:5000])  # Search first 5000 chars
            for match in matches:
                party_name = match.strip()
                # Clean up party name
                party_name = re.sub(r'\s+', ' ', party_name)
                if len(party_name) > 3 and party_name not in parties:
                    parties.append(party_name)
                if len(parties) >= 2:
                    break
            if len(parties) >= 2:
                break
        
        if len(parties) >= 2:
            return parties[0], parties[1]
        elif len(parties) == 1:
            return parties[0], None
        return None, None
    
    def extract_governing_law(self, text: str) -> Optional[str]:
        """Extract governing law/jurisdiction from the document."""
        for pattern in self.compiled_patterns['governing_law']:
            matches = pattern.findall(text)
            if matches:
                law = matches[0].strip()
                # Clean up
                law = re.sub(r'\s+', ' ', law)
                if len(law) > 2 and len(law) < 100:
                    return law
        return None
    
    def extract_dates(self, text: str) -> Dict[str, Optional[str]]:
        """Extract various dates from the document."""
        result = {
            'effective_date': None,
            'execution_date': None,
            'expiration_date': None
        }
        
        for pattern in self.compiled_patterns['date']:
            matches = pattern.findall(text[:3000])  # Search first 3000 chars
            for match in matches:
                date_str = match.strip()
                if 'effective' in text[text.find(date_str)-50:text.find(date_str)].lower():
                    result['effective_date'] = date_str
                elif 'execut' in text[text.find(date_str)-50:text.find(date_str)].lower():
                    result['execution_date'] = date_str
                elif 'expir' in text[text.find(date_str)-50:text.find(date_str)].lower():
                    result['expiration_date'] = date_str
                elif result['effective_date'] is None:
                    result['effective_date'] = date_str
        
        return result
    
    def classify_contract_type(self, text: str, file_name: str = "") -> Tuple[Optional[str], Optional[str]]:
        """
        Classify the contract type based on content and filename.
        
        Returns:
            Tuple of (contract_type, contract_subtype)
        """
        text_lower = text[:5000].lower()
        file_name_lower = file_name.lower()
        
        # Check filename first
        for contract_type, keywords in self.CONTRACT_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in file_name_lower:
                    return contract_type, None
        
        # Check content
        type_scores = {}
        for contract_type, keywords in self.CONTRACT_TYPE_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                type_scores[contract_type] = score
        
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            # Determine subtype
            subtype = None
            if 'saas' in text_lower or 'cloud' in text_lower:
                subtype = 'SaaS'
            elif 'distribution' in text_lower:
                subtype = 'Distribution'
            
            return best_type, subtype
        
        return None, None
    
    def detect_clause_types(self, text: str) -> List[str]:
        """Detect which clause types are present in the document."""
        detected_clauses = []
        text_lower = text.lower()
        
        for clause_type, patterns in self.clause_compiled.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    detected_clauses.append(clause_type)
                    break
        
        return detected_clauses
    
    def classify_industry(self, text: str) -> Optional[str]:
        """Classify the industry sector based on document content."""
        text_lower = text[:5000].lower()
        
        industry_scores = {}
        for industry, keywords in self.INDUSTRY_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                industry_scores[industry] = score
        
        if industry_scores:
            return max(industry_scores, key=industry_scores.get)
        
        return None
    
    def extract_financial_terms(self, text: str) -> Dict[str, Any]:
        """Extract financial terms like liability caps and termination fees."""
        result = {
            'has_liability_cap': False,
            'liability_cap_amount': None,
            'has_termination_fee': False,
            'termination_fee_amount': None
        }
        
        text_lower = text.lower()
        
        # Check for liability cap
        if 'limitation of liability' in text_lower or 'liability cap' in text_lower:
            result['has_liability_cap'] = True
            # Try to extract amount
            amount_pattern = r'\$[\d,]+(?:\s*(?:million|billion|thousand))?'
            amounts = re.findall(amount_pattern, text)
            if amounts:
                result['liability_cap_amount'] = amounts[0]
        
        # Check for termination fee
        if 'termination fee' in text_lower or 'breakup fee' in text_lower:
            result['has_termination_fee'] = True
            amount_pattern = r'\$[\d,]+(?:\s*(?:million|billion|thousand))?'
            amounts = re.findall(amount_pattern, text)
            if amounts:
                result['termination_fee_amount'] = amounts[0]
        
        return result
    
    def extract_maud_specific(self, text: str, file_name: str = "") -> Dict[str, Any]:
        """Extract MAUD-specific metadata for M&A agreements."""
        result = {
            'acquirer': None,
            'target': None,
            'deal_type': None
        }
        
        # Try to extract from filename (format: CompanyA_CompanyB.txt)
        if '_' in file_name and 'maud' in file_name.lower():
            parts = file_name.replace('.txt', '').split('_')
            if len(parts) >= 2:
                result['acquirer'] = parts[0].strip()
                result['target'] = parts[1].strip()
        
        # Detect deal type
        text_lower = text.lower()
        if 'merger' in text_lower:
            result['deal_type'] = 'Merger'
        elif 'asset purchase' in text_lower:
            result['deal_type'] = 'Asset Purchase'
        elif 'stock purchase' in text_lower or 'share purchase' in text_lower:
            result['deal_type'] = 'Stock Purchase'
        
        return result
    
    def extract_contractnli_specific(self, text: str) -> Dict[str, Any]:
        """Extract ContractNLI-specific metadata for NDAs."""
        result = {
            'has_injunctive_relief': False,
            'confidentiality_period': None,
            'mutual_nondisclosure': False
        }
        
        text_lower = text.lower()
        
        # Check for injunctive relief
        if 'injunctive' in text_lower or 'irreparable harm' in text_lower:
            result['has_injunctive_relief'] = True
        
        # Try to extract confidentiality period
        period_pattern = r'(?:confidentialit|non[- ]disclosure)\s+(?:period|term|shall\s+continue\s+for)\s+(\d+\s+(?:years?|months?|days?))'
        matches = re.findall(period_pattern, text_lower)
        if matches:
            result['confidentiality_period'] = matches[0]
        
        # Check if mutual
        if 'mutual' in text_lower and ('nondisclosure' in text_lower or 'confidentiality' in text_lower):
            result['mutual_nondisclosure'] = True
        
        return result
    
    def extract_metadata(self, text: str, file_path: Path, 
                        corpus_type: str, document_id: str) -> ContractMetadata:
        """
        Extract comprehensive metadata from a legal document.
        
        Args:
            text: Full document text
            file_path: Path to the document
            corpus_type: Type of corpus (cuad, maud, contractnli)
            document_id: Unique document identifier
            
        Returns:
            ContractMetadata object with extracted information
        """
        file_name = file_path.name
        
        # Initialize metadata
        metadata = ContractMetadata(
            document_id=document_id,
            file_name=file_name,
            corpus_type=corpus_type,
            word_count=len(text.split())
        )
        
        # Extract basic information
        party_1, party_2 = self.extract_parties(text)
        metadata.party_1 = party_1
        metadata.party_2 = party_2
        
        governing_law = self.extract_governing_law(text)
        metadata.governing_law = governing_law
        
        dates = self.extract_dates(text)
        metadata.effective_date = dates['effective_date']
        metadata.execution_date = dates['execution_date']
        metadata.expiration_date = dates['expiration_date']
        
        contract_type, contract_subtype = self.classify_contract_type(text, file_name)
        metadata.contract_type = contract_type
        metadata.contract_subtype = contract_subtype
        
        # Detect clause types
        metadata.clause_types_present = self.detect_clause_types(text)
        
        # Classify industry
        metadata.industry_sector = self.classify_industry(text)
        
        # Extract financial terms
        financial_terms = self.extract_financial_terms(text)
        metadata.has_liability_cap = financial_terms['has_liability_cap']
        metadata.liability_cap_amount = financial_terms['liability_cap_amount']
        metadata.has_termination_fee = financial_terms['has_termination_fee']
        metadata.termination_fee_amount = financial_terms['termination_fee_amount']
        
        # Corpus-specific extraction
        if corpus_type == 'maud':
            maud_data = self.extract_maud_specific(text, file_name)
            metadata.acquirer = maud_data['acquirer']
            metadata.target = maud_data['target']
            metadata.deal_type = maud_data['deal_type']
            metadata.acquirer_industry = self.classify_industry(text[:5000])
            # For M&A, we might want separate target industry detection
            metadata.target_industry = metadata.industry_sector
            
        elif corpus_type == 'contractnli':
            cnli_data = self.extract_contractnli_specific(text)
            metadata.has_injunctive_relief = cnli_data['has_injunctive_relief']
            metadata.confidentiality_period = cnli_data['confidentiality_period']
            metadata.mutual_nondisclosure = cnli_data['mutual_nondisclosure']
            metadata.contract_type = 'Non-Disclosure Agreement'  # Override for ContractNLI
        
        # Count sections
        metadata.section_count = len(re.findall(r'^\d+\.', text, re.MULTILINE))
        
        return metadata


def save_metadata_to_file(metadata: ContractMetadata, output_dir: Path, 
                          document_id: str, format: str = 'json') -> Path:
    """
    Save extracted metadata to a file.
    
    Args:
        metadata: ContractMetadata object
        output_dir: Directory to save metadata
        document_id: Document ID for naming
        format: Output format ('json' or 'pickle')
    
    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        output_file = output_dir / f"{document_id}_metadata.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
    else:
        import pickle
        output_file = output_dir / f"{document_id}_metadata.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(metadata, f)
    
    return output_file


if __name__ == "__main__":
    """Test the metadata extractor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Metadata Extraction")
    parser.add_argument('file_path', type=str, help='Path to test document')
    parser.add_argument('--corpus-type', type=str, default='cuad',
                       choices=['cuad', 'maud', 'contractnli'])
    parser.add_argument('--output-dir', type=str, default='data/metadata_test')
    
    args = parser.parse_args()
    
    extractor = MetadataExtractor()
    
    file_path = Path(args.file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Extracting metadata from: {file_path.name}")
    print(f"Corpus type: {args.corpus_type}")
    print("=" * 80)
    
    metadata = extractor.extract_metadata(
        text=text,
        file_path=file_path,
        corpus_type=args.corpus_type,
        document_id="test_doc_001"
    )
    
    print("\nExtracted Metadata:")
    print(json.dumps(metadata.to_dict(), indent=2))
    
    # Save to file
    output_path = save_metadata_to_file(
        metadata, 
        Path(args.output_dir), 
        "test_doc_001"
    )
    print(f"\nMetadata saved to: {output_path}")
