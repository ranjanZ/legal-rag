"""
Legal RAG Preprocessing Pipeline
Extracts structured metadata from CUAD, ContractNLI, and MAUD documents
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class LegalDocumentPreprocessor:
    """
    Extracts party names, contract types, dates, and clause classifications
    from legal documents to enable advanced filtering in RAG queries.
    """
    
    # Contract type mappings based on filename patterns and content
    CONTRACT_TYPE_PATTERNS = {
        'cuad': [
            (r'SERVICES AGREEMENT', 'Services Agreement'),
            (r'LICENSE.*AGREEMENT', 'License Agreement'),
            (r'DISTRIBUTION.*AGREEMENT', 'Distribution Agreement'),
            (r'NON-COMPETE', 'Non-Compete Agreement'),
            (r'EMPLOYMENT.*AGREEMENT', 'Employment Agreement'),
            (r'LEASE.*AGREEMENT', 'Lease Agreement'),
            (r'PURCHASE.*AGREEMENT', 'Purchase Agreement'),
            (r'SUBCONTRACTING', 'Subcontracting Agreement'),
            (r'CO-BRANDING', 'Co-Branding Agreement'),
            (r'OUTSOURCING', 'Outsourcing Agreement'),
        ],
        'contractnli': [
            (r'NON-DISCLOSURE', 'Non-Disclosure Agreement'),
            (r'CONFIDENTIALITY', 'Confidentiality Agreement'),
            (r'NDA', 'Non-Disclosure Agreement'),
            (r'MUTUAL.*DISCLOSURE', 'Mutual NDA'),
        ],
        'maud': [
            (r'AGREEMENT AND PLAN OF MERGER', 'Merger Agreement'),
            (r'PURCHASE AGREEMENT', 'Purchase Agreement'),
            (r'ACQUISITION', 'Acquisition Agreement'),
            (r'STOCK PURCHASE', 'Stock Purchase Agreement'),
            (r'ASSET PURCHASE', 'Asset Purchase Agreement'),
        ]
    }
    
    def __init__(self):
        self.party_patterns = [
            # Pattern: "BETWEEN Company A AND Company B"
            r'BETWEEN\s+([A-Z][A-Za-z\s\.,&]+?)\s+\(.*?\)\s+AND\s+([A-Z][A-Za-z\s\.,&]+?)\s+\(',
            # Pattern: "by and between X, a company... and Y, a company..."
            r'by and between\s+(.+?),\s+a\s+(?:company|corporation|inc\.|llc)',
            # Pattern for MAUD: "BY AND AMONG\n\nCOMPANY A,\n\nCOMPANY B\n\nAND\n\nCOMPANY C"
            r'BY AND AMONG\s*\n+\s*([A-Z][A-Z\s\.,&]+?),\s*\n+\s*([A-Z][A-Z\s\.,&]+?)\s*\n+\s*AND\s*\n+\s*([A-Z][A-Z\s\.,&]+?)',
        ]
        
        self.date_patterns = [
            r'(\w+\s+\d{1,2},?\s+\d{4})',  # January 14, 2021
            r'(\d{1,2}/\d{1,2}/\d{4})',     # 01/14/2021
            r'(\d{4}-\d{2}-\d{2})',         # 2021-01-14
        ]
        
        self.governing_law_patterns = [
            r'governed by the laws of\s+([^\.]+?)(?:\.|$)',
            r'laws of the State of\s+([^\.]+?)(?:\.|$)',
            r'jurisdiction of\s+([^\.]+?)(?:\.|$)',
        ]
    
    def extract_parties(self, text: str, corpus_type: str) -> Dict[str, Optional[str]]:
        """
        Extract party names from document text.
        Returns different structures based on corpus type.
        """
        parties = {
            'party_1': None,
            'party_2': None,
            'acquirer': None,
            'target': None,
        }
        
        if corpus_type == 'maud':
            # MAUD: Extract acquirer and target from merger agreements
            # Look for "BY AND AMONG ACQUIRER, TARGET, AND MERGER SUB"
            maud_pattern = r'BY AND AMONG\s*\n+\s*([A-Z][A-Z\s\.,&]+?),\s*\n+\s*([A-Z][A-Z\s\.,&]+?)\s*\n+\s*AND\s*\n+\s*([A-Z][A-Z\s\.,&]+?)'
            match = re.search(maud_pattern, text[:2000])  # Search first 2000 chars
            
            if match:
                # Typically: Acquirer, Merger Sub, Target OR Acquirer, Target, Merger Sub
                company_names = [m.strip().strip(',') for m in match.groups()]
                # Heuristic: shortest name is often the merger sub (e.g., "AMARONE ACQUISITION CORP.")
                company_names_sorted = sorted(company_names, key=len)
                parties['merger_sub'] = company_names_sorted[0]
                parties['acquirer'] = company_names_sorted[1] if len(company_names_sorted) > 1 else None
                parties['target'] = company_names_sorted[2] if len(company_names_sorted) > 2 else company_names_sorted[1]
            
            # Alternative: Parse from "AMENDED AND RESTATED AGREEMENT AND PLAN OF MERGER BY AND AMONG..."
            if not parties['acquirer']:
                lines = text.split('\n')[:30]
                companies = []
                for line in lines:
                    line = line.strip().rstrip(',')
                    if len(line) > 5 and line.isupper() and 'INC' in line or 'CORP' in line or 'LLC' in line or 'CO' in line:
                        companies.append(line)
                
                if len(companies) >= 2:
                    parties['acquirer'] = companies[0]
                    parties['target'] = companies[-1]  # Last is usually target
        
        elif corpus_type in ['cuad', 'contractnli']:
            # Extract two parties
            # Try pattern matching first
            for pattern in self.party_patterns[:2]:
                match = re.search(pattern, text[:3000], re.IGNORECASE)
                if match:
                    groups = match.groups()
                    if len(groups) >= 2:
                        parties['party_1'] = groups[0].strip().strip(',')
                        parties['party_2'] = groups[1].strip().strip(',')
                        break
            
            # Fallback: Extract from filename
            if not parties['party_1']:
                # Filename format: COMPANYNAME_DATE-TYPE.txt
                pass  # Will be handled in filename parsing
        
        return parties
    
    def extract_contract_type(self, text: str, filename: str, corpus_type: str) -> str:
        """Determine contract type from content and filename."""
        # Check filename first
        filename_upper = filename.upper()
        
        patterns = self.CONTRACT_TYPE_PATTERNS.get(corpus_type, [])
        for pattern, contract_type in patterns:
            if re.search(pattern, filename_upper):
                return contract_type
        
        # Fall back to content search
        text_sample = text[:5000].upper()
        for pattern, contract_type in patterns:
            if re.search(pattern, text_sample):
                return contract_type
        
        return f"{corpus_type.upper()} Document"
    
    def extract_dates(self, text: str) -> Dict[str, Optional[str]]:
        """Extract contract date, effective date, etc."""
        dates = {
            'contract_date': None,
            'effective_date': None,
        }
        
        # Look for "entered into on DATE" or "Effective Date: DATE"
        entered_match = re.search(r'entered into on\s+(\w+\s+\d{1,2},?\s+\d{4})', text[:2000], re.IGNORECASE)
        if entered_match:
            dates['contract_date'] = entered_match.group(1)
        
        effective_match = re.search(r'(?:Effective Date|effective as of)[:\s]+(\w+\s+\d{1,2},?\s+\d{4})', text[:2000], re.IGNORECASE)
        if effective_match:
            dates['effective_date'] = effective_match.group(1)
        
        return dates
    
    def extract_governing_law(self, text: str) -> Optional[str]:
        """Extract governing law/jurisdiction."""
        for pattern in self.governing_law_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def detect_clause_types(self, text: str) -> Dict[str, bool]:
        """Detect presence of key clause types in document."""
        clause_keywords = {
            'has_non_compete': ['non-compete', 'non compete', 'competitive business', 'competing business'],
            'has_indemnification': ['indemnif', 'hold harmless'],
            'has_liability_cap': ['liability cap', 'limitation of liability', 'maximum liability'],
            'has_force_majeure': ['force majeure', 'act of god', 'unforeseeable'],
            'has_termination_clause': ['termination', 'terminate this agreement'],
            'has_confidentiality': ['confidential', 'non-disclosure', 'proprietary information'],
            'has_ip_ownership': ['intellectual property', 'ip rights', 'ownership of inventions'],
            'has_governing_law': ['governing law', 'choice of law', 'jurisdiction'],
            'has_injunctive_relief': ['injunctive relief', 'irreparable harm'],
        }
        
        text_lower = text.lower()
        results = {}
        
        for clause_type, keywords in clause_keywords.items():
            results[clause_type] = any(kw in text_lower for kw in keywords)
        
        return results
    
    def parse_filename_metadata(self, filename: str, corpus_type: str) -> Dict:
        """Extract metadata from filename patterns."""
        metadata = {}
        
        if corpus_type == 'cuad':
            # Pattern: COMPANY_DATE-EX-X.X-TYPE.txt
            # Example: ABILITYINC_06_15_2020-EX-4.25-SERVICES AGREEMENT.txt
            parts = filename.replace('.txt', '').split('_')
            if len(parts) >= 2:
                metadata['party_1_from_filename'] = parts[0]
                date_part = parts[1] if len(parts) > 1 else None
                if date_part:
                    # Parse date from format like 06_15_2020
                    date_clean = date_part.replace('_', '-')
                    metadata['date_from_filename'] = date_clean
        
        elif corpus_type == 'maud':
            # Pattern: CompanyA_CompanyB.txt
            # Example: Acacia_Communications_Cisco_Systems.txt
            base_name = filename.replace('.txt', '').replace('.pdf||', '')
            companies = base_name.split('_')
            if len(companies) >= 2:
                # Reconstruct company names (they may have underscores)
                # Heuristic: split on capital letters after underscore
                metadata['parties_from_filename'] = base_name.replace('_', ' ')
        
        elif corpus_type == 'contractnli':
            # Pattern: Descriptive_NDA_name.txt
            # Example: 01_Bosch-Automotive-Service-Solutions-Mutual-Non-Disclosure-Agreement-7-12-17.txt
            metadata['descriptive_name'] = filename.replace('.txt', '')
        
        return metadata
    
    def process_document(self, file_path: Path) -> Dict:
        """
        Main processing function: reads document and extracts all metadata.
        """
        # Determine corpus type from path
        corpus_type = file_path.parent.name.lower()
        if 'contractnli' in str(file_path).lower():
            corpus_type = 'contractnli'
        elif 'maud' in str(file_path).lower():
            corpus_type = 'maud'
        elif 'cuad' in str(file_path).lower():
            corpus_type = 'cuad'
        
        # Read document
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        filename = file_path.name
        
        # Extract all metadata
        metadata = {
            'file_name': filename,
            'file_path': str(file_path),
            'corpus_type': corpus_type,
        }
        
        # Add filename-based metadata
        metadata.update(self.parse_filename_metadata(filename, corpus_type))
        
        # Extract parties
        parties = self.extract_parties(text, corpus_type)
        metadata.update(parties)
        
        # Extract contract type
        metadata['contract_type'] = self.extract_contract_type(text, filename, corpus_type)
        
        # Extract dates
        metadata.update(self.extract_dates(text))
        
        # Extract governing law
        metadata['governing_law'] = self.extract_governing_law(text)
        
        # Detect clause types
        metadata.update(self.detect_clause_types(text))
        
        # Estimate industry sector (simple keyword matching)
        metadata['industry_sector'] = self.detect_industry_sector(text)
        
        return metadata
    
    def detect_industry_sector(self, text: str) -> Optional[str]:
        """Detect industry sector from document content."""
        sector_keywords = {
            'Technology': ['software', 'technology', 'computer', 'digital', 'saas', 'cloud'],
            'Healthcare': ['healthcare', 'pharmaceutical', 'medical', 'drug', 'biotech'],
            'Automotive': ['automotive', 'vehicle', 'car', 'auto parts'],
            'Finance': ['financial', 'bank', 'investment', 'securities'],
            'Manufacturing': ['manufacturing', 'factory', 'production'],
            'Retail': ['retail', 'consumer', 'e-commerce'],
            'Energy': ['energy', 'oil', 'gas', 'renewable'],
            'Telecommunications': ['telecom', 'communication', 'network', 'wireless'],
        }
        
        text_lower = text[:10000].lower()
        
        for sector, keywords in sector_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return sector
        
        return None


def main():
    """Example usage of the preprocessor."""
    preprocessor = LegalDocumentPreprocessor()
    
    # Example: Process a CUAD document
    cuad_file = Path('/workspace/data/corpus/cuad/SPARKLINGSPRINGWATERHOLDINGSLTD_07_03_2002-EX-10.13-SOFTWARE LICENSE AND MAINTENANCE AGREEMENT.txt')
    if cuad_file.exists():
        metadata = preprocessor.process_document(cuad_file)
        print("=== CUAD Document Metadata ===")
        print(json.dumps(metadata, indent=2, default=str))
    
    # Example: Process a ContractNLI document
    nli_file = Path('/workspace/data/corpus/contractnli/01_Bosch-Automotive-Service-Solutions-Mutual-Non-Disclosure-Agreement-7-12-17.txt')
    if nli_file.exists():
        metadata = preprocessor.process_document(nli_file)
        print("\n=== ContractNLI Document Metadata ===")
        print(json.dumps(metadata, indent=2, default=str))
    
    # Example: Process a MAUD document
    maud_file = Path('/workspace/data/corpus/maud/Acacia_Communications_Cisco_Systems.txt')
    if maud_file.exists():
        metadata = preprocessor.process_document(maud_file)
        print("\n=== MAUD Document Metadata ===")
        print(json.dumps(metadata, indent=2, default=str))


if __name__ == '__main__':
    main()
