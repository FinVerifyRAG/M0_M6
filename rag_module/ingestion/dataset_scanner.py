import os
import re
from pathlib import Path
from typing import List, Dict, Any

class DatasetScanner:
    """Discovers, categorizes, and extracts high-level metadata from regulatory PDF files."""

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)

    def scan_documents(self, max_docs: int = None) -> List[Dict[str, Any]]:
        """Walks dataset directory and returns structured document descriptors."""
        documents = []

        if not self.root_dir.exists():
            print(f"[Warning] Dataset directory not found: {self.root_dir}")
            return documents

        for root, _, files in os.walk(self.root_dir):
            pdf_files = [f for f in files if f.lower().endswith(".pdf")]
            for filename in pdf_files:
                file_path = Path(root) / filename
                metadata = self._extract_file_metadata(file_path)
                documents.append(metadata)

                if max_docs and len(documents) >= max_docs:
                    return documents

        print(f"[Ingestion] Scanned {len(documents)} PDF documents from {self.root_dir}")
        return documents

    def _extract_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        path_str = str(file_path)
        filename = file_path.name
        rel_path = file_path.relative_to(self.root_dir)

        # Determine Issuer (RBI vs SEBI)
        if "sebi" in path_str.lower():
            issuer = "SEBI"
        else:
            issuer = "RBI"

        # Determine Category
        category = "Guideline"
        lower_path = path_str.lower()
        if "master_circular" in lower_path or "master circular" in lower_path:
            category = "Master Circular"
        elif "master_direction" in lower_path or "master direction" in lower_path:
            category = "Master Direction"
        elif "act" in lower_path or "acts" in lower_path:
            category = "Act"
        elif "regulation" in lower_path or "regulations" in lower_path:
            category = "Regulation"
        elif "rule" in lower_path or "rules" in lower_path:
            category = "Rule"
        elif "gazette" in lower_path:
            category = "Gazette Notification"
        elif "circular" in lower_path:
            category = "Circular"

        # Extract Year from filename or parent directories
        year_match = re.search(r'(19\d{2}|20\d{2})', path_str)
        year = year_match.group(1) if year_match else "Unknown"

        # Document Title Cleaning
        doc_title = file_path.stem
        # Clean underscores/dashes for human readability
        clean_title = re.sub(r'[_\-]+', ' ', doc_title).strip()

        return {
            "doc_id": f"{issuer}_{doc_title}".replace(" ", "_"),
            "file_path": str(file_path),
            "filename": filename,
            "relative_path": str(rel_path),
            "doc_title": clean_title,
            "issuer": issuer,
            "category": category,
            "year": year
        }
