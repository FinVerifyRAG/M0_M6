from typing import List
from common.schemas import Atom, GeneratedAnswer
from atoms.llm_extract import LLMAtomExtractor
from atoms.regex_extract import regex_extract, find_span
from common.llm_client import LLMClient
import uuid
import logging

logger = logging.getLogger("atoms.splitter")

class AtomSplitter:
    """
    Two-pass atom extraction pipeline:
    1. Fast Regex pass: catches RATE, THRESHOLD, SECTION, DATE patterns immediately
    2. LLM pass: semantic extraction with full claims and ENTITY/APPLICABILITY atoms
    
    The two result sets are merged and deduplicated by span/text.
    """

    def __init__(self, llm: LLMClient, prompt_path: str = "atoms/prompts/extract_v1.txt"):
        self.llm_extractor = LLMAtomExtractor(llm=llm, prompt_path=prompt_path)

    def split(self, answer: GeneratedAnswer) -> List[Atom]:
        answer_text = answer.answer_text

        # Short-circuit: "Not found in evidence" yields no atoms
        if "not found in evidence" in answer_text.lower():
            return []

        # --- Pass 1: Fast Regex ---
        regex_hits = regex_extract(answer_text)
        regex_texts = {h["text"] for h in regex_hits}

        # --- Pass 2: LLM Extraction ---
        llm_atoms = self.llm_extractor.extract(answer_text)

        # --- Merge: LLM takes priority (has claims); regex fills gaps ---
        llm_texts = {a.text for a in llm_atoms}
        merged_atoms = list(llm_atoms)

        for hit in regex_hits:
            if hit["text"] not in llm_texts:
                # LLM missed this regex hit; create a minimal atom
                merged_atoms.append(Atom(
                    atom_id=f"atom_{uuid.uuid4().hex[:8]}",
                    type=hit["type"],
                    text=hit["text"],
                    claim=f"The answer states '{hit['text']}'.",
                    cited_chunk=None,
                    span=hit["span"]
                ))
                logger.debug(f"Regex rescued atom: {hit['text']} ({hit['type']})")

        # Sort by position in text for readability
        merged_atoms.sort(key=lambda a: a.span[0] if a.span else 999999)

        return merged_atoms
