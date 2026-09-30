import json
import uuid
import logging
from typing import List, Optional
from common.schemas import Atom, ATOM_TYPES
from common.llm_client import LLMClient
from atoms.regex_extract import find_span

logger = logging.getLogger("atoms.llm_extract")

class LLMAtomExtractor:
    """
    LLM-powered atom extractor using the extract_v1 prompt.
    The LLM is prompted to return a JSON list with type/text/claim/cited_chunk.
    A fast regex pass is run first; the LLM fills in claims and catches what regex misses.
    """

    def __init__(self, llm: LLMClient, prompt_path: str = "atoms/prompts/extract_v1.txt"):
        self.llm = llm
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read().strip()

    def extract(self, answer_text: str) -> List[Atom]:
        """
        Extract atoms from a generated answer using the LLM.
        Returns a list of Atom objects.
        """
        # Short-circuit: if the model said "not found", return nothing
        if "not found in evidence" in answer_text.lower():
            return []

        user_prompt = f"ANSWER:\n{answer_text}"

        response = self.llm.chat(
            user=user_prompt,
            system=self.system_prompt,
            temperature=0.0,
            logprobs=False
        )

        raw_text = response["text"].strip()

        # Strip code fences if model ignores the instruction
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        try:
            raw_atoms = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.warning(f"LLM returned invalid JSON for atom extraction. Raw: {raw_text[:200]}")
            return []

        atoms = []
        seen_texts = set()

        for i, item in enumerate(raw_atoms):
            atom_type = item.get("type", "").strip().upper()
            text = item.get("text", "").strip()
            claim = item.get("claim", "").strip()
            cited_chunk = item.get("cited_chunk") or None

            # Validate atom type first
            if atom_type not in ATOM_TYPES:
                logger.warning(f"Unknown atom type '{atom_type}' — skipping.")
                continue

            # Skip empty text
            if not text:
                continue

            # Deduplicate by (type, text)
            key = (atom_type, text)
            if key in seen_texts:
                continue
            seen_texts.add(key)

            # Align span: find exact occurrence in the answer
            span = find_span(answer_text, text)

            atoms.append(Atom(
                atom_id=f"atom_{uuid.uuid4().hex[:8]}",
                type=atom_type,
                text=text,
                claim=claim,
                cited_chunk=cited_chunk,
                span=span
            ))

        logger.info(f"Extracted {len(atoms)} atoms from answer ({len(answer_text)} chars).")
        return atoms
