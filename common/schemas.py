from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict, Tuple

class Chunk(BaseModel):
    chunk_id: str
    text: str
    regulator: str
    issue_date: str
    source_url: str
    effective_from: Optional[str] = None
    effective_to: Optional[str] = None
    superseded_by: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "1.0"

class RetrievalResult(BaseModel):
    query: str
    query_date: str
    chunks: List[Chunk]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "1.0"

class GeneratedAnswer(BaseModel):
    query: str
    answer_text: str
    citations: List[str]
    token_logprobs: Optional[List[float]] = None
    model_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "1.0"

ATOM_TYPES = {"RATE", "THRESHOLD", "SECTION", "DATE", "ENTITY", "APPLICABILITY"}

class Atom(BaseModel):
    atom_id: str
    type: str  # RATE, THRESHOLD, SECTION, DATE, ENTITY, APPLICABILITY
    text: str  # Exact substring copied from the answer
    claim: str  # Full self-contained sentence for checking
    cited_chunk: Optional[str] = None  # chunk_id nearest to the fact
    span: Optional[Tuple[int, int]] = None  # character offsets into answer_text
    schema_version: str = "1.0"

class VerifiedAtom(BaseModel):
    atom: Atom
    v1_status: str  # MATCH, MISMATCH, NOT_FOUND, NA
    v2_entail_prob: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "1.0"

class ScoredAtom(BaseModel):
    verified: VerifiedAtom
    risk: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "1.0"

class Thresholds(BaseModel):
    stratum_name: str
    accept_below: float
    abstain_above: float
    schema_version: str = "1.0"

class Decision(BaseModel):
    atom_id: str
    status: str  # SUPPORTED, ABSTAINED, VERIFIED, NOT_VERIFIED
    risk: float
    schema_version: str = "1.0"
