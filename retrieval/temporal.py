from typing import List, Optional
from common.schemas import Chunk
from ingest.version_graph.build import VersionGraph
from ingest.version_graph.query import in_force

def temporal_filter(chunks: List[Chunk], graph: VersionGraph, query_date: Optional[str] = None) -> List[Chunk]:
    """
    Filters chunks by effective dates and version graph.
    If query_date is given, returns only the version of each section active on that date.
    Otherwise, drops superseded chunks entirely (returns only currently active).
    """
    valid_chunks = []
    
    for chunk in chunks:
        # Check against graph supersession logic if section_id is available
        section_id = chunk.metadata.get("section_id")
        
        if query_date and section_id:
            active_doc_id = in_force(graph, section_id, query_date)
            # If there's a known active doc and this chunk's doc isn't it, skip.
            if active_doc_id and chunk.metadata.get("doc_id") != active_doc_id:
                continue
                
        # Hard temporal bounds
        if query_date:
            if chunk.effective_from and chunk.effective_from > query_date:
                continue
            if chunk.effective_to and chunk.effective_to < query_date:
                continue
                
        valid_chunks.append(chunk)
        
    return valid_chunks
