from typing import Optional, List
from ingest.version_graph.build import VersionGraph, VersionNode

def in_force(graph: VersionGraph, section_id: str, date: str) -> Optional[str]:
    """
    Return the doc_id version of section_id valid on `date`, else None.
    Based strictly on the M1 plan implementation.
    """
    if section_id not in graph.nodes:
        return None
        
    versions = sorted(graph.nodes[section_id], key=lambda v: v.effective_from)
    for v in reversed(versions):
        if v.effective_from <= date and (v.effective_to is None or date <= v.effective_to):
            return v.doc_id
    return None

def history(graph: VersionGraph, section_id: str) -> List[VersionNode]:
    """Returns the historical versions of a section."""
    if section_id not in graph.nodes:
        return []
    return sorted(graph.nodes[section_id], key=lambda v: v.effective_from)
