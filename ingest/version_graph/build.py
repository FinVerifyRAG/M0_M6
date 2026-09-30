import json
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

@dataclass
class VersionNode:
    doc_id: str
    section_id: str
    effective_from: str
    effective_to: str | None = None
    supersedes: List[str] = None
    amends: List[str] = None

    def __post_init__(self):
        self.supersedes = self.supersedes or []
        self.amends = self.amends or []

class VersionGraph:
    def __init__(self):
        self.nodes: Dict[str, List[VersionNode]] = {} # keyed by section_id/doc_id

    def add_node(self, node: VersionNode):
        if node.section_id not in self.nodes:
            self.nodes[node.section_id] = []
        self.nodes[node.section_id].append(node)

    def save(self, filepath: str):
        data = {k: [asdict(n) for n in v] for k, v in self.nodes.items()}
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'VersionGraph':
        graph = cls()
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for k, v in data.items():
                    graph.nodes[k] = [VersionNode(**node_data) for node_data in v]
        except FileNotFoundError:
            pass
        return graph

def build_graph_from_chunks(chunks: List[Any]) -> VersionGraph:
    """Builds a version graph from parsed chunks metadata."""
    graph = VersionGraph()
    for chunk in chunks:
        # Assuming chunk is common.schemas.Chunk
        sec_id = f"{chunk.regulator}_{chunk.metadata.get('chapter', 'General')}_{chunk.metadata.get('section', 'General')}"
        doc_id = chunk.chunk_id
        eff_from = chunk.effective_from or chunk.issue_date
        
        # Pull supersedes from metadata if available
        supersedes = chunk.metadata.get("supersedes", [])
        amends = chunk.metadata.get("amends", [])
        
        node = VersionNode(
            doc_id=doc_id,
            section_id=sec_id,
            effective_from=eff_from,
            effective_to=chunk.effective_to,
            supersedes=supersedes,
            amends=amends
        )
        graph.add_node(node)
    
    return graph
