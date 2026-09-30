from typing import List, Optional
from common.schemas import Chunk, RetrievalResult
from retrieval.bm25 import BM25Index
from retrieval.dense import DenseIndex
from retrieval.fusion import rrf
from retrieval.rerank import Reranker
from retrieval.temporal import temporal_filter
from ingest.version_graph.build import VersionGraph
from datetime import date

class Retriever:
    """Master retrieval pipeline combining dense, sparse, RRF, reranking, and temporal filters."""
    def __init__(self, bm25_index: BM25Index, dense_index: DenseIndex, reranker: Reranker, version_graph: VersionGraph):
        self.bm25 = bm25_index
        self.dense = dense_index
        self.reranker = reranker
        self.version_graph = version_graph
        
        # Load chunks from BM25 index as ground truth storage for simplicity
        self.chunk_store = self.bm25.chunks

    def retrieve(self, query: str, query_date: Optional[str] = None, top_k_dense: int = 50, top_k_bm25: int = 50, rerank_top_in: int = 30, final_k: int = 6) -> RetrievalResult:
        
        if not query_date:
            query_date = date.today().strftime("%Y-%m-%d")
            
        # 1. Sparse search
        bm25_ids = self.bm25.search(query, top_k=top_k_bm25)
        
        # 2. Dense search
        dense_ids = self.dense.search(query, top_k=top_k_dense)
        
        # 3. Reciprocal Rank Fusion
        fused_ids = rrf([bm25_ids, dense_ids])
        
        # Hydrate candidates
        candidates = [self.chunk_store[cid] for cid in fused_ids if cid in self.chunk_store]
        
        # 4. Temporal Filter (Drop superseded chunks)
        time_filtered = temporal_filter(candidates, self.version_graph, query_date)
        
        # Take top 30 for reranker
        rerank_candidates = time_filtered[:rerank_top_in]
        
        # 5. Cross-Encoder Reranking
        final_chunks = self.reranker.rerank(query, rerank_candidates, top_k=final_k)
        
        return RetrievalResult(
            query=query,
            query_date=query_date or "latest",
            chunks=final_chunks,
            metadata={"total_retrieved_before_temporal": len(candidates), "after_temporal": len(time_filtered)}
        )
