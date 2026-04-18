from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import torch
from collections import defaultdict

# We're implementing 3 different search methods here: BM25, Semantic and Hybrid 

# BM25 is keyword-based, it tries to match words in the query with the words in the document
class BM25Retriever:
    def __init__(self, documents, k1=1.5, b=0.75):
        
        self.docs = documents

        # combining the title and abstract
        self.texts = [
            (doc["title"] or "") + " " + (doc["abstract"] or "")
            for doc in documents
        ]

        # splitting into words
        self.tokenized = [text.lower().split() for text in self.texts]

        # BM25 index
        self.bm25 = BM25Okapi(self.tokenized, k1=k1, b=b)

    def search(self, query, top_k=5):
        tokens = query.lower().split()

        scores = self.bm25.get_scores(tokens)

        # ranking the documents by score
        ranked = sorted(
            zip(self.docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:top_k]

# A Semantic search is meaning-based, it'll try to infer meaning not just focus on words
class SemanticRetriever:
    def __init__(self, documents):
       
        self.docs = documents

        # loading multilingual model
        self.model = SentenceTransformer(
            "intfloat/multilingual-e5-small"
        )

        # prepare text
        self.texts = [
            "passage: " + (doc["title"] or "") + " " + (doc["abstract"] or "")
            for doc in documents
        ]

        # converting the documents into vectors
        self.embeddings = self.model.encode(
            self.texts,
            convert_to_tensor=True
        )

    def search(self, query, top_k=5):
        query_emb = self.model.encode(
            "query: " + query,
            convert_to_tensor=True
        )

        # similarity score to match results
        scores = util.cos_sim(query_emb, self.embeddings)[0]

        top_results = torch.topk(scores, k=top_k)

        return [
            (self.docs[i], float(scores[i]))
            for i in top_results.indices
        ]

# Hybrid RRF combines BM25 and Semantic together by combining their rankings
def reciprocal_rank_fusion(results_list, k=60, top_k=5):
    scores = defaultdict(float)

    for results in results_list:
        for rank, (doc, _) in enumerate(results, start=1):
            # RRF formula
            scores[doc["pmid"]] += 1 / (k + rank)

    # Sort by fused score
    ranked = sorted(scores.items(),
                    key=lambda x: x[1],
                    reverse=True)

    # Map back to docs
    pmid_to_doc = {
        doc["pmid"]: doc
        for results in results_list
        for doc, _ in results
    }

    return [
        (pmid_to_doc[pmid], score)
        for pmid, score in ranked[:top_k]
    ]