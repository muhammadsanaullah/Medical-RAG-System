# Check how many relevant documents are in top-k
def precision_at_k(results, relevant_pmids, k=5):

    retrieved = [doc["pmid"] for doc, _ in results[:k]]
    relevant = set(relevant_pmids)

    return len(set(retrieved) & relevant) / k

# Mean Reciprocal Rank, to find when first relevant document appears
def mrr(results, relevant_pmids):

    for i, (doc, _) in enumerate(results, start=1):
        if doc["pmid"] in relevant_pmids:
            return 1 / i
    return 0