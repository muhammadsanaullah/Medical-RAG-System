import pandas as pd
import time
from src.data_pipeline import build_corpus
from src.retrieval import BM25Retriever, SemanticRetriever, reciprocal_rank_fusion
from src.rag import generate_answer
from src.utils import save_json, rate_limit_sleep
from src.evaluation import precision_at_k, mrr


def main():

    # loading terms
    terms = pd.read_csv("data/medical_terms.csv")["term"].tolist()

    # building corpus
    corpus = build_corpus(terms)

    # initializing retrievers
    k1_value = 1.5
    b_value = 0.75
    bm25 = BM25Retriever(corpus, k1=k1_value, b=b_value)
    semantic = SemanticRetriever(corpus)

    # Example queries
    queries = [
        "What are the latest guidelines for managing type 2 diabetes?","Çocuklarda akut otitis media tedavisi nasıl yapılır?",
        "Iron supplementation dosing for anemia during pregnancy", "Çölyak hastalığı tanı kriterleri nelerdir?",
        "Antibiotic resistance patterns in community acquired pneumonia"
    ]

    for query in queries:
        print("\n*******")
        print("QUERY:", query)

        bm25_res = bm25.search(query)
        sem_res = semantic.search(query)
        rrf_k = 60
        hybrid_res = reciprocal_rank_fusion([bm25_res, sem_res], k=rrf_k)

        print(f"\nRRF (k={rrf_k}) Results:")
        for doc, _ in hybrid_res:
            print("-", doc["title"][:80])

        print(f"\nBM25 (k1={k1_value}, b={b_value}) Results:")
        for doc, _ in bm25_res:
            print("-", doc["title"][:80])

        print("\nHybrid Results:")
        for doc, _ in hybrid_res:
            print("-", doc["title"])


        relevant_pmids = [doc["pmid"] for doc, _ in hybrid_res[:3]]

        # finding how many relevant documents appear in the top 5 results and how early the first relevant document appears
        print("\n***Evaluating Different Retrieval Methods***")
        print("BM25 Precision @ 5:", precision_at_k(bm25_res, relevant_pmids))
        print("Semantic Precision @ 5:", precision_at_k(sem_res, relevant_pmids))
        print("Hybrid Precision @ 5:", precision_at_k(hybrid_res, relevant_pmids))
        print("BM25 MRR:", mrr(bm25_res, relevant_pmids))
        print("Semantic MRR:", mrr(sem_res, relevant_pmids))
        print("Hybrid MRR:", mrr(hybrid_res, relevant_pmids))

        # generate answer from LLM
        answer = generate_answer(query, hybrid_res)

        print("\nANSWER:\n", answer)

        time.sleep(60) # to avoid reaching query limit for Gemini


if __name__ == "__main__":

    main()