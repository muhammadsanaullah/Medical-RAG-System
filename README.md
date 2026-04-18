### Medical RAG System: Clinical Q/A's using PubMed
This is a sample RAG system that can answer medical questions by retrieving relevant research papers from PubMed and generating an answer using an LLM using only those retrieved papers. This tool helps doctors quickly find answers to their queries backed up by scientific literature relevant to the topic of the query.

The system is built in 3 stages: data collection, retrieval and answer generation. 
1. For data collection, PubMed's E-utilities API is used to get the 5 most recent articles for each medical term provided. From each article, important information like title, abstract, author(s), journal, year and DOI are extracted and stored in a JSON file.
2. For retrieval, three different approaches are implemented for the query: BM25, Semantic and Hybrid. BM25 is a keyword-based approach that ranks the documents based on how well their words match the words in the query. Semantic search, using the intfloat/multilingual-e5-small sentence transformer model, understands the meaning of the query and documents rather than just matching words. This model was chosen as it supports both English and Turkish queries and gives a good balance between performance and size.
3. Lastly, the hybrid approach makes use of both the BM25 and Semantic approaches, using a reciprocal rank fusion (RRF) which combines rankings from the other two searches to give its own, based on the combined strengths of the other two.

