# Medical RAG System: Clinical Q/A's using PubMed
This is a sample RAG system that can answer medical questions by retrieving relevant research papers from PubMed and generating an answer using an LLM using only those retrieved papers. This tool helps doctors quickly find answers to their queries backed up by scientific literature relevant to the topic of the query.

## Setup (Windows)
1. Clone the repository using:
   ```
   git clone https://github.com/muhammadsanaullah/Medical-RAG-System.git
   cd Medical-RAG-System
   ```
3. Create a virtual environment to isolate dependencies:
   ```
   python -m venv medrag_env
   medrag_env\Scripts\activate
   ```
4. Inside the activated environment install the required packages:
   ```
   pip install -r requirements.txt
   ```
5. The RAG system uses a LLM API and needs a secure API key. Create a file `.env` in the root directory and add your API key as:
   ```
   GEMINI_API_KEY=XXXXXXXXXXXX
   ```
   Make sure it's included in `.gitignore`, if not, make one.
6. At this point your setup should work and you can run the entire pipeline using:
   ```
   python main.py
   ```
   This will run the main system script that fetches medical articles from PubMed based on the queries you've given it. It will build a retrieval system, evaluate
   performacne of each, and generate answers using the Gemini LLM.
   You can edit parameter values and add/edit/remove queries in the `main.py` file for usage. *Future implementations will include a data parses so this step is
   not required and queries and parameters can be directly given from the terminal command line.*

## Methodology
The system is built in 3 stages: data collection, retrieval and answer generation. 
1. For data collection, PubMed's E-utilities API is used to get the 5 most recent articles for each medical term provided. From each article, important information like title, abstract, author(s), journal, year and DOI are extracted and stored in a JSON file.
2. For retrieval, three different approaches are implemented for the query: BM25, Semantic and Hybrid. BM25 is a keyword-based approach that ranks the documents based on how well their words match the words in the query. Semantic search, using the intfloat/multilingual-e5-small sentence transformer model, understands the meaning of the query and documents rather than just matching words. This model was chosen as it supports both English and Turkish queries and gives a good balance between performance and size.
3. Lastly, the hybrid approach makes use of both the BM25 and Semantic approaches, using a reciprocal rank fusion (RRF) which combines rankings from the other two searches to give its own, based on the combined strengths of the other two.

### BM25 Analysis
BM25 has two key parameters: k1 and b, which control how it ranks documents. k1 determines how much weightage is given to repeated keywords in a document and b controls how document length affects ranking. If k1 is high, documents that repeat important terms more often will rank higher, and if it is low, repetition matters less. If b = 0, document length is completely ignored, but if b = 1, longer documents are normalized so they don't unfairly dominate results. While different values for k1 and b were experimented, it was found that k1 = 1.5 and b = 0.75 provide a good balance to allow the system to consider both keyword weightage and document length without having a bias for either.

### RRF Analysis
RRF combines rankings using score = 1/(k + rank), where k controls how much importance is given to the top ranked results. When k is small like 0, the top few results dominate the final ranking, and when it is large like 1000, all documents contribute almost equally, which reduces the impact of ranking differences. A moderate value like k = 60 gives a good balance.

### Evaluation and RAG Generation
Two metrics evaluated the retrieval methods: precision@5 and mean reciprocal rank (MRR). P@5 measures how many of the top 5 retrieved documents are actually relevant while MRR measures how early the first relevant result appears. From the experiments, the hybrid method performed better than BM25 or semantic methods alone. Once the relevant documents are retrieved, the system uses the given query to generate an answer based on the paper context alone using the gemini-2.5-flash LLM, and returns cited references accordingly. This ensures we don't have any inferred or imagined responses, which may even be relevant somehow, but have factually grounded responses from the relevant literature. 

~

*This was a time-constrained implementation of a Medical RAG System, it can be further worked upon given more time and resources.*
*Some challenges included handling API rate limits and assigning the appropriate LLM for execution for Q/A. These were handled by adding time delays between each query and simplifying the simplest available LLM for this system. Further improvements can include adding a translation step from Turkish to English before retrieval so we avoid any systematic errors cascaded from step to step. Extended experimentation between different LLMs and implementation to other clinical databases can also be significant improvements to this system. A data parses would also be very useful so that queries can be given straight from the terminal and not require manual editing inside the source code file*
*Given a hypothetical scenario where there's a need to use a 70B open-source LLM but without access to a powerful GPU like L40S, alternative GPUs may be looed at like A100 or H100. To optimize memory usage, smaller versions of the model may be used with lesser no. of bits for processing, as long as it doesn't compromise on accuracy and performance.*




