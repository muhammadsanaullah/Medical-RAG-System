import requests
import xml.etree.ElementTree as ET
from collections import defaultdict
from src.utils import save_json, rate_limit_sleep

# URLs to search papers from and return their details
BASE_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
BASE_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Searching PubMed for a term
def esearch(term, retmax=5):
    # Parameters to search for 
    params = {
        "db": "pubmed",           # database
        "term": term,             # search query
        "retmax": retmax,         # number of results
        "sort": "pub+date",       # latest first
        "retmode": "json"
    }

    # sending the search request
    response = requests.get(BASE_ESEARCH, params=params)
    response.raise_for_status()  # crash if error

    return response.json()["esearchresult"]["idlist"]

# Get article details
def efetch(pmids):
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),   # multiple IDs
        "retmode": "xml"
    }

    response = requests.get(BASE_EFETCH, params=params)
    response.raise_for_status()

    return response.text

# Converts PubMed XML to Python dictionaries
def parse_xml(xml_text, term):

    root = ET.fromstring(xml_text)
    articles = []

    for article in root.findall(".//PubmedArticle"):
        try:
            pmid = article.findtext(".//PMID")
            title = article.findtext(".//ArticleTitle")

            # Combine multiple abstract sections
            abstract = " ".join([
                a.text for a in article.findall(".//AbstractText") if a.text
            ])

            authors = article.findall(".//Author")
            first_author = "Unknown"

            if authors:
                last = authors[0].findtext("LastName")
                fore = authors[0].findtext("ForeName")
                first_author = f"{last} {fore}"

            journal = article.findtext(".//Journal/Title")
            year = article.findtext(".//PubDate/Year")

            # Extract DOI
            doi = None
            for id_ in article.findall(".//ArticleId"):
                if id_.attrib.get("IdType") == "doi":
                    doi = id_.text

            articles.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "first_author": first_author,
                "journal": journal,
                "year": year,
                "doi": doi,
                "terms": [term]
            })

        except Exception as e:
            print("Parse error:", e)

    return articles

# Dictionary to store unique articles found
def build_corpus(terms, output_path="data/pubmed_corpus.json"):

    corpus = {}  # key = PMID

    # iterating voer all medical terms
    for term in terms:
        try:
            pmids = esearch(term)
            rate_limit_sleep()

            xml_data = efetch(pmids)
            rate_limit_sleep()

            articles = parse_xml(xml_data, term)

            for art in articles:
                if art["pmid"] in corpus:

                    # if duplicate then add term
                    corpus[art["pmid"]]["terms"].append(term)
                else:
                    corpus[art["pmid"]] = art

        except Exception as e:
            print(f"Error with term {term}: {e}")

    # converting dictionary into list
    corpus_list = list(corpus.values())

    save_json(corpus_list, output_path)

    print(f"Total number of unique articles found: {len(corpus_list)}")

    return corpus_list