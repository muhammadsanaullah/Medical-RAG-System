import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Used a deault Gemini API key here
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# We build the prompt from the retrieved documents to send to the LLM for an answer
def build_prompt(query, docs):
    
    context = "\n\n".join([
        f"Title: {doc['title']} (PMID: {doc['pmid']})\nAbstract: {doc['abstract']}"
        for doc, _ in docs
    ])

    prompt = f"""Here's some information about a paper: {context}. This is the only context you should use,
    to answer this question: {query}. Cite sources by title or PMID."""
    
    return prompt

# Calling LLM to generate an answer to the given prompt
def generate_answer(query, docs):

    prompt = build_prompt(query, docs)

    model = genai.GenerativeModel("gemini-2.5-flash")

    response = model.generate_content(prompt)

    return response.text