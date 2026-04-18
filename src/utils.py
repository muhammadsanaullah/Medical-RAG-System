import json
import time

# save and load a JSON file for a given path
def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# PubMed has a rate limit, so keep a delay between queries 
def rate_limit_sleep(delay=0.3):
    time.sleep(delay)

# Remove unwanted spaces in the texts
def clean_text(text):
    if not text:
        return ""
    return " ".join(text.lower().split())