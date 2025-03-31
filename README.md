# DANA 2024 Social Sensing

This repository contains the code used in the study **"Harnessing Social Sensing for Real-Time Flood Event Reconstruction: A Digital Autopsy of the 2024 Valencia DANA"**, submitted to *Nature Communications*.

We reconstruct the 2024 DANA flood event in Valencia (Spain) using citizen-generated data from X (formerly Twitter). The project combines Natural Language Processing (NLP) techniques with a Retrieval-Augmented Generation (RAG) system to extract real-time insights and generate structured intelligence to support emergency response.

---

## 📂 Contents

This repository includes:

- `dana_twitter_analysis.ipynb`: Preprocessing, geolocation, sentiment analysis, named entity recognition (NER), and topic modeling using Twitter data.

- `rag_qa_tweets.py`: Implementation of a Retrieval-Augmented Generation system to process citizen reports and generate structured outputs for crisis management. This code uses the OpenAI API (via langchain-openai) and requires an API key to run the pipeline. Due to security and licensing restrictions, we are unable to share API credentials.

If you wish to test the code, please obtain your own OpenAI API key from https://platform.openai.com/ and add it to a .env file as:

```bash
OPENAI_API_KEY=your_api_key
```
---

## ⚠️ Data Disclaimer

Due to [Twitter's Developer Policy](https://developer.twitter.com/en/developer-terms/agreement-and-policy), we cannot publicly release the full dataset. Tweet IDs and metadata can be provided for academic purposes upon request to the corresponding author.

---

## 🧠 Requirements

- Python 3.9+  
- Jupyter Notebook  
- HuggingFace Transformers  
- BERTopic  
- HDBSCAN  
- OpenAI API (for GPT-4 and embeddings)

Install required packages using:

```bash
pip install -r requirements.txt
```


