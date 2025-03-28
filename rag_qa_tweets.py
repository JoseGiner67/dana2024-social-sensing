import os
import re
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema.runnable import RunnableMap
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import logging
from langchain.chains import RetrievalQA
import pandas as pd
import json

#from langchain_ollama import ChatOllama

#variables de entorno de langchain & openai
os.environ["OPENAI_API_KEY"] = "your key"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your key"

def remove_urls(text):
    return re.sub(r"http\S+|www.\S+", "", text)

def load_tweets(file_path):
    """
    Load and preprocess tweets from the provided JSON file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        tweets = json.load(file)
    
    # Extract relevant fields and filter by DANA-related keywords
    data = []
    for tweet in tweets:
        content = tweet.get("content", "")
        nlp = tweet.get("nlp", {})  # Default to an empty dictionary if 'nlp' is missing
        data.append({
            "title": tweet.get("title", ""),
            "content": content,
            "sentiment": nlp.get("sentiment", 0),  # Default sentiment to 0
            "entities": nlp.get("entities", []),   # Default entities to an empty list
            "hashtags": tweet.get("hashtags", []),
            "creator": tweet.get("creator", {}).get("name", ""),
            "url": tweet.get("url", ""),
            "timestamp": pd.to_datetime(tweet.get("ts", 0), unit='ms', errors='coerce')  # Handle invalid timestamps
        })
    df = pd.DataFrame(data)
    df['content'] = df['content'].apply(lambda x: remove_urls(x) if isinstance(x, str) else x)
    df.drop_duplicates(subset='content', keep='first', inplace=True)
    
    return df

def process_metadata(row):
    """
    Extract metadata from a row of the tweet DataFrame.
    """
    entities = row.get("entities", [])
    if not isinstance(entities, list):
        entities = []  # Ensure entities is always a list
    
    locations = [entity["text"] for entity in entities if entity.get("label") == "LOC"]
    persons = [entity["text"] for entity in entities if entity.get("label") == "PER"]
    
    return {
        "sentiment": row.get("sentiment", 0),
        "hashtags": row.get("hashtags", []),
        "locations": locations,
        "persons": persons,
        "creator": row.get("creator", ""),
        "url": row.get("url", ""),
        "datetime": row.get("timestamp", None)
    }

def create_vector_store(texts, metadatas, embedding_model, index_path):
    """
    Create a FAISS vector store with a progress bar.
    """
    try:
        # Try to load the vector store
        vector_store = FAISS.load_local(index_path, embeddings=embedding_model)
        print("Vector store loaded successfully from disk.")
    except Exception as e:
        print(f"Vector store not found or could not be loaded: {e}")
        print("Creating a new vector store...")
 
        # Create the FAISS vector store from the full embeddings and metadata
        vector_store = FAISS.from_texts(texts, embedding=embedding_model, metadatas=metadatas)
        vector_store.save_local(index_path)
        print("Vector store saved successfully.")
    
    return vector_store


def create_rag_chain(vector_store, k=30, model='gpt-3.5-turbo', temp=0):
    template = """Utiliza los siguientes tuits para responder a la pregunta. No indiques el texto en el que te basas para generar la respuesta. Tienes que ser extenso.

    {context}

    Pregunta: {question}

    Respuesta:"""

    custom_rag_prompt = PromptTemplate.from_template(template)

    # Initialize LLM
    llm = ChatOpenAI(model=model, temperature=temp)
    # Use the filter in the retriever
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k 
        }
    )
    
    rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,  # Include source docs in output
            chain_type_kwargs={"prompt": custom_rag_prompt}
        )
    
    return rag_chain
 
def log_retrieved_docs(docs):
    """
    Log retrieved documents for debugging.
    """
    logging.info("Retrieved Documents:")
    for doc in docs:
        logging.info(f"Metadata: {doc.metadata}\nContent: {doc.page_content}")   
    
def main(question):
    """
    Main function to process tweets and answer a question.
    
    Parameters:
    - question: The question to answer using the RAG chain.
    """

     # Load tweets mentioning the DANA
    tweets_df = load_tweets("data/merged_dana.json")

    texts = tweets_df["content"].tolist()
    metadatas = tweets_df.apply(process_metadata, axis=1).tolist()
    
    # Initialize embeddings
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

    # Create vector store
    vector_store = create_vector_store(texts, metadatas, embedding_model, index_path="faiss_index_twitter_dana")

    # Create RAG chain (existing logic)
    rag_chain = create_rag_chain(vector_store)
    
    # Use the chain to generate a response
    result = rag_chain.invoke({"query": question})
    
    # Format results
    response = result['result']
    retrieved_docs = result["source_documents"]

    # Get the output
    print("\n=== RAG System Output ===")
    print("Question:", question)
    print("Response:", response)
    log_retrieved_docs(retrieved_docs)
    
if __name__ == "__main__":
    user_question = "¿Qué temas de conversación emergen respecto a los impactos ambientales y el cambio climático en relación con el fenómeno de la DANA?"
    main(user_question)

    