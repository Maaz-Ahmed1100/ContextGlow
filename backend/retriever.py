# this script handles document retrieval using chromadb
import chromadb
from chromadb.utils import embedding_functions
import re

# we use a local directory to store our database
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# this function loads the embedding model
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# we create or get the collection for our documents
collection = chroma_client.get_or_create_collection(name="space_docs", embedding_function=embedding_fn)

# this is our mock document about space
mock_document = """
Space is the boundless three-dimensional extent in which objects and events have relative position and direction. 
In classical physics physical space is often conceived in three linear dimensions although modern physicists usually consider it with time to be part of a boundless four-dimensional continuum known as spacetime. 
The concept of space is considered to be of fundamental importance to an understanding of the physical universe. 
However disagreement continues between philosophers over whether it is itself an entity a relationship between entities or part of a conceptual framework.
"""

def split_into_sentences(text: str):
    # we split text into sentences using basic regex
    # this looks for periods exclamation marks and question marks
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    # we remove empty strings just in case
    return [s.strip() for s in sentences if s.strip()]

def populate_database():
    # we create chunks from our document
    # for simplicity this document is just one chunk here
    chunks = [mock_document]
    
    for chunk_index, chunk in enumerate(chunks):
        sentences = split_into_sentences(chunk)
        for sent_index, sentence in enumerate(sentences):
            # we create a unique id for each sentence
            doc_id = f"chunk_{chunk_index}_sent_{sent_index}"
            
            # we add the sentence to our collection
            collection.add(
                documents=[sentence],
                ids=[doc_id],
                metadatas=[{"chunk_index": chunk_index, "sent_index": sent_index}]
            )
            print(f"added {doc_id} to database")

def retrieve_context(query: str, top_k: int = 5):
    # we search the database for the closest matches
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    matches = []
    # we loop through the results to format them
    if results and results.get('documents') and results.get('ids'):
        # results return list of lists for each query
        for i in range(len(results['documents'][0])):
            doc_text = results['documents'][0][i]
            doc_id = results['ids'][0][i]
            matches.append({"id": doc_id, "text": doc_text})
            
    return matches

if __name__ == "__main__":
    # we run the population step if the database is empty
    if collection.count() == 0:
        populate_database()
        
    # we run a quick test query
    test_query = "what is spacetime"
    print(f"searching for {test_query}")
    results = retrieve_context(test_query)
    for result in results:
        print(f"found match {result['id']} {result['text']}")
