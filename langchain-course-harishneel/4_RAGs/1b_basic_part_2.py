import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables from .env
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistance_dir = os.path.join(current_dir, "db", "chroma_db")

# Define the embedding model
embeddings = OpenAIEmbeddings(
  model = "text-embedding-3-small"
) # the model should match exactly with what the vector embeddings of the book/resources was created using, else it will not find matches

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistance_dir, embedding_function=embeddings)

# Define the user's question
query = "Where does Gandalf meet Frodo?"

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
  search_type="similarity_score_threshold",
  search_kwargs={"k": 3, "score_threshold": 0.5},
)
relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
  print(f"Document {i}:\n{doc.page_content}\n")
  if doc.metadata:
    print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
