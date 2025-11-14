import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters.character import CharacterTextSplitter

# Load environment variables from .env file
load_dotenv()


def main():
    # Part-1: Load Document --> Split into Chunks --> Perform Embedding of Chunks --> Store Embeddings in vector DB
    print("~~~~~ Loading Document ~~~~~")
    loader = TextLoader(
        "E:/000-AI/tutorials/youtube/langchain-tuts/langchain-langgraph-edenmarco/section-6-rag-embeddings-vector-dbs/mediumblog1.txt",
        encoding="utf-8",
    )
    document = loader.load()

    print("~~~~~ Splitting ~~~~~")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(document)
    print("--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")

    print("~~~~~ Embedding & Ingesting ~~~~~")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    PineconeVectorStore.from_documents(
        docs, embeddings, index_name=os.environ["PINECONE_INDEX_NAME"]
    )


if __name__ == "__main__":
    main()
