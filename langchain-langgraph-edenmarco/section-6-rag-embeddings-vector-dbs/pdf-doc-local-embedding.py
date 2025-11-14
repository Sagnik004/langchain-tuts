from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic import hub

# Load environment variables from .env file
load_dotenv()

def main():
    print("~~~~~ Loading the PDF chunks/documents...")
    pdf_path = "E:/000-AI/tutorials/youtube/langchain-tuts/langchain-langgraph-edenmarco/section-6-rag-embeddings-vector-dbs/react_prompt_paper.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    print("~~~~~ Loading complete!")

    print("~~~~~ Chunking it again to ensure LLM token limit is not hit...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    chunked_docs = text_splitter.split_documents(documents=documents)
    print("~~~~~ Chunking complete!")

    print("~~~~~ Creating embeddings and saving to local vector store...")
    openai_embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(documents=chunked_docs, embedding=openai_embedding)
    vectorstore.save_local("faiss_index_react")
    print("~~~~~ Local Vector Store saving complete!")

    print("~~~~~ Retrieving relevant chunks and have LLM answer question specific to that context...")
    retrieved_vectorstore = FAISS.load_local(
        "faiss_index_react", embeddings=openai_embedding, allow_dangerous_deserialization=True
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        OpenAI(), retrieval_qa_chat_prompt
    )
    retrieval_chain = create_retrieval_chain(
        retrieved_vectorstore.as_retriever(), combine_docs_chain
    )

    res = retrieval_chain.invoke({"input": "Give me the gist of ReAct in 3 sentences"})
    print(f"Answer: {res["answer"]}")


if __name__ == "__main__":
    main()
