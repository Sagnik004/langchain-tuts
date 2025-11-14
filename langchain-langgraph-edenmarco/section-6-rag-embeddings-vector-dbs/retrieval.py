import os

from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables from .env file
load_dotenv()


def main():
    # Part-2: User query --> Embed user query --> Retrieve relevant vector embeddings from Pinecone --> Pass to LLM with user prompt and relevant chunks --> Answer
    print("~~~~~ Initializing Embeddings and LLM ~~~~~")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4o")

    query = "What is Pinecone in Machine Learning?"
    chain = PromptTemplate.from_template(template=query) | llm

    vector_store = PineconeVectorStore(
        index_name=os.environ["PINECONE_INDEX_NAME"], embedding=embeddings
    )

    # https://smith.langchain.com/hub/langchain-ai/retrieval-qa-chat
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    result = retrieval_chain.invoke(input={"input": query})
    print(result)


if __name__ == "__main__":
    main()
