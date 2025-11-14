import os

from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables from .env file
load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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

    # result = retrieval_chain.invoke(input={"input": query})
    # print(result)

    # A better prompt way, and implementation with LCEL
    template = """Use the following pieces of context to answer the question at the end. If you don't find the answer, just respond with "I am not sure". Use three sentences maximum and keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
    
    {context}
    
    Question: {question}
    
    Helpful Answer:
    """
    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": vector_store.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )

    res = rag_chain.invoke(query)
    print(res.content)


if __name__ == "__main__":
    main()
