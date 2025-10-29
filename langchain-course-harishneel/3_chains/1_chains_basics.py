# Reference: https://youtu.be/MkiSRUNoWwE?list=PLNIQLFWpQMRU1Ayjc-LX2k01Jj7uDG0rU

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
llm = ChatOpenAI(
  model="gpt-4o",
  max_tokens=2048,
  temperature=0.6
)

# Define prompt templates (no need for separate Runnable chains)
prompt_template = ChatPromptTemplate.from_messages(
  [
    ("system", "You are a facts expert who knows facts about {animal}"),
    ("human", "Tell me {fact_count} facts")
  ]
)

# Create the combined chain using LangChain Expression Language (LCEL)
# chain = prompt_template | llm
chain = prompt_template | llm | StrOutputParser()

# Run the chain
result = chain.invoke({"animal": "elephant", "fact_count": 2})

# Output
print(result)
