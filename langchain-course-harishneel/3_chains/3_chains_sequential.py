# Reference: https://youtu.be/p_lnRl7g0oU?list=PLNIQLFWpQMRU1Ayjc-LX2k01Jj7uDG0rU

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
llm_model = ChatOpenAI(
  model="gpt-4o",
  max_tokens=2048,
  temperature=0.6
)

# Define prompt templates
animal_facts_template = ChatPromptTemplate.from_messages(
  [
    ("system", "You like telling facts and you tell facts about {animal}"),
    ("human", "Tell me {count} facts"),
  ]
)

# Define a prompt template for translation to French
translation_template = ChatPromptTemplate.from_messages(
  [
    ("system", "You are a translator and convert the provided text into {language}"),
    ("human", "Translate the following text to {language}: {text}"),
  ]
)

# Define additional processing steps using RunnableLambda
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")
prepare_for_translation = RunnableLambda(lambda output: {"text": output, "language": "french"})

# Create the combined chain using LangChain Expression Language (LCEL)
chain = animal_facts_template | llm_model | StrOutputParser() | prepare_for_translation | translation_template | llm_model | StrOutputParser()

# Run the chain
result = chain.invoke({"animal": "dog", "count": 2})

# Output
print(result)
