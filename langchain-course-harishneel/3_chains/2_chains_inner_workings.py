# Reference: https://youtu.be/NRfRzs0p_0E?list=PLNIQLFWpQMRU1Ayjc-LX2k01Jj7uDG0rU

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableSequence

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
llm = ChatOpenAI(
  model="gpt-4o",
  max_tokens=2048,
  temperature=0.6
)

# Define prompt templates
prompt_template = ChatPromptTemplate.from_messages(
  [
    ("system", "You love facts and you tell facts about {animal}"),
    ("human", "Tell me {count} facts."),
  ]
)

# Create individual runnables (steps in the chain)
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: llm.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

# Create the RunnableSequence (equivalent to the LCEL chain)
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

# Run the chain
response = chain.invoke({"animal": "giraffe", "count": 1})

# Output
print(response)
