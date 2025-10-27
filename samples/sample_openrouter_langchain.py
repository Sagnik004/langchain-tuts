import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain

load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

llm = ChatOpenAI(
  openai_api_base="https://openrouter.ai/api/v1",
  openai_api_key=openrouter_api_key,
  model_name="meta-llama/llama-3.3-8b-instruct:free",
  temperature=0.7
)

prompt = PromptTemplate(
  input_variables=["cuisine_type"],
  template="I want to open a restaurant for {cuisine_type} cuisines. Suggest a fancy and unique name for my restaurant."
)

chain = LLMChain(llm=llm, prompt=prompt)

response = chain.run(cuisine_type="Indian")
print(response)
