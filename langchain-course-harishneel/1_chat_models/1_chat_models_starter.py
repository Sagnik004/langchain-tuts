# Resources: https://docs.langchain.com/oss/python/integrations/chat/openai, https://www.youtube.com/playlist?list=PLNIQLFWpQMRU1Ayjc-LX2k01Jj7uDG0rU

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
  model="gpt-4o",
  temperature=0.7
)

response = llm.invoke("Convert the statement \"I love programming\" from English to French.")
print(response.content) # (or) response.text
