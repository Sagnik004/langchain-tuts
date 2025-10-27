from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage

load_dotenv()

# https://youtu.be/1ezPmIWcFgU?list=PLNIQLFWpQMRU1Ayjc-LX2k01Jj7uDG0rU

llm = ChatOpenAI(
  model="gpt-4o",
  max_tokens=2048,
  temperature=0.6
)

messages = [
  SystemMessage("You are an expert in social media content strategy"),
  HumanMessage("Give me a short tip on how to create engaging posts on X (formerly Twitter)")
]

response = llm.invoke(messages)
print(response.content)
