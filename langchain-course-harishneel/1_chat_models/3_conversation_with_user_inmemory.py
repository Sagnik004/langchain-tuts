from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

# https://youtu.be/d84U8snhj6g?list=PLNIQLFWpQMRU1Ayjc-LX2k01Jj7uDG0rU
# Have a real time conversation with LLM with chat history stored locally

llm = ChatOpenAI(
  model="gpt-4o",
  max_tokens=2048,
  temperature=0.6
)

chat_history = []  # Use a list to store messages

# Set an initial system message (optional)
system_message = SystemMessage("You are an helpful AI assistant")
chat_history.append(system_message)  # Add system message to chat history

# Chat loop
while True:
  query = input("You: ")
  if query.lower() == 'exit':
    break
  chat_history.append(HumanMessage(content=query))  # Add user message to chat history

  # Get AI response using chat history
  result = llm.invoke(chat_history)
  response = result.content
  chat_history.append(AIMessage(content=response))  # Add AI response to chat history

  print(f"AI: {response}")

print("--------- Message History ---------")
print(chat_history)
