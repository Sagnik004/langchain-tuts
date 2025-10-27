import os
from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()

# https://youtu.be/NfvCyxcMjew?list=PLNIQLFWpQMRU1Ayjc-LX2k01Jj7uDG0rU
# Have a real time conversation with LLM with chat history stored in cloud (firebase)

"""
Steps to replicate this example:
1. Create a Firebase account
2. Create a new Firebase project and FireStore Database
3. Retrieve the Project ID
4. Install the Google Cloud CLI on your computer
    - https://cloud.google.com/sdk/docs/install
    - Authenticate the Google Cloud CLI with your Google account
        - https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev
    - Set your default project to the new Firebase project you created
5. pip install langchain-google-firestore
6. Enable the Firestore API in the Google Cloud Console:
    - https://console.cloud.google.com/apis/enableflow;apiid=firestore.googleapis.com?project=langchain-tuts-595d9
"""

# Setup Firebase Firestore
PROJECT_ID = os.getenv("FIRESTORE_PROJECT_ID")
SESSION_ID = "sagnik_session_new"  # This could be a username or a unique ID
COLLECTION_NAME = "chat_history"

# Initialize Firestore Client
print("Initializing Firestore Client...")
client = firestore.Client(project=PROJECT_ID)

# Initialize Firestore Chat Message History
print("Initializing Firestore Chat Message History...")
chat_history = FirestoreChatMessageHistory(
  session_id=SESSION_ID,
  collection=COLLECTION_NAME,
  client=client
)

print("Chat History Initialized.")
print("Current Chat History:", chat_history.messages)

# Initialize Chat Model
llm = ChatOpenAI(
  model="gpt-4o",
  max_tokens=2048,
  temperature=0.6
)

print("Start chatting with the AI. Type 'exit' to quit.")

while True:
  human_input = input("User: ")
  if human_input.lower() == "exit":
    break

  chat_history.add_user_message(human_input)

  ai_response = llm.invoke(chat_history.messages)
  chat_history.add_ai_message(ai_response.content)

  print(f"AI: {ai_response.content}")
