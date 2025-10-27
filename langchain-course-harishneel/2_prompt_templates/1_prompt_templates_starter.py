# Resources: https://reference.langchain.com/python/langchain_core/prompts/#langchain_core.prompts.chat.ChatPromptTemplate, https://www.youtube.com/watch?v=VXtkSQVab1g&list=PLNIQLFWpQMRU1Ayjc-LX2k01Jj7uDG0rU&index=12

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(
  model="gpt-4o",
  max_tokens=2048,
  temperature=0.6
)

'''
# Very Basic Usage (this is mostly not used in production because this always create just one Human message which may not be enough in real world use cases)
template = "Write a {tone} email to {company} expressing interest in the {position} position, mentioning {skills} as key strength. Keep it to 4 line at max"

prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({
  "tone": "enthusiastic",
  "company": "zoho",
  "position": "AI Engineer",
  "skills": "AI, Langchain"
})

response = llm.invoke(prompt)

print(response.content)
'''

# Prompt with both System and Human messages (Using Tuples)
messages = [
  ("system", "You are a comedian who tells jokes about {topic}."),
  ("human", "Tell me {joke_count} jokes.")
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({
  "topic": "athletes",
  "joke_count": 3
})
response = llm.invoke(prompt)

print(response.content)
