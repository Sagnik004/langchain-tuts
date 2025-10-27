import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_key,
)

model = "meta-llama/llama-3.3-8b-instruct:free"

response = client.chat.completions.create(
  model=model,
  messages=[
    {
      "role": "system",
      "content": "You are a helpful assistant that can answer questions and help with tasks."
    },
    {
      "role": "user",
      "content": "I want to open a restaurant for American cuisines. Suggest a fancy and unique name for my restaurant."
    }
  ],
  temperature=0.6,
  top_p=0.6,
)

print(response.choices[0].message.content)
