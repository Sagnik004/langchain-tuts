import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
token = os.getenv("GITHUB_TOKEN")
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1"

client = OpenAI(
  base_url=endpoint,
  api_key=token,
)

response = client.chat.completions.create(
  messages=[
    {
      "role": "system",
      "content": "You are a helpful assistant that can answer questions and help with tasks."
    },
    {
      "role": "user",
      "content": "I want to open a restaurant for Indian cuisines. Suggest a fancy and unique name for my restaurant."
    }
  ],
  temperature=0.6,
  top_p=0.6,
  model=model
)

print(response.choices[0].message.content)
