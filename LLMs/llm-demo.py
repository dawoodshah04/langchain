from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model="gpt-4o")

result = llm.invoke("What is the Capital of US")

print(result)