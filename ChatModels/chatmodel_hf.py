import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HF_ACCESS_TOKEN"),
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What's the capital of the US?")

print(result)
