from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# ── Model ────────────────────────────────────────────────────────────────────
# ChatHuggingFace wraps HuggingFaceEndpoint and uses the 'conversational'
# task — required by the Groq provider (not 'text-generation').
_endpoint = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    provider="groq",
    huggingfacehub_api_token=os.getenv("HF_ACCESS_TOKEN"),
    max_new_tokens=512,
)
llm = ChatHuggingFace(llm=_endpoint)


prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()
chain = prompt | llm | parser

result = chain.invoke({'topic': 'Football'})
print(result)
chain.get_graph().print_ascii()