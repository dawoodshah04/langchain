from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableSequence
from dotenv import load_dotenv
import os

load_dotenv()

_endpoint = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    provider="groq",
    huggingfacehub_api_token=os.getenv("HF_ACCESS_TOKEN"),
    max_new_tokens=512,
)

llm = ChatHuggingFace(llm=_endpoint)

prompt1 = PromptTemplate(
    template='Generate a twitter tweet on {topic}',
    input_variables=['topic']
)


prompt2 = PromptTemplate(
    template='Generate a Linked post on {topic}',
    input_variables=['topic']
)



parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, llm, parser),
    'linked': RunnableSequence(prompt2, llm, parser)
})

print(parallel_chain.invoke({'topic':'AI Engineer'}))
