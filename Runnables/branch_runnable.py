#used for if-else conditioning
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

def word_count(text):
    return len(text.split())


_endpoint = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    provider="groq",
    huggingfacehub_api_token=os.getenv("HF_ACCESS_TOKEN"),
    max_new_tokens=512,
)
llm = ChatHuggingFace(llm=_endpoint)


prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}\n',
    input_variables=['topic']
)

#if word-limit exceeds 500
prompt2 = PromptTemplate(
    template='Summarize the following text\n{text}',
    input_variables=['text']
)

parser = StrOutputParser()
report_gen_chain = RunnableSequence(prompt1,llm,parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 200, RunnableSequence(prompt2,llm,parser)),
    RunnablePassthrough()
)

chain = RunnableSequence(report_gen_chain,branch_chain)

print(chain.invoke({'topic':'Russia VS Ukraine'}))