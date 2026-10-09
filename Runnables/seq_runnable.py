from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSequence
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
    template='Give me one joke on {topic}\n',
    input_variables=['topic']
)


prompt2 = PromptTemplate(
    template='Explain the following joke in a simple paragraph- {text}\n',
    input_variables=['text']
)

parser = StrOutputParser()
#converting prompt1 result to text

to_text = RunnableLambda(lambda x:{'text',x})
chain = RunnableSequence(prompt1,llm,parser,to_text,prompt2,llm,parser)



     # With streaming - use transform() to process a stream
#stream = chain.stream({"topic": "AI"})
#for chunk in parser.transform(stream):
#   print(chunk, end="")

print(chain.invoke({"topic": "AI"}))