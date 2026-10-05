from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSequence, RunnableParallel, RunnablePassthrough
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
    template='Give me one joke on {topic}\n',
    input_variables=['topic']
)

parser = StrOutputParser()
joke_chain = RunnableSequence(prompt1, llm, parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

chain = RunnableSequence(joke_chain,parallel_chain)

result = chain.invoke({'topic','Tech'})
print("""{} \n word count""".format(result['joke']),result['word_count'])
