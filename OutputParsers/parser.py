from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")


template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text.\n {text}",
    input_variables=['text']
)

prompt1 = template1.invoke({'topic':'black hole'})
result = llm.invoke(prompt1)
prompt2 = template2.invoke({'text':result.content})

result2 = llm.invoke(prompt2)
print(result2.content)