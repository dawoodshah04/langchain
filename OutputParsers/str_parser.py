from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text.\n {text}",
    input_variables=['text']
)


parser = StrOutputParser()

chain = template1 | llm | parser | template2 | llm | parser

result = chain.invoke({'topic':'black hole'})
print(result)
