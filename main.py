from fastapi import FastAPI
from pydantic import BaseModel
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser


class InputModel(BaseModel):
    OPENAI_API_KEY: str
    MODEL_ID: str
    article: str

my_app = FastAPI()

@my_app.post('/easy_japanese')
async def easy_japanese(inputs: InputModel):
    llm = OpenAI(openai_api_key = inputs.OPENAI_API_KEY, model =inputs.MODEL_ID)
    summary_prompt = PromptTemplate.from_template(
        """あなたは新聞記者です。以下で与えられる記事の原稿を元に、その内容を短く5文以内で要約しなさい。
        記事内容：{article}
        要約内容：以下は、記事の短い要約である：
        """
        )
    easy_prompt = PromptTemplate.from_template(
        """あなたは、難しい日本語の文章を、日本語が少ししか分からない方でも分かるようにする仕事をしています。以下で与えられる文章を、小学一年生でも分かるように書き換えなさい。特に、漢字をあまり使わないようにすることに注意すること。
        文章：{summary}
        簡易化内容：以下は、上の文章を簡単な日本語に書き換えたものである：
        """
        )
    summary_chain = summary_prompt | llm | StrOutputParser()
    easy_chain = easy_prompt | llm | StrOutputParser()
    chain = {'summary': summary_chain} | RunnablePassthrough.assign(review=easy_chain)
    
    return chain.invoke({'article': inputs.article})
