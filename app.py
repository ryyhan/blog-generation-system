import os
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


from dotenv import load_dotenv
load_dotenv()

chat = ChatGroq(
    temperature=0.9,
    model="llama3-70b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)


st.title("Blog Generation System!")

system = """
You have to generate a blog on the topic which is given by the human. 
The generated blog should include the following sections:
Heading: Clearly define the topic of the blog.
Introduction: Provide an engaging introduction to the topic.
Content: Present detailed and informative content, supported by research and relevant sources.
Summary: Summarize the main points covered in the blog.
Generate the response in plain text.
"""

human = "{text}"

prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat

topic = st.text_input("Enter the title of the Blog which you want to generate!")

if topic:
    result = chain.invoke({"text": topic})
    st.write(result.content)
