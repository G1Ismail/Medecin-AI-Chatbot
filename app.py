import os
from dotenv import load_dotenv

from flask import Flask, render_template, jsonify, request

from src.functions import downloadHuggingFace_Embeddings
from src.prompt import *

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MeinLLM = OpenAI(temperature=0.9, max_tokens=500)

os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", System_Prompt),
        ("human", "{input}")
    ]
)

embeddings = downloadHuggingFace_Embeddings()

docSearch = PineconeVectorStore.from_existing_index(
    index_name="medicin-chat-bot",
    embedding=embeddings
)
retriever = docSearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

question_answer_chain = create_stuff_documents_chain(MeinLLM, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Meine App:
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chatbot.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])



if __name__ == '__main__':
    #app.run(host="0.0.0.0", port=8080, debug=True)
    app.run()
