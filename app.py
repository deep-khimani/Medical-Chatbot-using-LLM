from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

system_prompt = (
    "You are a helpful medical assistant. "
    "Use the retrieved documents to answer questions about medical topics. "
    "If you are not sure about something, say so."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}"),
])

qa_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt,
)

rag_chain = create_retrieval_chain(retriever, qa_chain)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_msg = data.get("message", "").strip()
        if not user_msg:
            return jsonify({"reply": "⚠️ Please enter a valid query."})

        print("User:", user_msg)
        response = rag_chain.invoke({"input": user_msg})
        bot_reply = response.get("answer", "Sorry, I couldn't process that.")

        print("Bot:", bot_reply)
        return jsonify({"reply": bot_reply})

    except Exception as e:
        print("Error:", e)
        return jsonify({"reply": "⚠️ An error occurred while processing your request."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
