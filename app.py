from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import os
from flask_cors import CORS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

# Load .env variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)

@app.route("/")
def home():
    # Serve index.html from project root
    return send_from_directory(".", "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    response = llm.invoke([
        SystemMessage(content="""You are a friendly and knowledgeable healthcare assistant. 
        - Answer in simple, calm, and supportive language.
        - Focus on explaining symptoms, precautions, home care suggestions.
        - Do NOT give strict medical instructions or say things like 'take this medicine'.
        - Always include a soft reminder: 'This is not medical advice, please consult a doctor for confirmation."""),
        HumanMessage(content=user_message)
    ])
    return jsonify({
        "user": user_message,
        "bot": response.content
    })

if __name__ == "__main__":
    app.run(debug=True)
