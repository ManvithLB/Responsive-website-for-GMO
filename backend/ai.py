import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai

load_dotenv()
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

API_KEY = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])
instruction = "ANSWER AS IF YOU ARE AN EXPERT IN GMOS AND GEAC. PROVIDE ACCURATE AND DETAILED INFORMATION."

@app.route('/api/chat', methods=['POST'])
def chat_with_ai():
    data = request.json
    question = data.get('question', '')

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    response = chat.send_message(question)
    
    # Remove asterisks from the response text
    cleaned_response = response.text.replace('*', '')

    return jsonify({"response": cleaned_response})

@app.route('/')
def index():
    return render_template('index1.html')

if __name__ == '__main__':
    app.run(debug=True, port=5500)
