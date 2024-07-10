import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])
instruction = "ANSWER AS IF YOU ARE AN EXPERT IN GMOS AND GEAC. PROVIDE ACCURATE AND DETAILED INFORMATION."

analyzer = SentimentIntensityAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

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

@app.route('/api/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.json
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data['text']
    sentiment_scores = analyzer.polarity_scores(text)

    return jsonify({
        "polarity": sentiment_scores['compound'],  # Compound score is the overall sentiment
        "positive": sentiment_scores['pos'],
        "negative": sentiment_scores['neg'],
        "neutral": sentiment_scores['neu']
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
