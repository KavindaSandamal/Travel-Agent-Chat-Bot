from flask import Flask, request, jsonify
from chatbot_core import chatbot_main

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = chatbot_main(user_input)
    return jsonify({'response': response})

@app.route('/')
def home():
    return "Travel Agent Chat Bot is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
