from chatterbot.trainers import ListTrainer
from flask import Flask, render_template, request
from chatterbot import ChatBot


app = Flask(__name__)


'''
This is an example showing how to train a chat bot using the
ChatterBot ListTrainer.
'''

chatbot = ChatBot('Example Bot')

# Start by training our bot with the ChatterBot corpus data
trainer = ListTrainer(chatbot)

trainer.train([
    'Hello, how are you?',
    'I am doing well',
    'Would you like to know if your daily lifestyle may or may not lead to any heart problems?',
    'Yes',
    'Do you smoke?'
])

# You can train with a second list of data to add response variations

trainer.train([
    'Hello, how are you?',
    'I am great',
    'Would you like to know if your daily lifestyle may or may not lead to any heart problems?',
    'Yes',
    'Do you smoke?'
])

trainer.train([
    'Hello, how are you?',
    'Good',
    'Would you like to know if your daily lifestyle may or may not lead to any heart problems?',
    'Yes',
    'Do you smoke?'
])


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(chatbot.get_response(userText))


if __name__ == "__main__":
    app.run()
