from flask import Flask, request, render_template
from chat import Chatbot
from markupsafe import Markup

app = Flask(__name__, static_folder='static')
chat_history = []
chatbot = Chatbot()
chatbot.load()

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_input = request.form["message"]
        bot_response = chatbot.start_chatbot(user_input)
        chat_history.append((user_input,Markup(bot_response)))
    return render_template("index.html", chat_history=chat_history)

if __name__ == "__main__":
    app.run(debug=True)
