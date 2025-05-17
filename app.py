from flask import Flask, render_template, request, jsonify
from char_model import CharRNN
from utils import Sample_text
from tensorflow.keras.models import load_model
import pickle
from textblob import TextBlob
import nltk

nltk.download('punkt')
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

app = Flask(__name__)

# Load vocab dicts
with open("char_to_idx.pkl", "rb") as f:
    char_to_idx = pickle.load(f)

with open("idx_to_char.pkl", "rb") as f:
    idx_to_char = pickle.load(f)

# Load Keras model (replace 'char_rnn_model' with your model path)
try:
    rnn_model = load_model("char_rnn_model")  # or "char_rnn_model.h5"
except Exception as e:
    print(f"Error loading model: {e}")
    rnn_model = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    text = request.form.get("input_text", "")
    spell = request.form.get("spell") == "on"
    sentiment = request.form.get("sentiment") == "on"
    autocomplete = request.form.get("autocomplete") == "on"
    generate = request.form.get("generate") == "on"

    output = text

    # Spell correction
    if spell:
        corrected = str(TextBlob(output).correct())
        output = "[SpellCorrected] " + corrected

    # Sentiment analysis
    if sentiment:
        score = sia.polarity_scores(output)
        sentiment_label = (
            "Positive" if score['compound'] > 0.05 
            else "Negative" if score['compound'] < -0.05 
            else "Neutral"
        )
        output += f" | [Sentiment: {sentiment_label}]"

    # Autocomplete (simple logic)
    if autocomplete:
        output += " ...and more to come"

    # Text generation
    if generate:
        if rnn_model:
            generated = Sample_text(rnn_model, start_string=text, length=500, temperature=0.6)
            output += f" | [Generated Text: {generated}]"
        else:
            output += " | [Model not loaded]"

    return jsonify({"output": output})

if __name__ == "__main__":
    app.run(debug=True)


