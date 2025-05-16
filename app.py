from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

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

    if spell:
        output = "[SpellCorrected] " + output
    if sentiment:
        output += " | [Sentiment: Neutral]"
    if autocomplete:
        output += " | ...Autocompleted"
    if generate:
        output += " | [New Generated Text]"

    return jsonify({"output": output})

if __name__ == "__main__":
    app.run(debug=True)
