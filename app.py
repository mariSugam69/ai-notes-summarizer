from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import re

app = Flask(__name__)
CORS(app)

# ⚡ Faster model
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6"
)

def is_code(text):
    code_patterns = [
        r";", r"\{", r"\}", r"def\s+", r"class\s+", r"import\s+",
        r"#include", r"public\s+static", r"console\.log",
        r"function\s*\(", r"var\s+", r"let\s+", r"const\s+"
    ]
    matches = sum(bool(re.search(p, text)) for p in code_patterns)
    return matches >= 2


@app.route("/summarize", methods=["POST"])
def summarize():

    data = request.get_json()
    text = data.get("text")

    if not text:
        return jsonify({"summary": "No text provided"})

    # 🚫 If code detected
    if is_code(text):
        return jsonify({
            "summary": "⚠️ Programming code detected. Please use code explanation tool."
        })

    # ⭐ Smaller chunk = faster
    max_chunk = 700
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]

    summaries = []

    for chunk in chunks:
        result = summarizer(
            chunk,
            max_length=180,
            min_length=60,
            do_sample=False
        )
        summaries.append(result[0]["summary_text"])

    final_summary = " ".join(summaries)

    return jsonify({"summary": final_summary})


if __name__ == "__main__":
    app.run(debug=False, threaded=True)
