from flask import Flask, request, jsonify
from process_text import process_text_si

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/process", methods=["POST"])
def process():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400

    preds, specialist_preds = process_text_si(data["text"])
    return jsonify({
        "predictions": preds.tolist(),
        "specialist_predictions": specialist_preds.tolist(),
    })

if __name__ == "__main__":
    app.run(debug=True)
