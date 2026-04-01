from flask import Flask, request, jsonify
from api.process_text import process_text_si, process_text_tc

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/process", methods=["POST"])
def process():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400

    spans = process_text_si(data["text"])
    result = process_text_tc(data["text"], spans)
    return jsonify(result), 200

def main():
    app.run(debug=True)

if __name__ == "__main__":
    main()
