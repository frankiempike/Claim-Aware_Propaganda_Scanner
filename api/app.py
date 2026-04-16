import os

from flask import Flask, request, jsonify
from flask_cors import CORS
from api.process_text import process_text_si, process_text_tc

app = Flask(__name__)
CORS(app)

PRIVACY_POLICY = """
Privacy Policy
--------------

Our Propaganda Scanner Extension is committed to protecting your privacy. 

We do not collect, store, or share any personally identifiable information. 

When you use our extension, the text content from your current webpage that you submit for analysis is processed on our servers.

We do not retain any of this data after processing nor share it with third parties.
"""

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

@app.route('/privacy_policy', methods=['GET'])
def privacy_policy():
        return jsonify({"message": PRIVACY_POLICY}), 200

def main():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)

if __name__ == "__main__":
    main()
