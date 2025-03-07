import warnings

from flask import Flask, request, jsonify
from flask_cors import CORS

from src.meowfi.crew import MeowFiAgent
from src.meowfi.utils import parse_json_string

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/health", methods=["GET"])
def health_check():
    print("INFO: Health check endpoint was called.")
    return jsonify({"status": "ok"})


@app.route("/query", methods=["POST"])
def process_input():
    try:
        user_input = request.json.get("query", "")
        print(f"INFO: Received request: {user_input}")

        meowfi_crew = MeowFiAgent().crew()
        result = meowfi_crew.kickoff(inputs={"user_input": user_input})

        final_out = result.raw

        parsed_output = parse_json_string(input_string=final_out) if final_out else []

        return jsonify({"data": parsed_output})
    except Exception as e:
        error_msg = f"ERROR: processing request: {str(e)}"
        return jsonify({"error": error_msg})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
