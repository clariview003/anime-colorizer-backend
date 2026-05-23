from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
import uuid
from inference import run_inference

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "temp"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return "Anime Colorizer Backend Running"


@app.route("/colorize", methods=["POST"])
def colorize():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    unique_id = uuid.uuid4().hex

    input_path = os.path.join(
        UPLOAD_FOLDER,
        f"input_{unique_id}.png"
    )

    output_path = os.path.join(
        UPLOAD_FOLDER,
        f"output_{unique_id}.png"
    )

    # Save uploaded image
    file.save(input_path)

    # Run AI model
    run_inference(input_path, output_path)

    # Return image directly
    return send_file(
        output_path,
        mimetype="image/png"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)