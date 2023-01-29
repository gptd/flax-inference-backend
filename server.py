#! /usr/bin/env python
import os
import io
from flask import Flask, render_template, request
from src.models.inference_pipeline import FlaxSequenceClassificationPipeline
from PIL import Image

import jax

HF_FLAX_MODEL_NAME = os.environ.get(
    "HF_FLAX_MODEL_NAME", "Hello-SimpleAI/chatgpt-detector-roberta"
)
HF_TOKENIZER_NAME = os.environ.get(
    "HF_TOKENIZER_NAME", "Hello-SimpleAI/chatgpt-detector-roberta"
)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 128))

app = Flask(__name__)
pipeline = FlaxSequenceClassificationPipeline(
    HF_FLAX_MODEL_NAME, HF_TOKENIZER_NAME, BATCH_SIZE, MAX_LENGTH
)


@app.route("/predict", methods=["POST"])
def eval():
    query_text = request.json["query_text"]  # type: ignore
    stride = request.json.get("stride", 2)
    predictions, splits = pipeline.predict(query_text, stride)

    output = []
    for logits, split in zip(predictions, splits):
        output.append({"text": split, "prediction": logits})

    # Dictionary return values are implicitly converted to JSON,
    # which the client can parse to get the result.
    return output


if __name__ == "__main__":
    app.run(host="0.0.0.0")
