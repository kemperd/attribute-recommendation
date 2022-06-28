# -*- coding: utf-8 -*-
"""
Inference script that extends from the base infer interface
"""

from os.path import exists
from os import environ
from joblib import load
import numpy as np
from flask import Flask, request, jsonify

from base64 import b64encode
import base64
import io
from json import dumps
import logging
import sys

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, PreTrainedTokenizer

FORMAT = "%(asctime)s:%(name)s:%(levelname)s - %(message)s"
# Use filename="file.log" as a param to logging to log to a file
logging.basicConfig(format=FORMAT, level=logging.INFO)

NUM_GPUS = environ["NUM_GPUS"]
MODEL_PATH = environ["MODEL_PATH"]

app = Flask(__name__)

tokenizer = None
use_gpu = False

@app.before_first_request
def init():
    """
    Load the model if it is available locally
    """
    
    try:
        if environ["NUM_GPUS"] != '0':
            use_gpu = True
        else:
            use_gpu = False
    except KeyError as e:
        use_gpu = False
    
    global image_pipeline
    path = MODEL_PATH
    model_name = "model.ckpt"



    return None

#def predict_from_t5_model(source_text):


@app.route("/v1/models/{}:predict".format("attribute-recommendation-model"), methods=["POST"])
def predict():
    source_text = request.args['source_text']
    
    filename = "{}/model.ckpt".format(MODEL_PATH)
    
    logging.info("Init T5 model from {}".format(filename))
    model = T5ForConditionalGeneration.from_pretrained(filename)
    
    logging.info("Init tokenizer from {}".format(filename))
    tokenizer = T5Tokenizer.from_pretrained(filename)
    if use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)    

    input_ids = tokenizer.encode(source_text, return_tensors="pt", add_special_tokens=True)
    input_ids = input_ids.to(device)
    generated_ids = model.generate(input_ids = input_ids)
    preds = [ tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids ]

    logging.info("Predictions from source_text {}: {}".format(source_text, preds))

    return jsonify(preds)


if __name__ == "__main__":
    init()
    app.run(host="0.0.0.0", debug=True, port=9001)

