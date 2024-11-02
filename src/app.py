import logging
import os

from flask import Flask, request, jsonify
from controllers.openai_helper import OpenAIHelper
from controllers.ad_controller import AdController
from configs.configs import ConfigManager
import pandas as pd
import json
import numpy as np
from flasgger import Swagger
from models.conversation_model import ConversationModel

if not os.path.exists('logs/'):
    os.mkdir('logs/')
logging.basicConfig(filename='logs/app.log', encoding='utf-8', level=logging.DEBUG)
app = Flask(__name__)

docs = pd.read_csv('files/ads_detail_embedding.csv')
docs['embedding'] = docs['embedding'].apply(lambda x: np.array(json.loads(x)))

openai_configs = ConfigManager.get_config_manager().get_prop('openai_configs')

instructions = dict(
    keywords_from_conversation_instruction=open('files/keyword_instruction.txt').read(),
    advertising_content_instruction=open('files/generator_instruction.txt').read(),
    advertising_format_assistant=open('files/generator_assistant_prompt.txt').read(),
    binary_classification_instruction=open('files/is_relevant_classification_model.txt').read()
)

models = dict(
    binary_classification_model='gpt-3.5-turbo',
    keywords_from_conversation_model='gpt-3.5-turbo',
    advertising_content_model='gpt-3.5-turbo',
    embedding_model='text-embedding-3-large',
)

openai_helper = OpenAIHelper(openai_configs=openai_configs,
                             instructions=instructions,
                             models=models, )
ad_controller = AdController(docs=docs,
                             openai_helper=openai_helper)

swag = {"swag": True,
        "tags": ["demo"],
        "responses": {200: {"description": "Success request"},
                      204: {"description": "Ad Not found"},
                      400: {"description": "Validation error"}}}
Swagger(app)


@app.route('/api/get_ad', methods=['POST'], **swag)
def get_ad_for_conversation(body: ConversationModel):
    data = request.json
    conversation = data.get('conversations', '')

    try:
        ad = ad_controller(conversation)
    except Exception as e:
        logging.error(e)
        raise e

    if ad:
        return jsonify({"status": "success", "ad": ad}), 200
    else:
        return jsonify({"status": "no_ad"}), 204


@app.route('/is_healthy', methods=['GET'])
def is_healthy():
    return jsonify({"status": "success"}), 200


if __name__ == '__main__':
    port = ConfigManager.get_config_manager().get_prop('service_configs').get('port')
    from waitress import serve

    logging.info('Starting metric server.')
    serve(app, host='0.0.0.0',
          port=port,
          threads=5)
