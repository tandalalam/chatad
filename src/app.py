import logging

from flask import Flask, request, jsonify
from controllers.openai_helper import OpenAIHelper
from controllers.ad_controller import AdController
from configs.configs import ConfigManager
import pandas as pd
import json
import numpy as np
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


@app.route('/get_ad', methods=['POST'])
def get_ad_for_conversation():
    data = request.json
    conversation = data.get('conversation', '')

    try:
        ad = ad_controller(conversation)
    except Exception as e:
        logging.error(e)
        ad = None

    if ad:
        return jsonify({"status": "success", "ad": ad}), 200
    else:
        return jsonify({"status": "no_ad"}), 204


@app.route('/is_healthy', methods=['GET'])
def is_healthy():
    return jsonify({"status": "success"}), 200


if __name__ == '__main__':
    port=ConfigManager.get_config_manager().get_prop('service_configs').get('port')
    app.run(port=port)
