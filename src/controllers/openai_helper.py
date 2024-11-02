import json
import logging
from typing import Dict, List
import re
import httpx
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from controllers.plugins.create_ad import CreateAd

plugins = dict(create_ad=CreateAd())
MUST_HAVE_INSTRUCTIONS = ['keywords_from_conversation_instruction',
                          'advertising_content_instruction',
                          'advertising_format_assistant',
                          'binary_classification_instruction']
MUST_HAVE_MODELS = ['binary_classification_model',
                    'keywords_from_conversation_model',
                    'advertising_content_model',
                    'embedding_model']


def check_must_haves(must_have_list: List, keys: List):
    for item in must_have_list:
        if item not in keys:
            assert False, f'Expected {item}, but it is not in {keys}'


class OpenAIHelper:
    create_ad = plugins['create_ad']

    def __init__(self, openai_configs, metis_configs, instructions: Dict, models: Dict):
        self.instructions = instructions
        check_must_haves(must_have_list=MUST_HAVE_INSTRUCTIONS,
                         keys=list(instructions.keys()))
        self.models = models
        openai_configs = {k: v for k, v in openai_configs.items() if v is not None}
        http_client = None
        if 'proxy' in openai_configs:
            http_client = httpx.Client(proxies=openai_configs['proxy'])
            del openai_configs['proxy']
        self.metis_client = OpenAI(**metis_configs)
        self.openai_client = OpenAI(**openai_configs, http_client=http_client)

    def search_advertisements(self, docs, query, n=3):
        df = docs.copy()
        embedding = self.get_embedding(query, model=self.models['embedding_model'])
        df['similarities'] = df.embedding.apply(lambda x: cosine_similarity([x], [embedding])[0][0])

        res = df.sort_values('similarities', ascending=False).head(n)
        return res

    def get_embedding(self, text, model="text-embedding-3-large"):
        text = text.replace("\n", " ")
        return self.openai_client.embeddings.create(input=[text], model=model).data[0].embedding

    def get_keywords_from_conversation(self, conversation):
        data = {
            "model": self.models['keywords_from_conversation_model'],
            "messages": [
                {
                    "role": "system",
                    "content": self.instructions['keywords_from_conversation_instruction']
                }
            ]
        }
        data['messages'].append({"role": "user", "content": str(conversation)})
        keywords = self.metis_client.chat.completions.create(
            **data
        ).choices[0].message.content
        logging.info(keywords)
        return keywords

    def create_advertising_content(self, conversation, url, aff_link, name, details, call_to_action, image_url=None):
        """
        :must haves: advertising_content_model, advertising_content_instruction, advertising_format_assistant
        :param conversation:
        :param url:
        :param name:
        :param details:
        :param call_to_action:
        :param image_url:
        :return:
        """
        data = {
            "model": self.models['advertising_content_model'],
            "messages": [
                {
                    "role": "system",
                    "content": self.instructions['advertising_content_instruction']
                },
                {
                    "role": "assistant",
                    "content": self.instructions['advertising_format_assistant'].format(name=name,
                                                                                        details=details,
                                                                                        call_to_action=call_to_action)
                },
                {
                    "role": "user",
                    "content": str(conversation)
                }
            ],
            "temperature": 0.5,
            "tools": [self.create_ad.get_spec()],
            "function_call": {"type": "function", "function": {"name": "create_ad"}}
        }

        resp = self.metis_client.chat.completions.create(
            **data
        ).choices[0].message

        if dict(resp).get('tool_calls'):
            function_args = json.loads(resp.tool_calls[0].function.arguments)
            function_args['url'] = url
            function_args['image_url'] = image_url
            function_args['aff_link'] = aff_link
            response_message = self.create_ad(**function_args)
        else:
            raise KeyError()

        conversational_ad_content = function_args['conversational_ad']
        logging.info(conversational_ad_content)
        return conversational_ad_content, response_message

    def is_conversation_related(self, conversation, advertisement):
        """
        :must haves: binary_classification_model, binary_classification_instruction
        """

        structured_input = f'''
    **conversation:** {conversation}
    **advertisement:** {advertisement}
        '''.strip()

        data = {
            "model": self.models['binary_classification_model'],
            "messages": [
                {
                    "role": "system",
                    "content": self.instructions['binary_classification_instruction']
                },
                {
                    "role": "user",
                    "content": structured_input
                }
            ],
            "temperature": 0.5,
            "response_format": {"type": "json_object"}
        }
        response = self.metis_client.chat.completions.create(**data).choices[0].message.content
        logging.info(response)
        regex = r'\{\n?  \"classification\": (\d)\n?\}'
        return int(re.match(regex, response).group(1))
