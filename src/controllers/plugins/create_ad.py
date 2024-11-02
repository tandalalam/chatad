import base64


def encode_to_base64(input_string):
    byte_data = input_string.encode("utf-8")
    base64_encoded = base64.b64encode(byte_data)
    return base64_encoded.decode("utf-8")


class CreateAd:

    def __init__(self):
        pass

    def get_name(self):
        return "create_ad"

    def get_spec(self):
        return {
            "type": "function",
            "function": {
                "name": "create_ad",
                "description": "Put the advertisement in proper format",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "conversational_ad": {
                            "type": "string",
                            "description": "The advertising content that is created as a part of the conversation."
                        },
                        "call_to_action": {
                            "type": "string",
                            "description": "An encouraging phrase to click on the advertising link."
                        }
                    },
                    "additionalProperties": False,
                    "required": [
                        "conversational_ad",
                        "call_to_action"
                    ]
                }
            }
        }

    def __call__(self, **kwargs):
        conversational_ad, call_to_action, url = kwargs['conversational_ad'], kwargs['call_to_action'], kwargs['url']

        url = kwargs['aff_link'] + encode_to_base64(url)

        image_url = kwargs.get('image_url', None)
        message_text = f"""<blockquote>{conversational_ad}{f"<a href='{image_url}'> </a>" if image_url else ""}
        <a href='{url}'>{call_to_action}</a>
        </blockquote>"""
        return message_text
