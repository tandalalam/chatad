from typing import List, Dict


class ConversationModel:
    def __init__(self, conversation: List[Dict[str, str]]):
        self.__check_format(conversation)
        self.conversation = conversation

    @staticmethod
    def __check_format(conversation):
        for message in conversation:
            if set(message.keys()) != {'role', 'content'}:
                raise ValueError(f'Invalid arguments: {message.keys()} != {{"role", "content"}}')

    def get_conversation(self):
        return str(self.conversation)