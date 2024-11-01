from controllers.openai_helper import OpenAIHelper


class AdController:
    ad_controller = None

    @staticmethod
    def get_ad_controller(**kwargs):
        if AdController.ad_controller is None:
            AdController.ad_controller = AdController(**kwargs)
        return AdController.ad_controller

    def __init__(self, docs, openai_helper: OpenAIHelper):
        self.docs = docs
        self.openai_helper = openai_helper

    def __call__(self, conversation, **kwargs):
        # TODO: Error handling
        conversation_keywords = self.openai_helper.get_keywords_from_conversation(conversation)
        most_similar_contents = self.openai_helper.search_advertisements(self.docs, conversation_keywords)

        url, aff_link, image_url, name, details, call_to_action = \
        most_similar_contents[['url', 'aff_link', 'image_url', 'name', 'properties', 'call_to_action']].iloc[0]

        conversational_ad_content, advertisement_content = self.openai_helper.create_advertising_content(conversation,
                                                                                                         url, aff_link,
                                                                                                         name, details,
                                                                                                         call_to_action,
                                                                                                         image_url=image_url)

        if self.openai_helper.is_conversation_related(conversation,
                                                      conversational_ad_content):

            return advertisement_content

        else:
            pass  # TODO: logging
