from flasgger import Schema, fields
from flask import abort
from marshmallow.validate import OneOf


class Message(Schema):
    role = fields.String(required=True, validate=OneOf(['user', 'assistant']))
    content = fields.Str(required=True)

    def swag_validation_function(self, data, main_def):
        self.load(data)

    def swag_validation_error_handler(self, err, data, main_def):
        abort(400, err)


class ConversationModel(Schema):
    conversations = fields.Nested(Message, required=True, many=True)

    def swag_validation_function(self, data, main_def):
        self.load(data)

    def swag_validation_error_handler(self, err, data, main_def):
        abort(400, err)


class Ad(Schema):
    adverting_content = fields.Str()
