import ollama
from pydantic import BaseModel

class ToolCall(BaseModel):
    name: str
    arguments: dict


class Message(BaseModel):
  role: str
  content: str
  tool_calls: list[ToolCall]


def chat(model: str, messages: list, tools: list=[], **kwargs):
  if messages[0]['role'] != 'system' and len(tools):
    messages = [{
      'role': 'system',
      'content': f'Tools available: {tools}'
    }] + messages
  elif str(tools) not in messages[0]['content']:
    messages[0]['content'] += f'\n\nTools available: {tools}'
  response = ollama.chat(
    messages=messages,
    model=model,
    format=Message.model_json_schema(),
    **kwargs
  )
  message = Message.model_validate_json(response.message.content)
  if len(tools) == 0:
    message.tool_calls = []
  return message
