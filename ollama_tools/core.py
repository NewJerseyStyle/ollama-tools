from pydantic import BaseModel
from collections.abc import Callable

import ollama

class ToolCall(BaseModel):
    name: str
    arguments: dict


class Message(BaseModel):
  role: str
  content: str
  tool_calls: list[ToolCall]


def _patch_response(response, tools):
  message = Message.model_validate_json(response.message.content)
  message.role = 'assistant'
  if len(tools) == 0:
    message.tool_calls = None
  response.message = message
  return response


def _generator_chat(response, tools):
  whole = None
  for chunk in response:
    if whole is None:
      whole = chunk
    else:
      whole.message.content += chunk.message.content
    if chunk.done:
      patched_chunk = _patch_response(whole, tools)
      chunk.message.tool_calls = patched_chunk.message.tool_calls
    yield chunk


def chat(model: str, messages: list, tools: list=[],
         chat_func: Callable=ollama.chat, **kwargs):
  if messages[0]['role'] != 'system' and len(tools):
    messages = [{
      'role': 'system',
      'content': f'Tools available: {tools}'
    }] + messages
  elif str(tools) not in messages[0]['content']:
    messages[0]['content'] += f'\n\nTools available: {tools}'
  response = chat_func(
    messages=messages,
    model=model,
    format=Message.model_json_schema(),
    **kwargs
  )
  if kwargs.get('stream', False):
    return _generator_chat(response, tools)
  else:
    return _patch_response(response, tools)


class Client(ollama.Client):
  def chat(self, *args, **kwargs):
    return chat(chat_func=super().chat, *args, **kwargs)


class AsyncClient(ollama.AsyncClient):
  async def chat(self, model, messages, tools=[], *args, **kwargs):
    if messages[0]['role'] != 'system' and len(tools):
      messages = [{
        'role': 'system',
        'content': f'Tools available: {tools}'
      }] + messages
    elif str(tools) not in messages[0]['content']:
      messages[0]['content'] += f'\n\nTools available: {tools}'
    response = super().chat(
      messages=messages,
      model=model,
      format=Message.model_json_schema(),
      **kwargs
    )
    if kwargs.get('stream', False):
      return self._generator_chat(response, tools)
    else:
      return _patch_response(response, tools)

  async def _generator_chat(self, response, tools):
    whole = None
    async for chunk in await response:
      if whole is None:
        whole = chunk
      else:
        whole.message.content += chunk.message.content
      if chunk.done:
        patched_chunk = _patch_response(whole, tools)
        chunk.message.tool_calls = patched_chunk.message.tool_calls
      yield chunk
