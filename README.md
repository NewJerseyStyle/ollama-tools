# ollama-tools

A workaround for models on Ollama that does not support tool calling. This package provides a temporary solution to use those models with tools.

Also works as Ollama server wrapper with `ollama-tools --host 0.0.0.0 --port 22434` to proxy the Ollama server with extra support for tool calling with models such as DeepSeek-R1.

## Installation
Install Ollama and get a model
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull deepseek-r1:1.5b
```
> ℹ️ Above Bash commands shows installation on Linux, installing Ollama on other operating systems refer to https://ollama.com/download

Install the package for Python binding
```bash
pip install ollama-tools
```

## Example Usage
### Develop in Python
1. Import the package
```python
import ollama_tools
```

2. Simple chat without tools
```python
response = ollama_tools.chat(
    model='deepseek-r1:1.5b',
    messages=[{'role': 'user', 'content': 'how is weather in Paris?'}])

print(response.message.content)
'''
# Example output:
Message(role='assistant', content="I don't have access to the current weather data for Paris. To find out the latest weather in Paris, you can check official weather websites or apps specifically designed for Paris, such as METAR or AccuWeather.", tool_calls=[])
'''
```

3. Chat with tools
```python
tools=[
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the weather for, e.g. San Francisco, CA"
                    },
                    "format": {
                        "type": "string",
                        "description": "The format to return the weather in, e.g. 'celsius' or 'fahrenheit'",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location", "format"]
            }
        }
    }
]
  
response = ollama_tools.chat(
    model='deepseek-r1:1.5b',
    messages=[{'role': 'user', 'content': 'how is weather in Paris?'}],
    tools=tools)

print(response)
'''
# Example output:
ChatResponse(model='deepseek-r1:1.5b', created_at='2025-01-31T06:09:22.095306834Z', done=True, done_reason='stop', total_duration=11153340905, load_duration=33513150, prompt_eval_count=138, prompt_eval_duration=1300000000, eval_count=265, eval_duration=9285000000, message=Message(role='assistant', content="To determine the current weather in Paris, you can use the following information: Paris is a significant city known for its historical landmarks, vibrant nightlife, and modern architecture. The local climate is generally milder than that of many other cities, with temperatures typically ranging between 20°C to 35°C (68°F to 95°F). However, Paris experiences occasional weather changes due to the region's natural geography and urbanization. On average, the rainy season in Paris can last about six months from June through September, while the dry season occurs during November through April. These seasonal changes may influence local rainfall patterns, which could impact weather-related activities like agriculture and transportation. Additionally, Paris has a strong presence of international students and professionals, who bring a diverse set of experiences into the city's daily life. As for specific weather information, you might want to check the official weather websites or weather apps that provide the most up-to-date and accurate forecasts. They typically offer detailed hourly updates and even short-term projections, which can help you plan your activities accordingly.", tool_calls=[ToolCall(name='get_current_weather', arguments={'location': 'Paris'})])
'''
```

## AsyncClient
> ⚠️ The behaviour of `ollama_tools.AsyncClient` is different from `ollama.AsyncClient` when `stream=True` is enabled, the for-loop `async for part in AsyncClient()` does not need `await` before `AsyncClient`
```python
import asyncio
from ollama_tools import AsyncClient

async def chat():
  message = {'role': 'user', 'content': 'Why is the sky blue?'}
  async for part in AsyncClient().chat(model='deepseek-r1:0.5b', messages=[message], stream=True):
    #             ^^^ With `Ollama` it will be `await AsyncClient()`
    print(part['message']['content'], end='', flush=True)

asyncio.run(chat())
```

## Ollama Wrapper (Server)
Install the package with Ollama wrapper
```bash
pip install ollama-tools[full]
ollama-tools --host 0.0.0.0 --port 22434
```

Now we can use this in `llama-index` or `langchain`
```python
### llama_index ==================
from llama_index.llms.ollama import Ollama

llm = Ollama(
    model="deepseek-r1",
    base_url="http://localhost:22434", # as we started the server with `--port 22434`
    request_timeout=60.0
)

response = llm.complete("What is the capital of France?")
print(response)


### langchain_ollama ==================
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="deepseek-r1",
    temperature=0,
    base_url="http://localhost:22434"
)

from langchain_core.messages import AIMessage

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
print(ai_msg)
```
