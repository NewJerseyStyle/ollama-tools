import os
import json
import base64
import tempfile
import argparse
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import ollama_tools as ollama
try:
    from fastapi import FastAPI, Request, status
    from fastapi.responses import StreamingResponse
    import uvicorn
except ImportError:
    import sys
    sys.exit("The 'fastapi' and 'uvicorn' is not installed. Please manual install or "
            "install by `pip install ollama-tools[full]` to use this feature.")

app = FastAPI()

def chat_response_to_dict(response):
    """Converts a ChatResponse object to a dict."""
    data = json.loads(response.json())
    data['message'] = json.loads(response.message.json())
    if getattr(response.message, "images", False):
        data['message']["images"] = json.loads(response.message.images.json())
    if (getattr(response.message, "tool_calls", False) and
        len(response.message.tool_calls)):
        data['message']["tool_calls"] = [json.loads(call.json())
            for call in response.message.tool_calls]
    return data


def model_exists(model):
    for _model in ollama.list().models:
        if _model.model == model:
            return True
    return False


@app.post("/api/generate")
async def generate(request: Request):
    data = await request.json()
    if not model_exists(data.get('model')):
        return {"error": f"model '{data.get('model')}' not found"}
    image_paths = []
    if data.get('images'):
        for image in data.get('images'):
            try:
                image_data = base64.b64decode(image)
                with tempfile.NamedTemporaryFile(suffix=".png",
                                                delete=False) as temp_file:
                    temp_file.write(image_data)
                    image_paths.append(temp_file.name)
            except Exception as e:
                print(f"Error decoding image: {e}")
                return {"error": f"Error decoding image: {e}"}
    response = ollama.generate(**data)
    #clean up temp files
    for image_path in image_paths:
        os.remove(image_path)
    if data.get('stream'):
        # Return streaming response
        def stream_response():
            for chunk in response:
                yield chunk.json()
        return StreamingResponse(content=stream_response())
    else:
        # Return single response
        return response.json()


@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    if not model_exists(data.get('model')):
        return {"error": f"model '{data.get('model')}' not found"}
    response = ollama.chat(**data)
    if data.get('stream'):
        if data.get('tools'):
            content = ''
            for chunk in response:
                content += chunk.message.content
            chunk.message.content = json.loads(content)['content']
            data = chat_response_to_dict(chunk)
            return data
        def stream_response():
            for chunk in response:
                yield json.dumps(chat_response_to_dict(chunk))
        return StreamingResponse(content=stream_response())
    else:
        return chat_response_to_dict(response)


@app.post("/api/show")
async def show_model(request: Request):
    data = await request.json()
    for _model in ollama.list().models:
        if _model.model == data.get('model'):
            data = json.loads(_model.json())
            data['details'] = json.loads(_model.details.json())
            return data
    return {"error": f"model '{data.get('model')}' not found"}
    

@app.post("/api/tags")
async def list_model():
    info = []
    for model in ollama.list().models:
        data = json.loads(model.json())
        data['details'] = json.loads(model.details.json())
        info.append(data)
    return {'models': info}


@app.post("/api/embed")
@app.post("/api/embeddings")
async def generate_embedding(request: Request):
    data = await request.json()
    if not model_exists(data.get('model')):
        return {"error": f"model '{data.get('model')}' not found"}
    if data.get('input'):
        response = ollama.embed(**data)
        return response.json()
    return {"embedding":[]}


def main():
    parser = argparse.ArgumentParser(description="Run the Ollama tools server.")
    parser.add_argument("--host", default="localhost", help="Host to listen on")
    parser.add_argument("--port", type=int, default=22434, help="Port to listen on")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)