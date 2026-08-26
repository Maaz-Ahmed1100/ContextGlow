# this is our main server file
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

from retriever import retrieve_context
from contextglow import ContextGlow

app = FastAPI()

# we set up cors so our nextjs app can talk to us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# initialize our new SDK globally
glow = ContextGlow(model_name="gpt2", device="cpu")

# simple health check route
@app.get("/")
def read_root():
    return {"status": "ok"}

class ChatRequest(BaseModel):
    query: str

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    # we create a generator function to send data step by step
    def sse_generator():
        query = request.query
        retrieved = retrieve_context(query)
        
        if not retrieved:
            yield 'data: {"token": "No context found.", "attention_map": {}}\n\n'
            return
            
        # extract just the text for the SDK
        context_sentences = [s["text"] for s in retrieved]
        
        # we still send the retrieved sentences to the frontend first so it can display them
        # (the SDK chunks them too, but we need to pass the IDs back)
        context_payload = {"context": retrieved}
        yield f"data: {json.dumps(context_payload)}\n\n"
        
        # let the SDK do all the heavy lifting!
        generator = glow.generate(query, context_sentences, stream=True, max_new_tokens=20)
        
        for payload in generator:
            yield payload
            
    return StreamingResponse(sse_generator(), media_type="text/event-stream")
