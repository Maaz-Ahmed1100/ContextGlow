# this is our main server file
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import time

from retriever import retrieve_context
from llm_engine import get_llm, map_attention_to_sentences

app = FastAPI()

# we set up cors so our nextjs app can talk to us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            yield f"data: {json.dumps({'token': 'No context found.', 'attention_map': {}})}\n\n"
            return
            
        model = get_llm()
        tokenizer = model.tokenizer
        
        # we build the prompt and keep track of where each sentence is
        context_parts = []
        sentence_spans = []
        current_char = len("Context: ")
        
        for s in retrieved:
            text = s["text"]
            context_parts.append(text)
            start = current_char
            end = current_char + len(text)
            sentence_spans.append({"id": s["id"], "start": start, "end": end})
            # we add one for the space we will join with
            current_char = end + 1
            
        context_str = " ".join(context_parts)
        full_prompt = f"Context: {context_str}\n\nQuery: {query}"
        
        # we map tokens to sentences using offset mapping if available
        sentence_id_mapping = {}
        try:
            # this tells us the character positions for each token
            encoded = tokenizer(full_prompt, return_offsets_mapping=True)
            offsets = encoded.get("offset_mapping", [])
            for token_idx, (start_char, end_char) in enumerate(offsets):
                for span in sentence_spans:
                    # we check if the token overlaps with the sentence text
                    if start_char < span["end"] and end_char > span["start"]:
                        sentence_id_mapping[token_idx] = span["id"]
                        break
        except Exception:
            # fallback if offset mapping fails
            pass
            
        # we generate the text and attributions
        # we limit to 20 tokens to keep things fast for local testing
        out = model.attribute(
            full_prompt,
            generation_args={"max_new_tokens": 20},
            show_progress=False
        )
        
        attr_out = out.sequence_attributions[0]
        target_tokens = attr_out.target_tokens
        target_attributions = attr_out.target_attributions
        source_tokens = attr_out.source_tokens
        
        # we stream the tokens one by one
        for i, token in enumerate(target_tokens):
            # we grab the attention weights for this specific step
            step_attention = target_attributions[:, i]
            
            # we map those weights to our original sentences
            attention_map = map_attention_to_sentences(step_attention, source_tokens, sentence_id_mapping)
            
            # we clean up the token string formatting
            clean_token = token.replace("Ġ", " ").replace("Ċ", "\n")
            
            data = {
                "token": clean_token,
                "attention_map": attention_map
            }
            yield f"data: {json.dumps(data)}\n\n"
            # we wait a tiny bit to act like a real stream
            time.sleep(0.1)
            
    return StreamingResponse(sse_generator(), media_type="text/event-stream")
