import json
import time
import torch
import inseq
from typing import List, Dict, Any, Generator, Union

from .utils import split_into_sentences, map_attention_to_sentences
from .metrics import get_utilization_score, get_ignored_chunks, detect_middle_drop

class ContextGlow:
    """
    ContextGlow SDK: Drop-in middleware for adding RAG attention heatmaps.
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = None):
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"Loading ContextGlow Engine (Model: {model_name}, Device: {self.device})...")
        # Load the model with attention attribution
        self.model = inseq.load_model(model_name, "attention", device=self.device)
        self.tokenizer = self.model.tokenizer

    def generate(self, query: str, context: Union[str, List[str]], stream: bool = True, max_new_tokens: int = 20):
        """
        Generates a response while calculating cross-attention mapped to context sentences.
        """
        # 1. Standardize and chunk context
        if isinstance(context, list):
            context_str = " ".join(context)
        else:
            context_str = context
            
        chunks = split_into_sentences(context_str)
        
        # Format the prompt
        full_prompt = f"Context: {context_str}\n\nQuery: {query}"
        
        # Map sub-word tokens back to chunk IDs
        sentence_id_mapping = {}
        try:
            encoded = self.tokenizer(full_prompt, return_offsets_mapping=True)
            offsets = encoded.get("offset_mapping", [])
            for token_idx, (start_char, end_char) in enumerate(offsets):
                for chunk in chunks:
                    if start_char < chunk["end"] and end_char > chunk["start"]:
                        sentence_id_mapping[token_idx] = chunk["id"]
                        break
        except Exception:
            pass # Fallback if tokenizer doesn't support offset mapping
            
        # 2. Yield initial context state (for frontend rendering)
        initial_payload = {"context": chunks}
        if stream:
            yield f"data: {json.dumps(initial_payload)}\n\n"
            
        # 3. Generate tokens & extract 4D attention tensor
        out = self.model.attribute(
            full_prompt,
            generation_args={"max_new_tokens": max_new_tokens},
            show_progress=False
        )
        
        attr_out = out.sequence_attributions[0]
        target_tokens = [t.token for t in attr_out.target] if hasattr(attr_out, 'target') else [t.token for t in attr_out.target_tokens]
        source_tokens = [t.token for t in attr_out.source] if hasattr(attr_out, 'source') else [t.token for t in attr_out.source_tokens]
        
        # Shape: [source_len, gen_len, layers, heads]
        raw_attr = attr_out.target_attributions
        
        # Aggregate across layers and heads -> [source_len, gen_len]
        target_attributions = raw_attr.mean(dim=-1).mean(dim=-1)
        
        num_source = len(source_tokens)
        num_generated = target_attributions.shape[1]
        
        # 4. Stream response
        final_attention_map = {}
        
        for i in range(num_generated):
            token = target_tokens[num_source + i] if (num_source + i) < len(target_tokens) else ''
            clean_token = token.replace("Ġ", " ").replace("Ċ", "\n")
            
            step_attention = target_attributions[:, i]
            attention_map = map_attention_to_sentences(step_attention, source_tokens, sentence_id_mapping)
            
            # Keep track of the final overall attention
            for k, v in attention_map.items():
                final_attention_map[k] = max(final_attention_map.get(k, 0), v)
                
            payload = {
                "token": clean_token,
                "attention_map": attention_map
            }
            
            if stream:
                yield f"data: {json.dumps(payload)}\n\n"
                time.sleep(0.05) # Simulate stream for UX
                
        # 5. Compute diagnostics metrics
        metrics = {
            "utilization_score": get_utilization_score(final_attention_map),
            "ignored_chunks": get_ignored_chunks(final_attention_map),
            "middle_drop_detected": detect_middle_drop(final_attention_map)
        }
        
        if stream:
            yield f"data: {json.dumps({'metrics': metrics, 'done': True})}\n\n"
        else:
            yield {
                "context": chunks,
                "generated_text": "".join([t.replace("Ġ", " ").replace("Ċ", "\n") for t in target_tokens[num_source:]]),
                "final_attention": final_attention_map,
                "metrics": metrics
            }
