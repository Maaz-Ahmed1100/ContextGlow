import re
import uuid

def split_into_sentences(text: str):
    """
    Splits raw text into sentences and assigns them deterministic IDs.
    Returns a list of dicts: {"id": str, "text": str, "start": int, "end": int}
    """
    # Basic regex to split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    chunks = []
    current_char = 0
    
    for i, sent in enumerate(sentences):
        if not sent.strip():
            continue
            
        chunk_id = f"chunk_0_sent_{i}"
        start = current_char
        end = start + len(sent)
        
        chunks.append({
            "id": chunk_id,
            "text": sent,
            "start": start,
            "end": end
        })
        
        current_char = end + 1 # account for the space
        
    return chunks

def map_attention_to_sentences(attention_tensor, source_tokens, sentence_id_mapping):
    """
    Maps 1D token attention weights back to sentence IDs and normalizes them.
    """
    sentence_scores = {}
    
    # Extract weights if tensor
    if hasattr(attention_tensor, "tolist"):
        weights = attention_tensor.tolist()
    else:
        weights = attention_tensor
        
    # Aggregate weights by sentence ID
    for i, weight in enumerate(weights):
        sent_id = None
        if isinstance(sentence_id_mapping, dict):
            sent_id = sentence_id_mapping.get(i)
        elif i < len(sentence_id_mapping):
            sent_id = sentence_id_mapping[i]
            
        if sent_id is not None:
            if sent_id not in sentence_scores:
                sentence_scores[sent_id] = []
            sentence_scores[sent_id].append(weight)
            
    # Sum weights and normalize max to 1.0
    final_scores = {}
    max_val = 0.0
    
    for sent_id, vals in sentence_scores.items():
        total = sum(vals)
        final_scores[sent_id] = total
        if total > max_val:
            max_val = total
            
    if max_val > 0:
        for sent_id in final_scores:
            final_scores[sent_id] = final_scores[sent_id] / max_val
            
    return final_scores
