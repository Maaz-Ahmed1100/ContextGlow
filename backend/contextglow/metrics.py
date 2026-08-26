def get_utilization_score(attention_map, threshold=0.10):
    """
    Calculates what percentage of the provided context chunks actually influenced the output.
    A chunk is considered 'utilized' if its attention score is above the threshold.
    """
    if not attention_map:
        return 0.0
        
    utilized = sum(1 for score in attention_map.values() if score > threshold)
    return utilized / len(attention_map)

def get_ignored_chunks(attention_map, threshold=0.10):
    """
    Returns chunks that the LLM completely ignored (attention below threshold).
    """
    return [chunk_id for chunk_id, score in attention_map.items() if score <= threshold]

def detect_middle_drop(attention_map, threshold=0.15):
    """
    Analyzes the distribution to flag if the 'Lost in the Middle' phenomenon occurred.
    (High attention on head/tail, low attention on middle chunks).
    """
    if len(attention_map) < 3:
        return False
        
    # Sort chunks to ensure correct ordering if they are named chunk_0_sent_N
    chunks = sorted(attention_map.keys())
    scores = [attention_map[c] for c in chunks]
    
    # Check head (first 25%) and tail (last 25%) vs middle
    third_idx = max(1, len(scores) // 3)
    
    head_scores = scores[:third_idx]
    tail_scores = scores[-third_idx:]
    middle_scores = scores[third_idx:-third_idx] if len(scores) > 2 else []
    
    if not middle_scores:
        return False
        
    head_max = max(head_scores) if head_scores else 0
    tail_max = max(tail_scores) if tail_scores else 0
    mid_max = max(middle_scores) if middle_scores else 0
    
    # If middle max is significantly lower than head and tail max
    if mid_max < threshold and (head_max > 0.5 or tail_max > 0.5):
        return True
        
    return False
