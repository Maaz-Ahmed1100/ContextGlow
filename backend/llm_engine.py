# this script sets up the local language model and feature attribution
import torch
import inseq

class LLMEngine:
    def __init__(self, model_name="gpt2"):
        # we check for the best available hardware device
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
            
        print(f"loading model on {self.device}")
        
        # we load the model directly with inseq wrapper
        # we use the attention method because it is fast for testing
        self.inseq_model = inseq.load_model(
            model_name,
            "attention",
            device=self.device
        )
        
    def get_model(self):
        # we return the loaded model for our fast api routes to use
        return self.inseq_model

# we create a global instance to load the model just once
engine = None

def get_llm():
    # this helper function returns the model instance
    global engine
    if engine is None:
        engine = LLMEngine("gpt2")
    return engine.get_model()

def generate_with_attribution(prompt_template, retrieved_sentences):
    # we get our loaded inseq model
    model = get_llm()
    
    # we format the retrieved sentences into one text block
    # we check if sentences are dicts from chromadb or just strings
    context = " ".join([s["text"] if isinstance(s, dict) else s for s in retrieved_sentences])
    
    # we add the context and the prompt together
    full_input = f"Context: {context}\n\nPrompt: {prompt_template}"
    
    print("running attribution generation")
    
    # we tell inseq to generate tokens and map their attention back to the input
    # we keep max new tokens small to make it run fast locally
    out = model.attribute(
        full_input,
        generation_args={"max_new_tokens": 5}
    )
    
    # we print the raw tensor output for verification
    # this shows the weights linking generated tokens to input tokens
    print("raw attribution tensor:")
    print(out.sequence_attributions[0].target_attributions)
    
    return out
