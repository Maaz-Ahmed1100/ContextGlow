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
