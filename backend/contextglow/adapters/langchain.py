from langchain.core.callbacks import BaseCallbackHandler
from contextglow.core import ContextGlow

class ContextGlowCallbackHandler(BaseCallbackHandler):
    """
    A LangChain callback handler that tracks and streams 
    attention weights using the ContextGlow SDK.
    """
    def __init__(self, model_name: str = "gpt2", device: str = "cpu"):
        super().__init__()
        # Initialize your decoupled SDK core
        self.glow = ContextGlow(model_name=model_name, device=device)
        self.current_context = []

    def on_retriever_end(self, documents, **kwargs):
        """
        Triggered when the vector DB finishes fetching documents.
        We intercept the documents to format them into tagged sentences.
        """
        # Extract the raw text from LangChain Document objects
        self.current_context = [doc.page_content for doc in documents]
        
    def on_llm_start(self, serialized, prompts, **kwargs):
        """
        Triggered right before the model starts generating.
        We can prepare the ContextGlow tensor extraction here.
        """
        self.query = prompts[0]
        
        # We start the generator from your SDK
        self.attention_stream = self.glow.generate(
            query=self.query,
            context=self.current_context,
            stream=True
        )

    def on_llm_new_token(self, token: str, **kwargs):
        """
        Triggered for every new generated token.
        We step the ContextGlow generator and yield the heatmap data.
        """
        try:
            # Step the SDK generator forward
            payload = next(self.attention_stream)
            
            # Here you would emit the payload to your Next.js frontend via SSE or WebSockets
            # e.g., sse_emitter.send(payload)
            print(f"ContextGlow Payload: {payload}")
            
        except StopIteration:
            pass

    def on_llm_end(self, response, **kwargs):
        """
        Triggered after the LLM execution cycle is completed.
        """
        # Clean up or log the final Context Efficiency Metrics here
        self.current_context = []
