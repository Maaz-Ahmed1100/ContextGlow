# ContextGlow 🌟

**ContextGlow** is a plug-and-play Python middleware and React component that visualizes exactly what your RAG (Retrieval-Augmented Generation) pipeline is "thinking" about in real-time. 

Drop it into your existing LangChain, LlamaIndex, or custom FastAPI pipeline in 3 lines of code to generate gorgeous, real-time attention heatmaps that expose *why* the AI answered the way it did.

![ContextGlow Heatmap Demo](./demo.jpg)

## Why ContextGlow?

RAG applications often fail silently. When a model hallucinates, developers usually blame the vector database retriever. But often, the retriever works perfectly, and the LLM just **ignores the context**.

ContextGlow extracts the raw, 4D cross-attention tensors during generation and dynamically maps them back to your source sentences, giving you:
- **X-Ray Vision**: See exactly which sentence the LLM is focusing on, token by token.
- **Utilization Scoring**: Measure how much of the retrieved context actually influenced the final answer.
- **Lost In The Middle Detection**: Automatically flag if the LLM completely ignored the middle chunks of your context.

## 🚀 Quick Start

### 1. Install the SDK
```bash
pip install contextglow
npm install @contextglow/react
```

### 2. Add to your Backend (LangChain Example)
ContextGlow ships with native adapters. You don't need to rewrite your pipeline.

```python
from contextglow.adapters.langchain import ContextGlowCallbackHandler
from langchain.chains import RetrievalQA

# Initialize the adapter
glow_handler = ContextGlowCallbackHandler(model_name="meta-llama/Llama-3-8B")

# Pass it into your existing LangChain setup!
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    callbacks=[glow_handler] # ContextGlow silently extracts attention here!
)

qa_chain.run("what is spacetime")
```

### 3. Add to your Frontend (React)
```tsx
import { ContextViewer, useAttentionStream } from '@contextglow/react';

export default function App() {
  const { context, attentionMap } = useAttentionStream('/api/chat');

  return (
    <ContextViewer 
      documents={context} 
      attentionMap={attentionMap} 
    />
  );
}
```

## Features

- **Model Agnostic**: Supports any HuggingFace model (Llama-3, Mistral, Phi-3).
- **Streaming Native**: Heatmaps update live as tokens stream to the user.
- **Low Overhead**: Operates entirely in inference mode without gradients.

## Coming Soon
- API Model Support (OpenAI / Anthropic) via self-reflection.
- LlamaIndex Adapters.

## License
MIT
