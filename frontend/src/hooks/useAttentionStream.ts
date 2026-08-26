import { useState, useCallback } from 'react';

// this hook manages our custom server sent events stream
export function useAttentionStream() {
  // we keep track of the text as it streams in
  const [generatedText, setGeneratedText] = useState<string>('');
  // we keep track of the latest attention weights for each sentence
  const [currentAttention, setCurrentAttention] = useState<Record<string, number>>({});
  // we store the retrieved sentences here
  const [contextSentences, setContextSentences] = useState<Array<{id: string, text: string}>>([]);
  // we keep track of whether we are currently loading
  const [isStreaming, setIsStreaming] = useState<boolean>(false);

  const startStream = useCallback(async (query: string) => {
    // we reset our state before starting a new stream
    setGeneratedText('');
    setCurrentAttention({});
    setContextSentences([]);
    setIsStreaming(true);

    try {
      // we call our python fast api backend
      const response = await fetch('http://localhost:8080/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });

      if (!response.body) {
        throw new Error('no response body found');
      }

      // we read the stream exactly as it comes in
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let buffer = '';

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;

        if (value) {
          // we add the new raw bytes to our string buffer
          buffer += decoder.decode(value, { stream: true });
          
          // sse separates events with double newlines
          const events = buffer.split('\n\n');
          // the last item might be incomplete so we save it for the next chunk
          buffer = events.pop() || '';
          
          for (const event of events) {
            // we strip out the data prefix
            if (event.startsWith('data: ')) {
              const dataStr = event.substring(6).trim();
              if (dataStr) {
                try {
                  // we parse the json payload from our python backend
                  const parsed = JSON.parse(dataStr);
                  
                  // we append the new token
                  if (parsed.token) {
                    setGeneratedText((prev) => prev + parsed.token);
                  }
                  
                  // we replace the old attention map with the new one
                  if (parsed.attention_map) {
                    setCurrentAttention(parsed.attention_map);
                  }
                  
                  // we save the retrieved sentences if the backend sent them
                  if (parsed.context) {
                    setContextSentences(parsed.context);
                  }
                } catch (e) {
                  console.error('error parsing chunk data', e);
                }
              }
            }
          }
        }
      }
    } catch (error) {
      console.error('error while streaming', error);
    } finally {
      // we make sure to mark the stream as finished
      setIsStreaming(false);
    }
  }, []);

  return {
    generatedText,
    currentAttention,
    contextSentences,
    isStreaming,
    startStream,
  };
}
