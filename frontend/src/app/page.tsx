'use client';

import React, { useState } from 'react';
import { useAttentionStream } from '../hooks/useAttentionStream';

// we store messages as objects
type Message = { role: 'user' | 'assistant'; content: string };

export default function Home() {
  // we set up our hook to talk to the backend
  const { generatedText, currentAttention, contextSentences, isStreaming, startStream } = useAttentionStream();
  
  // we keep track of the user input and the chat history
  const [inputValue, setInputValue] = useState('');
  const [chatHistory, setChatHistory] = useState<Message[]>([]);

  // this function runs when the user submits their question
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isStreaming) return;
    
    // we add the user message to the screen
    setChatHistory(prev => [...prev, { role: 'user', content: inputValue }]);
    
    // we start streaming the answer from the backend
    startStream(inputValue);
    setInputValue('');
  };

  return (
    <div className="flex h-screen w-full bg-gray-950 text-gray-100 overflow-hidden font-sans">
      
      {/* left column for context viewer */}
      <div className="w-[30%] h-full border-r border-white/10 bg-white/5 backdrop-blur-md p-6 overflow-y-auto flex flex-col gap-4">
        <h2 className="text-xl font-semibold mb-4 text-white/90">Context Viewer</h2>
        
        {contextSentences.length === 0 && (
          <p className="text-sm text-gray-500 italic">ask a question to load context</p>
        )}

        {contextSentences.map((item) => {
          // we get the score from the backend zero if missing
          const score = currentAttention[item.id] || 0.0;
          
          return (
            <div 
              key={item.id} 
              className="p-4 rounded-xl border border-white/10 transition-all duration-300"
              style={{
                // we set the background color dynamically based on the score
                backgroundColor: `rgba(239, 68, 68, ${score * 0.8})`
              }}
            >
              <div className="text-xs text-blue-300 mb-2 font-mono flex justify-between">
                <span>{item.id}</span>
                <span>{score.toFixed(2)}</span>
              </div>
              <p className="text-sm leading-relaxed text-gray-100 drop-shadow-md">{item.text}</p>
            </div>
          );
        })}
      </div>

      {/* right column for chat interface */}
      <div className="w-[70%] h-full flex flex-col relative bg-gradient-to-br from-gray-950 to-gray-900">
        
        {/* header */}
        <div className="h-16 border-b border-white/10 bg-white/5 backdrop-blur-sm flex items-center px-8 z-10 sticky top-0">
          <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
            RAG Attention Heatmapper
          </h1>
        </div>

        {/* chat history area */}
        <div className="flex-1 overflow-y-auto p-8 space-y-6">
          {chatHistory.map((msg, idx) => (
            <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[70%] p-5 rounded-2xl backdrop-blur-md border ${
                msg.role === 'user' 
                  ? 'bg-blue-600/20 border-blue-500/30 text-blue-50 rounded-br-none' 
                  : 'bg-white/5 border-white/10 text-gray-200 rounded-bl-none'
              }`}>
                <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>
              </div>
            </div>
          ))}
          
          {/* we show the streaming text if it exists */}
          {(isStreaming || generatedText) && (
            <div className="flex justify-start">
              <div className="max-w-[70%] p-5 rounded-2xl backdrop-blur-md border bg-white/5 border-white/10 text-gray-200 rounded-bl-none">
                <p className="text-sm leading-relaxed whitespace-pre-wrap">
                  {generatedText}
                  {isStreaming && <span className="inline-block w-2 h-4 ml-1 bg-blue-400 animate-pulse"></span>}
                </p>
              </div>
            </div>
          )}
        </div>

        {/* sticky input field at bottom */}
        <div className="p-6 bg-gray-950/80 backdrop-blur-xl border-t border-white/10">
          <form onSubmit={handleSubmit} className="relative max-w-4xl mx-auto">
            <input 
              type="text" 
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask a question about the context" 
              className="w-full bg-white/5 border border-white/10 rounded-full py-4 pl-6 pr-16 text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all shadow-lg"
              disabled={isStreaming}
            />
            <button 
              type="submit"
              disabled={isStreaming || !inputValue.trim()}
              className="absolute right-2 top-2 bottom-2 aspect-square rounded-full bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 text-white flex items-center justify-center transition-colors"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" clipRule="evenodd" />
              </svg>
            </button>
          </form>
        </div>

      </div>
    </div>
  );
}
