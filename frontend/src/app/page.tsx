import React from 'react';

// this is our mock data for the context viewer
const mockContext = [
  { id: 'chunk_0_sent_0', text: 'Space is the boundless three-dimensional extent in which objects and events have relative position and direction.' },
  { id: 'chunk_0_sent_1', text: 'In classical physics physical space is often conceived in three linear dimensions.' },
  { id: 'chunk_0_sent_2', text: 'The concept of space is considered to be of fundamental importance to an understanding of the physical universe.' }
];

// this is some mock chat history
const mockChat = [
  { role: 'user', content: 'What is space?' },
  { role: 'assistant', content: 'Space is a three dimensional extent where things happen.' }
];

export default function Home() {
  return (
    <div className="flex h-screen w-full bg-gray-950 text-gray-100 overflow-hidden font-sans">
      
      {/* left column for context viewer */}
      <div className="w-[30%] h-full border-r border-white/10 bg-white/5 backdrop-blur-md p-6 overflow-y-auto flex flex-col gap-4">
        <h2 className="text-xl font-semibold mb-4 text-white/90">Context Viewer</h2>
        
        {mockContext.map((item) => (
          <div key={item.id} className="p-4 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 transition-colors">
            <div className="text-xs text-blue-400 mb-2 font-mono">{item.id}</div>
            <p className="text-sm leading-relaxed text-gray-300">{item.text}</p>
          </div>
        ))}
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
          {mockChat.map((msg, idx) => (
            <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[70%] p-5 rounded-2xl backdrop-blur-md border ${
                msg.role === 'user' 
                  ? 'bg-blue-600/20 border-blue-500/30 text-blue-50 rounded-br-none' 
                  : 'bg-white/5 border-white/10 text-gray-200 rounded-bl-none'
              }`}>
                <p className="text-sm leading-relaxed">{msg.content}</p>
              </div>
            </div>
          ))}
        </div>

        {/* sticky input field at bottom */}
        <div className="p-6 bg-gray-950/80 backdrop-blur-xl border-t border-white/10">
          <div className="relative max-w-4xl mx-auto">
            <input 
              type="text" 
              placeholder="Ask a question about the context" 
              className="w-full bg-white/5 border border-white/10 rounded-full py-4 pl-6 pr-16 text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all shadow-lg"
            />
            <button className="absolute right-2 top-2 bottom-2 aspect-square rounded-full bg-blue-600 hover:bg-blue-500 text-white flex items-center justify-center transition-colors">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" clipRule="evenodd" />
              </svg>
            </button>
          </div>
        </div>

      </div>
    </div>
  );
}
