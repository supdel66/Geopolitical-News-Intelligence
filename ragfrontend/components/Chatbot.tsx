import { useState, useRef, useEffect } from 'react';

interface Message {
    role: 'user' | 'assistant';
    content: string;
}

export default function Chatbot({
    messages,
    setMessages,
    onNewSources,
}: {
    messages: Message[];
    setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
    onNewSources: (sources: any[]) => void;
}) {
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;

        const userMessage: Message = { role: 'user', content: input };
        setMessages((prev) => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await fetch('http://localhost:8000/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: userMessage.content }),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();

            setMessages((prev) => [
                ...prev,
                { role: 'assistant', content: data.answer },
            ]);

            onNewSources(data.sources);

        } catch (error) {
            console.error('Error fetching chat response:', error);
            setMessages((prev) => [
                ...prev,
                { role: 'assistant', content: 'Sorry, I encountered an error connecting to the backend. Is FastAPI and Ollama running?' },
            ]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-full bg-[#0a0a0a]">
            {/* Header */}
            <div className="p-4 border-b border-[#333] bg-[#111]">
                <h1 className="text-xl font-bold text-white tracking-wide">
                    Geopolitical Intelligence <span className="text-blue-500">RAG</span>
                </h1>
                <p className="text-sm text-gray-400">Powered by LLaMA 3.2:3B & ChromaDB</p>
            </div>

            {/* Chat Messages */}
            <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
                {messages.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full text-gray-500 space-y-4">
                        <svg xmlns="http://www.w3.org/Globe-svg" className="w-16 h-16 opacity-20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <p className="text-lg">Ask a question about the latest news.</p>
                        <div className="flex gap-2">
                            <button
                                onClick={() => setInput("What are the latest developments in the Middle East?")}
                                className="px-3 py-1 bg-[#1a1a1a] border border-[#333] rounded-full text-xs hover:bg-[#222]"
                            >
                                Latest Middle East news?
                            </button>
                            <button
                                onClick={() => setInput("What factors are risking WW3 escalation?")}
                                className="px-3 py-1 bg-[#1a1a1a] border border-[#333] rounded-full text-xs hover:bg-[#222]"
                            >
                                WW3 Escalation risks
                            </button>
                        </div>
                    </div>
                ) : (
                    <div className="flex flex-col gap-6">
                        {messages.map((msg, idx) => (
                            <div
                                key={idx}
                                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                            >
                                <div
                                    className={`max-w-[80%] rounded-2xl p-4 ${msg.role === 'user'
                                            ? 'bg-blue-600 text-white rounded-br-sm'
                                            : 'bg-[#1a1a1a] border border-[#333] text-gray-200 rounded-bl-sm shadow-lg'
                                        }`}
                                >
                                    {msg.role === 'assistant' && (
                                        <div className="flex items-center gap-2 mb-2 pb-2 border-b border-[#333]">
                                            <div className="w-6 h-6 rounded-full bg-blue-900 flex items-center justify-center text-xs font-bold text-white">AI</div>
                                            <span className="text-xs font-semibold text-gray-400">Analysis</span>
                                        </div>
                                    )}
                                    <div className="whitespace-pre-wrap leading-relaxed">{msg.content}</div>
                                </div>
                            </div>
                        ))}
                        {isLoading && (
                            <div className="flex justify-start">
                                <div className="bg-[#1a1a1a] border border-[#333] p-4 rounded-2xl rounded-bl-sm flex items-center gap-3">
                                    <div className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '0ms' }}></div>
                                    <div className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '150ms' }}></div>
                                    <div className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '300ms' }}></div>
                                    <span className="text-xs text-gray-400 ml-2 animate-pulse">Consulting documents...</span>
                                </div>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>
                )}
            </div>

            {/* Input Area */}
            <div className="p-4 bg-[#111] border-t border-[#333]">
                <form onSubmit={handleSubmit} className="flex gap-2">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Search the news database..."
                        className="flex-1 bg-[#1a1a1a] text-white border border-[#333] rounded-lg px-4 py-3 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all placeholder-gray-500"
                        disabled={isLoading}
                    />
                    <button
                        type="submit"
                        disabled={!input.trim() || isLoading}
                        className="bg-blue-600 hover:bg-blue-500 text-white px-6 py-3 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                    >
                        <span>Ask</span>
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                        </svg>
                    </button>
                </form>
            </div>
        </div>
    );
}
