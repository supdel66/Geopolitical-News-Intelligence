"use client";

import { useState } from "react";
import Chatbot from "../components/Chatbot";
import SourceSidebar from "../components/SourceSidebar";
import EdaSidebar from "../components/EdaSidebar";

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentSources, setCurrentSources] = useState<any[]>([]);

  return (
    <div className="flex h-screen bg-[#0a0a0a] text-white overflow-hidden font-sans">

      {/* LEFT COLUMN: Sources View (25%) */}
      <div className="w-1/4 min-w-[300px] h-full">
        <SourceSidebar sources={currentSources} />
      </div>

      {/* CENTER COLUMN: Main Chatbot Interface (50%) */}
      <div className="flex-1 w-1/2 h-full shadow-2xl z-10">
        <Chatbot
          messages={messages}
          setMessages={setMessages}
          onNewSources={setCurrentSources}
        />
      </div>

      {/* RIGHT COLUMN: EDA Stats / DB Snapshot (25%) */}
      <div className="w-1/4 min-w-[250px] h-full hidden lg:block">
        <EdaSidebar />
      </div>

    </div>
  );
}
