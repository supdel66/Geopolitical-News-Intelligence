import { useState } from 'react';

interface ArticleInfo {
    id: number;
    title: string;
    source: string;
    url?: string;
    img_link?: string;
    published_at?: string;
    content_snippet?: string;
    similarity_distance?: number;
}

export default function SourceSidebar({ sources }: { sources: ArticleInfo[] }) {
    if (!sources || sources.length === 0) {
        return (
            <div className="h-full bg-[#111] border-r border-[#333] p-4 text-gray-400">
                <h2 className="text-xl font-bold mb-4 text-white">Sources</h2>
                <p className="text-sm">No sources available for the current query.</p>
            </div>
        );
    }

    return (
        <div className="h-full bg-[#111] border-r border-[#333] p-4 overflow-y-auto custom-scrollbar">
            <h2 className="text-xl font-bold mb-4 text-white sticky top-0 bg-[#111] pb-2 z-10">
                Sources
            </h2>
            <div className="flex flex-col gap-4">
                {sources.map((source, idx) => (
                    <div
                        key={`${source.id}-${idx}`}
                        className="bg-[#1a1a1a] rounded-lg border border-[#333] p-3 hover:border-[#555] transition-colors cursor-pointer group"
                    >
                        {source.img_link && (
                            <div className="w-full h-32 mb-3 rounded overflow-hidden">
                                <img
                                    src={source.img_link}
                                    alt={source.title}
                                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                                    onError={(e) => {
                                        // Fallback if image fails to load
                                        (e.target as HTMLImageElement).style.display = 'none';
                                    }}
                                />
                            </div>
                        )}
                        <div className="flex justify-between items-start mb-2">
                            <span className="text-xs font-semibold px-2 py-1 bg-blue-900/50 text-blue-300 rounded">
                                {source.source}
                            </span>
                            {source.similarity_distance && (
                                <span className="text-xs text-gray-500">
                                    Dist: {source.similarity_distance.toFixed(3)}
                                </span>
                            )}
                        </div>
                        <h3 className="text-sm font-bold text-gray-200 mb-2 line-clamp-2">
                            {source.title}
                        </h3>
                        {source.content_snippet && (
                            <p className="text-xs text-gray-400 line-clamp-3 mb-2">
                                {source.content_snippet}
                            </p>
                        )}
                        {source.url && (
                            <a
                                href={source.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-xs text-blue-400 hover:text-blue-300 hover:underline"
                                onClick={(e) => e.stopPropagation()}
                            >
                                Read Original Article →
                            </a>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
}
