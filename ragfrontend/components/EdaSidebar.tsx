export default function EdaSidebar() {
    return (
        <div className="h-full bg-[#111] border-l border-[#333] flex flex-col">
            <div className="p-4 border-b border-[#333]">
                <h2 className="text-xl font-bold text-white">
                    EDA Report
                </h2>
            </div>
            <div className="flex-1 w-full bg-white relative">
                <iframe
                    src="http://localhost:8000/report/report.html"
                    className="absolute inset-0 w-full h-full border-none"
                    title="EDA Report"
                    sandbox="allow-scripts allow-same-origin"
                />
            </div>
        </div>
    );
}
