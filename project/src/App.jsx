import React, { useState, useEffect } from "react";
import VideoUploader from "./components/VideoUploader";
import VideoPlayer from "./components/VideoPlayer";
import "./App.css";

function App() {
  const [processedVideoUrl, setProcessedVideoUrl] = useState(null);
  useEffect(() => {
    return () => {
      URL.revokeObjectURL(processedVideoUrl); // Clean up the object URL when component unmounts
    };
  }, [processedVideoUrl]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 to-black text-white">
      <header className="py-6 px-4 mb-8">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-4xl font-bold text-center">
            Sports Video Analysis
          </h1>
          <p className="text-center text-gray-300 mt-2">
            Upload your sports video for professional analysis
          </p>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 pb-12">
        <div className="grid gap-8">
          {!processedVideoUrl && (
            <section className="bg-white/10 backdrop-blur-lg rounded-xl p-8">
              <h2 className="text-2xl font-semibold mb-6 text-center">
                Upload Your Video
              </h2>
              <VideoUploader onVideoProcessed={setProcessedVideoUrl} />
            </section>
          )}

          {processedVideoUrl && (
            <section className="bg-white/10 backdrop-blur-lg rounded-xl p-8">
              <h2 className="text-2xl font-semibold mb-6 text-center">
                Analysis Results
              </h2>
              <VideoPlayer videoUrl={processedVideoUrl} />
            </section>
          )}
        </div>
      </main>

      <footer className="py-6 text-center text-gray-400">
        <p>© 2024 Sports Video Analysis. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default App;
