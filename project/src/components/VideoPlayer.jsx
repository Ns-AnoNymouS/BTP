import React, { useState, useRef, useEffect } from 'react';
import { 
  FiPlay, 
  FiPause, 
  FiSkipBack, 
  FiSkipForward,
  FiVolume2,
  FiVolumeX
} from 'react-icons/fi';

const VideoPlayer = ({ videoUrl }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isMuted, setIsMuted] = useState(false);
  const videoRef = useRef(null);
  const progressRef = useRef(null);

  useEffect(() => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
    }
  }, [videoUrl]);

  const togglePlay = () => {
    if (videoRef.current.paused) {
      videoRef.current.play();
      setIsPlaying(true);
    } else {
      videoRef.current.pause();
      setIsPlaying(false);
    }
  };

  const handleTimeUpdate = () => {
    setCurrentTime(videoRef.current.currentTime);
  };

  const handleProgressClick = (e) => {
    const rect = progressRef.current.getBoundingClientRect();
    const pos = (e.clientX - rect.left) / rect.width;
    videoRef.current.currentTime = pos * duration;
  };

  const skipTime = (seconds) => {
    videoRef.current.currentTime += seconds;
  };

  const toggleMute = () => {
    videoRef.current.muted = !videoRef.current.muted;
    setIsMuted(!isMuted);
  };

  const formatTime = (time) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  return (
    <div className="w-full max-w-4xl mx-auto bg-black rounded-lg overflow-hidden shadow-xl">
      <video
        ref={videoRef}
        className="w-full max-h-[450px] object-contain"
        src={videoUrl}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={() => setDuration(videoRef.current.duration)}
      />
      
      <div className="bg-gray-900 p-4">
        <div
          ref={progressRef}
          className="w-full h-2 bg-gray-700 rounded-full mb-4 cursor-pointer"
          onClick={handleProgressClick}
        >
          <div
            className="h-full bg-blue-500 rounded-full"
            style={{ width: `${(currentTime / duration) * 100}%` }}
          />
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <button
              className="text-white hover:text-blue-500 transition"
              onClick={() => skipTime(-10)}
            >
              <FiSkipBack size={24} />
            </button>
            
            <button
              className="text-white hover:text-blue-500 transition"
              onClick={togglePlay}
            >
              {isPlaying ? <FiPause size={32} /> : <FiPlay size={32} />}
            </button>
            
            <button
              className="text-white hover:text-blue-500 transition"
              onClick={() => skipTime(10)}
            >
              <FiSkipForward size={24} />
            </button>

            <button
              className="text-white hover:text-blue-500 transition"
              onClick={toggleMute}
            >
              {isMuted ? <FiVolumeX size={24} /> : <FiVolume2 size={24} />}
            </button>
          </div>

          <div className="text-white text-sm">
            {formatTime(currentTime)} / {formatTime(duration)}
          </div>
        </div>
      </div>
    </div>
  );
};

export default VideoPlayer;