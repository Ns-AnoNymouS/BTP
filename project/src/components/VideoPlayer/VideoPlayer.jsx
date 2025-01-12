import PropTypes from 'prop-types';
import VideoControls from './VideoControls';
import { downloadVideo } from '../../utils/downloadUtils';
import { FiMaximize2, FiDownload, FiShare2 } from 'react-icons/fi';

const VideoPlayer = ({ videoUrl }) => {
  const handleDownload = () => {
    downloadVideo(videoUrl, 'sports-analysis.mp4');
  };

  return (
    <div className="card mt-8">
      <h2 className="section-title">Video Analysis</h2>
      <div className="relative rounded-lg overflow-hidden bg-gray-900">
        <video
          className="w-full aspect-video"
          controls
          src={videoUrl}
          poster="/placeholder-poster.jpg"
        >
          Your browser does not support the video tag.
        </video>
        
        <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/70 to-transparent">
          <div className="flex justify-end space-x-2">
            <button
              className="p-2 rounded-full bg-white/20 hover:bg-white/30 transition-colors text-white"
              title="Fullscreen"
            >
              <FiMaximize2 className="w-5 h-5" />
            </button>
            <button
              className="p-2 rounded-full bg-white/20 hover:bg-white/30 transition-colors text-white"
              title="Share"
            >
              <FiShare2 className="w-5 h-5" />
            </button>
            <button
              onClick={handleDownload}
              className="p-2 rounded-full bg-blue-500 hover:bg-blue-600 transition-colors text-white"
              title="Download"
            >
              <FiDownload className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>
      
      <div className="mt-6 grid grid-cols-3 gap-4">
        <div className="p-4 bg-gray-50 rounded-lg">
          <h3 className="font-semibold text-gray-700">Duration</h3>
          <p className="text-gray-600">03:45</p>
        </div>
        <div className="p-4 bg-gray-50 rounded-lg">
          <h3 className="font-semibold text-gray-700">Quality</h3>
          <p className="text-gray-600">1080p HD</p>
        </div>
        <div className="p-4 bg-gray-50 rounded-lg">
          <h3 className="font-semibold text-gray-700">Size</h3>
          <p className="text-gray-600">245 MB</p>
        </div>
      </div>
    </div>
  );
};

VideoPlayer.propTypes = {
  videoUrl: PropTypes.string.isRequired
};

export default VideoPlayer;