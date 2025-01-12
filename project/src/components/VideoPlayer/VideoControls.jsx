import PropTypes from 'prop-types';
import { FiDownload } from 'react-icons/fi';

const VideoControls = ({ onDownload }) => {
  return (
    <button
      onClick={onDownload}
      className="absolute bottom-4 right-4 bg-blue-600 text-white px-4 py-2 rounded-lg 
        flex items-center gap-2 hover:bg-blue-700 transition-colors shadow-lg"
    >
      <FiDownload />
      Download
    </button>
  );
};

VideoControls.propTypes = {
  onDownload: PropTypes.func.isRequired
};

export default VideoControls;