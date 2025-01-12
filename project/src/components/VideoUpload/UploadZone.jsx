import PropTypes from 'prop-types';
import { FiUploadCloud, FiFilm } from 'react-icons/fi';

const UploadZone = ({ onFileSelect, disabled }) => {
  return (
    <label
      htmlFor="video-upload"
      className={`w-full flex flex-col items-center justify-center border-2 border-dashed 
        rounded-xl p-8 transition-all duration-300
        ${
          disabled 
            ? 'border-gray-300 bg-gray-50 cursor-not-allowed' 
            : 'border-blue-300 hover:border-blue-500 hover:bg-blue-50 cursor-pointer'
        }`}
    >
      <div className="relative">
        <div className="absolute -inset-1 bg-gradient-to-r from-blue-600 to-blue-400 rounded-lg blur opacity-25 group-hover:opacity-100 transition duration-1000 group-hover:duration-200"></div>
        <div className="relative bg-white rounded-lg p-6">
          {disabled ? (
            <FiFilm className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          ) : (
            <FiUploadCloud className="w-16 h-16 text-blue-500 mx-auto mb-4" />
          )}
          <div className="text-center">
            <p className="text-lg font-semibold mb-2">
              {disabled ? 'Upload in Progress' : 'Upload Your Sports Video'}
            </p>
            <p className="text-sm text-gray-500">
              {disabled 
                ? 'Please wait while we process your video'
                : 'Drag and drop your video file here or click to browse'
              }
            </p>
            <p className="text-xs text-gray-400 mt-2">
              Supports: MP4, MOV, AVI (max 500MB)
            </p>
          </div>
        </div>
      </div>
      <input
        id="video-upload"
        type="file"
        accept="video/*"
        className="hidden"
        onChange={onFileSelect}
        disabled={disabled}
      />
    </label>
  );
};

UploadZone.propTypes = {
  onFileSelect: PropTypes.func.isRequired,
  disabled: PropTypes.bool
};

export default UploadZone;