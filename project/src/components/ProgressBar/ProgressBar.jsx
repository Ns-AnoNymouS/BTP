import PropTypes from 'prop-types';

const ProgressBar = ({ progress, showPercentage = true }) => {
  return (
    <div className="w-full">
      <div className="relative w-full h-4 bg-gray-100 rounded-full overflow-hidden">
        <div
          className="absolute top-0 left-0 h-full bg-gradient-to-r from-blue-500 to-blue-600 
                     transition-all duration-300 ease-out"
          style={{ width: `${progress}%` }}
        >
          <div className="absolute inset-0 bg-white/20 animate-pulse"></div>
        </div>
      </div>
      {showPercentage && (
        <div className="mt-2 flex justify-between text-sm">
          <span className="text-gray-600 font-medium">Progress</span>
          <span className="text-blue-600 font-semibold">{Math.round(progress)}%</span>
        </div>
      )}
    </div>
  );
};

ProgressBar.propTypes = {
  progress: PropTypes.number.isRequired,
  showPercentage: PropTypes.bool
};

export default ProgressBar;