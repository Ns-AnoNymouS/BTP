import PropTypes from 'prop-types';
import { FiCheckCircle, FiAlertCircle, FiInfo } from 'react-icons/fi';

const Toast = ({ message, type }) => {
  const styles = {
    success: {
      bg: 'bg-green-500',
      icon: FiCheckCircle
    },
    error: {
      bg: 'bg-red-500',
      icon: FiAlertCircle
    },
    info: {
      bg: 'bg-blue-500',
      icon: FiInfo
    }
  };

  const { bg, icon: Icon } = styles[type];

  return (
    <div className={`fixed bottom-4 right-4 ${bg} text-white px-6 py-3 rounded-lg shadow-lg
                    flex items-center space-x-2 animate-slide-up`}>
      <Icon className="w-5 h-5" />
      <span>{message}</span>
    </div>
  );
};

Toast.propTypes = {
  message: PropTypes.string.isRequired,
  type: PropTypes.oneOf(['success', 'error', 'info']).isRequired
};

export default Toast;