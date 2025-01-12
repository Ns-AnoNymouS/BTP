import { FiVideo, FiActivity } from 'react-icons/fi';

const Header = () => {
  return (
    <header className="bg-gradient-to-r from-blue-700 via-blue-600 to-blue-800 text-white py-8 mb-8">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-center space-x-4">
          <FiActivity className="w-8 h-8 text-blue-300" />
          <h1 className="text-4xl font-bold text-center">
            Sports Analysis Pro
          </h1>
          <FiVideo className="w-8 h-8 text-blue-300" />
        </div>
        <p className="text-center mt-4 text-blue-100 max-w-2xl mx-auto">
          Advanced video analysis platform for athletes and coaches. 
          Upload your sports footage and get professional insights.
        </p>
      </div>
    </header>
  );
};

export default Header;