import { useState } from 'react';
import PropTypes from 'prop-types';
import UploadZone from './UploadZone';
import ProgressBar from '../ProgressBar/ProgressBar';
import { uploadVideo } from '../../services/videoService';
import { useToast } from '../../hooks/useToast';

const VideoUpload = ({ onVideoProcessed }) => {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const { showToast } = useToast();

  const handleUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Check file size (500MB limit)
    if (file.size > 500 * 1024 * 1024) {
      showToast('File size exceeds 500MB limit', 'error');
      return;
    }

    setUploading(true);
    setProgress(0);

    try {
      const videoUrl = await uploadVideo(file, (progress) => {
        setProgress(progress);
      });
      
      onVideoProcessed(videoUrl);
      showToast('Video uploaded successfully!', 'success');
    } catch (error) {
      console.error('Error uploading video:', error);
      showToast('Failed to upload video. Please try again.', 'error');
    } finally {
      setUploading(false);
      setProgress(0);
    }
  };

  return (
    <div className="card">
      <h2 className="section-title">Upload Your Game Footage</h2>
      <div className="space-y-6">
        <UploadZone onFileSelect={handleUpload} disabled={uploading} />
        
        {uploading && (
          <div className="mt-6">
            <ProgressBar progress={progress} />
          </div>
        )}
      </div>
    </div>
  );
};

VideoUpload.propTypes = {
  onVideoProcessed: PropTypes.func.isRequired
};

export default VideoUpload;