import axios from 'axios';

export const uploadVideo = async (file, onProgress) => {
  const formData = new FormData();
  formData.append('video', file);

  try {
    const response = await axios.post('/api/upload', formData, {
      onUploadProgress: (progressEvent) => {
        const progress = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        );
        onProgress(progress);
      },
    });

    return response.data.videoUrl;
  } catch (error) {
    console.error('Upload error:', error);
    throw new Error('Failed to upload video');
  }
};