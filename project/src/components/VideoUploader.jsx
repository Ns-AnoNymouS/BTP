import React, { useState, useRef } from "react";
import { FiUpload } from "react-icons/fi";
import axios from "axios";

// Create an HTTP/HTTPS agent with keep-alive and a timeout of 1 hour
const agentOptions = {
  keepAlive: true,
  keepAliveMsecs: 3600000, // 1 hour in milliseconds
};

// Create an Axios instance
const axiosInstance = axios.create({
  timeout: 3600000, // Optional: Request timeout (1 hour)
});

const VideoUploader = ({ onVideoProcessed }) => {
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processing, setProcessing] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleUpload = async (file) => {
    const formData = new FormData();
    formData.append("file", file);

    try {
      setProcessing(true);
      const response = await axiosInstance.post(
        `${import.meta.env.VITE_APP_BACKEND_URL}/stability`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
          responseType: "blob",
          onUploadProgress: (progressEvent) => {
            const progress = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            setUploadProgress(progress);
          },
        }
      );

      console.log(response.headers["content-type"]);

      const videoUrl = URL.createObjectURL(response.data);
      window.open(videoUrl, "_blank");
      console.log("Video uploaded:", videoUrl);      
      onVideoProcessed(videoUrl);
    } catch (error) {
      console.error("Error uploading video:", error);
    } finally {
      setProcessing(false);
      setUploadProgress(0);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleUpload(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div
        className={`border-4 border-dashed rounded-lg p-8 text-center transition-all ${
          dragActive ? "border-blue-500 bg-blue-50" : "border-gray-300"
        }`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          type="file"
          ref={fileInputRef}
          className="hidden"
          accept="video/*"
          onChange={(e) => {
            if (e.target.files?.[0]) {
              handleUpload(e.target.files[0]);
            }
          }}
        />
        <FiUpload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
        <p className="text-lg font-semibold mb-2">
          Drag and drop your video here or click to browse
        </p>
        <p className="text-sm text-gray-500">
          Supported formats: MP4, MOV, AVI
        </p>
      </div>

      {(uploadProgress > 0 || processing) && (
        <div className="mt-6">
          <div className="flex justify-between mb-2">
            <span className="text-sm font-medium">
              {processing ? "Processing video..." : "Uploading video..."}
            </span>
            <span className="text-sm font-medium">{uploadProgress}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div
              className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
              style={{ width: `${uploadProgress}%` }}
            ></div>
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoUploader;
