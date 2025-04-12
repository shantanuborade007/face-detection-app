import React, { useRef, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';

interface WebcamCaptureProps {
  onCapture: (imageData: string) => void;
}

const WebcamCapture: React.FC<WebcamCaptureProps> = ({ onCapture }) => {
  const webcamRef = useRef<Webcam>(null);
  
  const captureImage = useCallback(() => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        // Extract the base64 data without the prefix
        const base64Data = imageSrc.split(',')[1];
        onCapture(base64Data);
      }
    }
  }, [onCapture]);
  
  useEffect(() => {
    // Listen for the custom capture event
    const handleCaptureEvent = () => {
      captureImage();
    };
    
    document.addEventListener('capture-image', handleCaptureEvent);
    
    return () => {
      document.removeEventListener('capture-image', handleCaptureEvent);
    };
  }, [captureImage]);
  
  const videoConstraints = {
    width: 400,
    height: 400,
    facingMode: "user"
  };
  
  return (
    <div className="webcam-container">
      <Webcam
        audio={false}
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        videoConstraints={videoConstraints}
        mirrored={true}
        className="webcam"
      />
    </div>
  );
};

export default WebcamCapture;