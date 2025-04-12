import React from 'react';

interface EmotionDisplayProps {
  emotion: string | null;
  isLoading: boolean;
}

const EmotionDisplay: React.FC<EmotionDisplayProps> = ({ emotion, isLoading }) => {
  // Map emotions to emojis
  const emotionEmojis: { [key: string]: string } = {
    'Angry': '😠',
    'Disgust': '🤢',
    'Fear': '😨',
    'Happy': '😊',
    'Sad': '😢',
    'Surprise': '😲',
    'Neutral': '😐'
  };
  
  return (
    <div className="emotion-display">
      <h2>Your Current Emotion</h2>
      
      {isLoading ? (
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Analyzing your expression...</p>
        </div>
      ) : emotion ? (
        <div className="emotion-result">
          <div className="emotion-emoji">{emotionEmojis[emotion] || '🤔'}</div>
          <h3>{emotion}</h3>
        </div>
      ) : (
        <p className="no-emotion">We haven't detected your emotion yet. Please wait for the next capture or press "Detect Emotion Now".</p>
      )}
    </div>
  );
};

export default EmotionDisplay;