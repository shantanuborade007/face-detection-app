import { useState, useEffect, useRef } from 'react';
import './App.css';
import WebcamCapture from './components/WebcamCapture';
import EmotionDisplay from './components/EmotionDisplay';
import SongRecommendations from './components/SongRecommendations';
import FilterOptions from './components/FilterOptions';
import NotificationModal from './components/NotificationModal';
import { Song, FilterOptionsType } from './types';

const API_URL = 'http://localhost:8000';

function App() {
  const [currentEmotion, setCurrentEmotion] = useState<string | null>(null);
  const [previousEmotion, setPreviousEmotion] = useState<string | null>(null);
  const [showNotification, setShowNotification] = useState(false);
  const [isCapturing, setIsCapturing] = useState(true);
  const [songs, setSongs] = useState<Song[]>([]);
  const [filters, setFilters] = useState<FilterOptionsType>({
    language: '',
    era: '',
    limit: 10
  });
  const [isLoading, setIsLoading] = useState(false);
  const captureIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Start capture interval when component mounts
  useEffect(() => {
    if (isCapturing) {
      captureIntervalRef.current = setInterval(triggerCapture, 90000);
      console.log('Auto-capture started (interval 1 min 30s)');
    }
    
    return () => {
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current);
        console.log('Auto-capture stopped');
      }
    };
  }, [isCapturing]);

  // Handle emotion change notification
  useEffect(() => {
    if (previousEmotion && currentEmotion && previousEmotion !== currentEmotion) {
      setShowNotification(true);
      console.log(`Emotion changed: ${previousEmotion} → ${currentEmotion}`);
    }
  }, [currentEmotion, previousEmotion]);

  const triggerCapture = () => {
    console.log('Triggering image capture...');
    document.dispatchEvent(new Event('capture-image'));
  };

  const handleImageCapture = async (imageData: string) => {
    try {
      setIsLoading(true);
      console.log('Sending image to backend for emotion detection...');
      // Call API to detect emotion
      const response = await fetch(`${API_URL}/detect-emotion`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to detect emotion');
      }
      
      const data = await response.json();
      console.log('Emotion detected:', data.emotion);

      // Update emotions
      setPreviousEmotion(currentEmotion);
      setCurrentEmotion(data.emotion);
      
      setIsLoading(false);
    } catch (error) {
      console.error('Error processing image:', error);
      setIsLoading(false);
    }
  };

  const handleNotificationResponse = (accepted: boolean) => {
    setShowNotification(false);
    console.log(`User ${accepted ? 'accepted' : 'declined'} emotion change notification`);

    if (accepted && currentEmotion) {
      fetchRecommendations(currentEmotion);
    }
  };

  const fetchRecommendations = async (emotion: string) => {
    try {
      setIsLoading(true);
      console.log(`Fetching recommendations for emotion: ${emotion}, filters:`, filters);

      const response = await fetch(`${API_URL}/recommend-songs`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          emotion: emotion,
          filters: filters
        }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to get recommendations');
      }
      
      const data = await response.json();
      setSongs(data.songs);
      setIsLoading(false);
    } catch (error) {
      console.error('Error fetching recommendations:', error);
      setIsLoading(false);
    }
  };

  const handleFilterChange = (newFilters: FilterOptionsType) => {
    setFilters(newFilters);
    console.log('Filter options updated:', newFilters);

    // If we have an emotion, fetch new recommendations with updated filters
    if (currentEmotion) {
      fetchRecommendations(currentEmotion);
    }
  };

  const handleManualCapture = () => {
    triggerCapture();
    
    // Reset the interval timer
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
    }
    
    if (isCapturing) {
      captureIntervalRef.current = setInterval(triggerCapture, 35000);
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>Emotion-Based Music Recommender</h1>
        <p>We detect your mood and recommend the perfect music for you</p>
      </header>
      
      <main>
        <div className="webcam-section">
          <WebcamCapture onCapture={handleImageCapture} />
          <button 
            className="capture-button" 
            onClick={handleManualCapture}
            disabled={isLoading}
          >
            Detect Emotion Now
          </button>
          <div className="capture-status">
            Auto-capture: {isCapturing ? 'On' : 'Off'}
            <label className="toggle">
              <input
                type="checkbox"
                checked={isCapturing}
                onChange={() => setIsCapturing(!isCapturing)}
              />
              <span className="slider"></span>
            </label>
          </div>
        </div>
        
        <div className="content-section">
          <EmotionDisplay emotion={currentEmotion} isLoading={isLoading} />
          
          <FilterOptions 
            filters={filters} 
            onFilterChange={handleFilterChange} 
          />
          
          <SongRecommendations 
            songs={songs} 
            isLoading={isLoading} 
            emotion={currentEmotion} 
          />
        </div>
      </main>
      
      {showNotification && (
        <NotificationModal
          emotion={currentEmotion || ''}
          onResponse={handleNotificationResponse}
        />
      )}
      
      <footer>
        <p>© 2025 Emotion-Based Music Recommender</p>
      </footer>
    </div>
  );
}

export default App;