import React from 'react';

interface NotificationModalProps {
    emotion: string;
    onResponse: (accepted: boolean) => void;
}

const NotificationModal: React.FC<NotificationModalProps> = ({ emotion, onResponse }) => {
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
        <div className="modal-overlay">
            <div className="notification-modal">
                <div className="modal-header">
                    <div className="emotion-emoji large">{emotionEmojis[emotion] || '🤔'}</div>
                    <h2>New Emotion Detected!</h2>
                </div>
                <div className="modal-content">
                    <p>We've detected that your emotion has changed to <strong>{emotion}</strong>.</p>
                    <p>Would you like us to update your music recommendations?</p>
                </div>
                <div className="modal-actions">
                    <button
                        className="button secondary"
                        onClick={() => onResponse(false)}
                    >
                        No, Keep Current Songs
                    </button>
                    <button
                        className="button primary"
                        onClick={() => onResponse(true)}
                    >
                        Yes, Update Recommendations
                    </button>
                </div>
            </div>
        </div>
    );
};

export default NotificationModal;