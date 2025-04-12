import React from 'react';
import { Song } from '../types';

interface SongRecommendationsProps {
    songs: Song[];
    isLoading: boolean;
    emotion: string | null;
}

const SongRecommendations: React.FC<SongRecommendationsProps> = ({
                                                                     songs,
                                                                     isLoading,
                                                                     emotion
                                                                 }) => {
    if (isLoading) {
        return (
            <div className="recommendations-container">
                <h2>Music Recommendations</h2>
                <div className="loading-recommendations">
                    <div className="spinner"></div>
                    <p>Finding the perfect songs for your mood...</p>
                </div>
            </div>
        );
    }

    if (!emotion) {
        return (
            <div className="recommendations-container">
                <h2>Music Recommendations</h2>
                <p className="no-recommendations">
                    We'll recommend music based on your detected emotion. Please wait for emotion detection.
                </p>
            </div>
        );
    }

    if (songs.length === 0) {
        return (
            <div className="recommendations-container">
                <h2>Music Recommendations</h2>
                <p className="no-recommendations">
                    No songs found for your current mood and filters. Try adjusting your preferences.
                </p>
            </div>
        );
    }

    return (
        <div className="recommendations-container">
            <h2>Music Recommendations</h2>
            <div className="song-grid">
                {songs.map((song) => (
                    <div key={song.id} className="song-card">
                        <div className="album-cover">
                            {song.album_image ? (
                                <img src={song.album_image} alt={`${song.album} cover`} />
                            ) : (
                                <div className="no-image">No Image</div>
                            )}
                        </div>
                        <div className="song-info">
                            <h3 className="song-title">{song.name}</h3>
                            <p className="song-artist">{song.artist}</p>
                            <p className="song-album">{song.album}</p>
                        </div>
                        <div className="song-actions">
                            {song.preview_url && (
                                <a
                                    href={song.preview_url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="preview-button"
                                >
                                    Preview
                                </a>
                            )}
                            <a
                                href={song.external_url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="spotify-button"
                            >
                                Open in Spotify
                            </a>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default SongRecommendations;