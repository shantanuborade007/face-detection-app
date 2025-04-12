# main.py
import base64
import io
import os
from typing import List, Optional
import logging

import cv2
import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from dotenv import load_dotenv

# Load environment variables

logger = logging.getLogger(__name__)
# Initialize FastAPI app
app = FastAPI(title="Emotion-Based Music Recommender API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Spotify API credentials
# SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
# SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

SPOTIFY_CLIENT_ID="7e560ebbdabf451785442dafa1587729"
SPOTIFY_CLIENT_SECRET="b5f7e89eaa05451890565a8a852a2701"

# Emotion to genre mapping
EMOTION_TO_GENRE = {
    "Angry": "metal,hard-rock,punk",
    "Disgust": "blues,jazz,classical",
    "Fear": "ambient,soundtrack,post-rock",
    "Happy": "pop,dance,edm",
    "Sad": "sad,singer-songwriter,ambient",
    "Surprise": "electronic,indie,alternative",
    "Neutral": "chill,acoustic,indie-pop"
}

# Models
class EmotionDetectionModel(nn.Module):
    def __init__(self):
        super(EmotionDetectionModel, self).__init__()
        # Load MobileNetV2
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        num_ftrs_mobilenet = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs_mobilenet, 7)
        )

        # Load ResNet34
        self.resnet = models.resnet34(pretrained=True)
        num_ftrs_resnet = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs_resnet, 7)

        # Final classifier (combines outputs from both models)
        self.classifier = nn.Linear(14, 7)

    def forward(self, x):
        mobilenet_output = self.mobilenet(x)
        resnet_output = self.resnet(x)
        combined = torch.cat((mobilenet_output, resnet_output), dim=1)
        return self.classifier(combined)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionDetectionModel().to(device)

# In a real implementation, you would load pre-trained weights
# model.load_state_dict(torch.load("emotion_detection_model.pth"))
model.eval()

# Image processing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Detect faces using OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_emotion(image_data):
    # Decode base64 to image
    img_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    # Convert to OpenCV format for face detection
    open_cv_image = np.array(img)
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    # Detect faces
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return "Neutral"  # Default if no face detected

    # Process the largest face
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    face_img = open_cv_image[y:y+h, x:x+w]
    face_pil = Image.fromarray(face_img[:, :, ::-1])

    # Transform and prepare for model
    face_tensor = transform(face_pil).unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        outputs = model(face_tensor)
        _, predicted = torch.max(outputs, 1)

    # Map index to emotion
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    return emotions[predicted.item()]

# Get Spotify access token
def get_spotify_token():
    logger.info("Spotify Credentials - Client ID: %s, Client Secret: %s", SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)

    auth_url = "https://accounts.spotify.com/api/token"
    auth_response = requests.post(auth_url,
                                  data={
                                      'grant_type': 'client_credentials',
                                      'client_id': SPOTIFY_CLIENT_ID,
                                      'client_secret': SPOTIFY_CLIENT_SECRET
                                  },
                                  headers={'Content-Type': 'application/x-www-form-urlencoded'}
                                  )

    if auth_response.status_code != 200:
        logger.error("Spotify Authentication Failed: %s", auth_response.text)
        raise HTTPException(status_code=500, detail=auth_response.text)

    auth_data = auth_response.json()
    return auth_data['access_token']

# Recommend songs based on emotion
def recommend_songs(emotion, language=None, era=None, limit=10):
     genre = EMOTION_TO_GENRE.get(emotion, "pop")
     token = get_spotify_token()

     headers = {
        'Authorization': f'Bearer {token}'
     }

    # Construct query string manually
     query = f'genre:"{genre}"'
    
     if language:
         query += f' {language}'
    
     if era:
        era_ranges = {
            "1960s": "1960-1969",
            "1970s": "1970-1979",
            "1980s": "1980-1989",
            "1990s": "1990-1999",
            "2000s": "2000-2009",
            "2010s": "2010-2019",
            "2020s": "2020-2029"
        }
        year_range = era_ranges.get(era)
        if year_range:
            query += f' year:{year_range}'

    # Set parameters
     params = {
        'q': query,
        'type': 'track',
        'market': 'IN',
        'limit': limit
    }

    # Make request to Spotify Search API
     url = "https://api.spotify.com/v1/search"
     response = requests.get(url, headers=headers, params=params)

     if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch songs from Spotify")

     data = response.json()
    #  return data
     tracks = data.get("tracks", {}).get("items", [])

     # Format response
     songs = []
     for track in tracks:
        songs.append({
            'id': track['id'],
            'name': track['name'],
            'artist': track['artists'][0]['name'],
            'album': track['album']['name'],
            'preview_url': track['preview_url'],
            'external_url': track['external_urls']['spotify'],
            'album_image': track['album']['images'][0]['url'] if track['album']['images'] else None
        })
 
     return songs

# Request/Response models
class ImageRequest(BaseModel):
    image: str  # Base64 encoded image

class FilterOptions(BaseModel):
    language: Optional[str] = None
    era: Optional[str] = None
    limit: Optional[int] = 10

class EmotionResponse(BaseModel):
    emotion: str
    genre: str

class RecommendationRequest(BaseModel):
    emotion: str
    filters: Optional[FilterOptions] = None

class Song(BaseModel):
    id: str
    name: str
    artist: str
    album: str
    preview_url: Optional[str]
    external_url: str
    album_image: Optional[str]

class RecommendationResponse(BaseModel):
    emotion: str
    genre: str
    songs: List[Song]

# API endpoints
@app.post("/detect-emotion", response_model=EmotionResponse)
async def detect_emotion_endpoint(request: ImageRequest):
    try:
        emotion = detect_emotion(request.image)
        genre = EMOTION_TO_GENRE.get(emotion, "pop")
        return {"emotion": emotion, "genre": genre}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/recommend-songs", response_model=RecommendationResponse)
async def recommend_songs_endpoint(request: RecommendationRequest):
    try:
        emotion = request.emotion
        filters = request.filters or FilterOptions()

        genre = EMOTION_TO_GENRE.get(emotion, "pop")
        songs = recommend_songs(
            emotion,
            language=filters.language,
            era=filters.era,
            limit=filters.limit
        )
        # return songs
        return {
            "emotion": emotion,
            "genre": genre,
            "songs": songs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")

@app.post("/process-and-recommend", response_model=RecommendationResponse)
async def process_and_recommend(request: ImageRequest, filters: FilterOptions = Body(...)):
    try:
        # Detect emotion
        emotion = detect_emotion(request.image)
        genre = EMOTION_TO_GENRE.get(emotion, "pop")

        # Get recommendations
        songs = recommend_songs(
            emotion,
            language=filters.language,
            era=filters.era,
            limit=filters.limit
        )

        return {
            "emotion": emotion,
            "genre": genre,
            "songs": songs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)