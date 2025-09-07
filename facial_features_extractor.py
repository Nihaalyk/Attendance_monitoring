"""
Advanced Facial Feature Extraction System for Attendance Monitoring
Combines multiple state-of-the-art face detection and recognition models
for robust performance in classroom environments.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
import mediapipe as mp
from facenet_pytorch import MTCNN, InceptionResnetV1
import face_recognition
from insightface.app import FaceAnalysis
from PIL import Image
import logging
from dataclasses import dataclass
from pathlib import Path
import pickle
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FaceDetection:
    """Container for face detection results"""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    landmarks: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    face_id: Optional[str] = None

@dataclass
class ExtractionConfig:
    """Configuration for facial feature extraction"""
    detection_confidence: float = 0.7
    recognition_threshold: float = 0.6
    max_faces_per_image: int = 50
    face_size: Tuple[int, int] = (160, 160)
    enable_alignment: bool = True
    enable_quality_filter: bool = True
    min_face_size: int = 40

class AdvancedFacialExtractor:
    """
    Advanced facial feature extraction system using multiple models:
    - MTCNN for face detection
    - FaceNet for embeddings
    - InsightFace for additional features
    - MediaPipe for landmarks
    """
    
    def __init__(self, config: ExtractionConfig = None):
        self.config = config or ExtractionConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self._init_models()
        
    def _init_models(self):
        """Initialize all face detection and recognition models"""
        try:
            # MTCNN for face detection (Enhanced settings)
            self.mtcnn = MTCNN(
                image_size=self.config.face_size[0],
                margin=20,  # Increased margin for better face crops
                min_face_size=30,  # Smaller minimum to detect distant faces
                thresholds=[0.5, 0.6, 0.6],  # More sensitive thresholds
                factor=0.709,
                post_process=True,  # Enable post-processing for better quality
                device=self.device,
                keep_all=True  # Keep all detected faces
            )
            
            # FaceNet for embeddings
            self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            
            # InsightFace for additional analysis (Enhanced settings)
            self.insight_app = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.insight_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.4)  # Lower detection threshold
            
            # MediaPipe for facial landmarks
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=self.config.max_faces_per_image,
                refine_landmarks=True,
                min_detection_confidence=self.config.detection_confidence
            )
            
            logger.info("All models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces using MTCNN with confidence filtering
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of FaceDetection objects
        """
        try:
            # Convert BGR to RGB for MTCNN
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Detect faces with MTCNN
            boxes, probs, landmarks = self.mtcnn.detect(pil_image, landmarks=True)
            
            detections = []
            if boxes is not None:
                for i, (box, prob, landmark) in enumerate(zip(boxes, probs, landmarks)):
                    if prob > self.config.detection_confidence:
                        # Convert to integer coordinates
                        x, y, x2, y2 = map(int, box)
                        w, h = x2 - x, y2 - y
                        
                        detection = FaceDetection(
                            bbox=(x, y, w, h),
                            confidence=float(prob),
                            landmarks=landmark
                        )
                        detections.append(detection)
            
            logger.info(f"Detected {len(detections)} faces")
            return detections
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return []
    
    def extract_face_embeddings(self, image: np.ndarray, detections: List[FaceDetection]) -> List[FaceDetection]:
        """
        Extract face embeddings using FaceNet
        
        Args:
            image: Input image
            detections: List of face detections
            
        Returns:
            Updated detections with embeddings
        """
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            for detection in detections:
                x, y, w, h = detection.bbox
                
                # Extract face region with padding
                padding = 20
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(image.shape[1], x + w + padding)
                y2 = min(image.shape[0], y + h + padding)
                
                face_img = rgb_image[y1:y2, x1:x2]
                
                if face_img.size > 0:
                    # Resize and preprocess for FaceNet
                    face_pil = Image.fromarray(face_img)
                    face_tensor = self.mtcnn(face_pil)
                    
                    if face_tensor is not None:
                        face_tensor = face_tensor.unsqueeze(0).to(self.device)
                        
                        with torch.no_grad():
                            embedding = self.facenet(face_tensor)
                            embedding = F.normalize(embedding, p=2, dim=1)
                            detection.embedding = embedding.cpu().numpy().flatten()
            
            return detections
            
        except Exception as e:
            logger.error(f"Error extracting embeddings: {e}")
            return detections
    
    def process_image(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Complete facial feature extraction pipeline
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of processed face detections with embeddings
        """
        start_time = time.time()
        
        # Step 1: Detect faces
        detections = self.detect_faces(image)
        
        if not detections:
            logger.info("No faces detected")
            return []
        
        # Step 2: Extract embeddings
        detections = self.extract_face_embeddings(image, detections)
        
        processing_time = time.time() - start_time
        logger.info(f"Processed {len(detections)} faces in {processing_time:.2f}s")
        
        return detections
    
    def get_face_embedding(self, image: np.ndarray, face_info: dict) -> Optional[np.ndarray]:
        """
        Compatibility method for existing code
        Extract embedding for a single face detection
        """
        try:
            # Convert face_info to FaceDetection if needed
            if isinstance(face_info, dict):
                if 'bbox' in face_info:
                    bbox = face_info['bbox']
                else:
                    # Assume it's in x,y,w,h format
                    bbox = (face_info.get('x', 0), face_info.get('y', 0), 
                           face_info.get('w', 100), face_info.get('h', 100))
                
                detection = FaceDetection(
                    bbox=bbox,
                    confidence=face_info.get('confidence', 1.0)
                )
            else:
                detection = face_info
            
            # Extract embedding
            detections_with_embeddings = self.extract_face_embeddings(image, [detection])
            
            if detections_with_embeddings and detections_with_embeddings[0].embedding is not None:
                return detections_with_embeddings[0].embedding
            
            return None
            
        except Exception as e:
            logger.error(f"Error in get_face_embedding: {e}")
            return None
    
    def get_face_landmarks(self, image: np.ndarray, face_info: dict) -> Optional[np.ndarray]:
        """
        Compatibility method for existing code
        Get landmarks for a single face detection
        """
        try:
            # For compatibility, return the landmarks if available
            if isinstance(face_info, dict) and 'landmarks' in face_info:
                return face_info['landmarks']
            elif hasattr(face_info, 'landmarks'):
                return face_info.landmarks
            
            # If no landmarks available, return None
            return None
            
        except Exception as e:
            logger.error(f"Error in get_face_landmarks: {e}")
            return None

# Example usage and testing
if __name__ == "__main__":
    # Initialize the extractor
    config = ExtractionConfig(
        detection_confidence=0.8,
        recognition_threshold=0.6,
        max_faces_per_image=30
    )
    
    extractor = AdvancedFacialExtractor(config)
    
    # Test with a sample image (you would replace this with actual image loading)
    print("Advanced Facial Feature Extraction System initialized successfully!")
    print(f"Using device: {extractor.device}")
    print("Ready for processing classroom images...")
