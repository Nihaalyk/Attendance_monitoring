"""
Known Faces Database System for Attendance Monitoring
Processes the known_faces_optimized folder with left, right, and front view images
per student to create robust face embeddings for recognition.
"""

import cv2
import numpy as np
import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, asdict
from facial_features_extractor import AdvancedFacialExtractor, FaceDetection, ExtractionConfig
import sqlite3
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class StudentProfile:
    """Student profile with multi-angle face data"""
    student_id: str
    name: str
    front_embedding: Optional[np.ndarray] = None
    left_embedding: Optional[np.ndarray] = None
    right_embedding: Optional[np.ndarray] = None
    combined_embedding: Optional[np.ndarray] = None
    front_landmarks: Optional[np.ndarray] = None
    left_landmarks: Optional[np.ndarray] = None
    right_landmarks: Optional[np.ndarray] = None
    quality_scores: Optional[Dict[str, float]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class KnownFacesDatabase:
    """
    Database system for managing known student faces with multi-angle support
    Processes left, right, and front view images for each student
    """
    
    def __init__(self, known_faces_path: str = "known_faces_optimized", 
                 db_path: str = "faces_database.db",
                 embeddings_path: str = "face_embeddings.pkl"):
        
        self.known_faces_path = Path(known_faces_path)
        self.db_path = db_path
        self.embeddings_path = embeddings_path
        
        # Initialize facial extractor with optimized settings
        config = ExtractionConfig(
            detection_confidence=0.8,
            recognition_threshold=0.6,
            max_faces_per_image=5,  # Reduced since these are individual photos
            enable_quality_filter=True,
            min_face_size=50
        )
        
        self.extractor = AdvancedFacialExtractor(config)
        self.student_profiles: Dict[str, StudentProfile] = {}
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for student profiles"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                front_embedding BLOB,
                left_embedding BLOB,
                right_embedding BLOB,
                combined_embedding BLOB,
                front_landmarks BLOB,
                left_landmarks BLOB,
                right_landmarks BLOB,
                quality_scores TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                timestamp TEXT,
                confidence REAL,
                detection_method TEXT,
                image_hash TEXT,
                FOREIGN KEY (student_id) REFERENCES students (student_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("Database initialized successfully")
    
    def _determine_image_type(self, filename: str, index: int) -> str:
        """
        Determine if image is front, left, or right view
        Based on filename patterns or index position
        """
        filename_lower = filename.lower()
        
        # Try to determine from filename
        if 'front' in filename_lower or 'center' in filename_lower:
            return 'front'
        elif 'left' in filename_lower:
            return 'left'
        elif 'right' in filename_lower:
            return 'right'
        
        # If no clear indication, use index-based assumption
        # Assuming order: left, right, front (or similar pattern)
        if index == 0:
            return 'left'
        elif index == 1:
            return 'right'
        else:
            return 'front'
    
    def _calculate_image_quality(self, image: np.ndarray, detection: FaceDetection) -> float:
        """
        Calculate image quality score based on multiple factors
        """
        x, y, w, h = detection.bbox
        face_region = image[y:y+h, x:x+w]
        
        if face_region.size == 0:
            return 0.0
        
        # Convert to grayscale for quality metrics
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # 1. Sharpness (Laplacian variance)
        sharpness = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        sharpness_score = min(sharpness / 500.0, 1.0)  # Normalize
        
        # 2. Brightness distribution
        brightness = np.mean(gray_face)
        brightness_score = 1.0 - abs(brightness - 128) / 128.0
        
        # 3. Contrast (standard deviation)
        contrast = np.std(gray_face)
        contrast_score = min(contrast / 64.0, 1.0)  # Normalize
        
        # 4. Face size score
        face_area = w * h
        size_score = min(face_area / (100 * 100), 1.0)  # Normalize to 100x100 baseline
        
        # 5. Detection confidence
        conf_score = detection.confidence
        
        # Weighted combination
        quality_score = (
            0.3 * sharpness_score +
            0.2 * brightness_score +
            0.2 * contrast_score +
            0.1 * size_score +
            0.2 * conf_score
        )
        
        return quality_score
    
    def process_student_folder(self, student_name: str) -> Optional[StudentProfile]:
        """
        Process all images for a single student
        
        Args:
            student_name: Name of the student (folder name)
            
        Returns:
            StudentProfile with processed embeddings
        """
        student_folder = self.known_faces_path / student_name
        
        if not student_folder.exists():
            logger.error(f"Student folder not found: {student_folder}")
            return None
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(student_folder.glob(ext))
            image_files.extend(student_folder.glob(ext.upper()))
        
        if len(image_files) < 3:
            logger.warning(f"Expected 3 images for {student_name}, found {len(image_files)}")
        
        # Sort files for consistent processing
        image_files.sort()
        
        # Initialize student profile
        student_id = hashlib.md5(student_name.encode()).hexdigest()[:8]
        profile = StudentProfile(
            student_id=student_id,
            name=student_name,
            quality_scores={},
            created_at=datetime.now().isoformat()
        )
        
        embeddings = {}
        landmarks = {}
        
        # Process each image
        for idx, image_path in enumerate(image_files[:3]):  # Process up to 3 images
            try:
                logger.info(f"Processing {student_name}: {image_path.name}")
                
                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.error(f"Could not load image: {image_path}")
                    continue
                
                # Determine image type (front, left, right)
                image_type = self._determine_image_type(image_path.name, idx)
                
                # Extract facial features
                detections = self.extractor.process_image(image)
                
                if not detections:
                    logger.warning(f"No faces detected in {image_path}")
                    continue
                
                # Use the best detection (highest confidence)
                best_detection = max(detections, key=lambda d: d.confidence)
                
                # Calculate quality score
                quality_score = self._calculate_image_quality(image, best_detection)
                profile.quality_scores[image_type] = quality_score
                
                # Store embedding and landmarks
                if best_detection.embedding is not None:
                    embeddings[image_type] = best_detection.embedding
                    
                if best_detection.landmarks is not None:
                    landmarks[image_type] = best_detection.landmarks
                
                logger.info(f"  {image_type} view: confidence={best_detection.confidence:.3f}, quality={quality_score:.3f}")
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                continue
        
        # Assign embeddings to profile
        profile.front_embedding = embeddings.get('front')
        profile.left_embedding = embeddings.get('left')
        profile.right_embedding = embeddings.get('right')
        
        profile.front_landmarks = landmarks.get('front')
        profile.left_landmarks = landmarks.get('left')
        profile.right_landmarks = landmarks.get('right')
        
        # Create combined embedding (average of available embeddings)
        available_embeddings = [emb for emb in embeddings.values() if emb is not None]
        if available_embeddings:
            profile.combined_embedding = np.mean(available_embeddings, axis=0)
        
        profile.updated_at = datetime.now().isoformat()
        
        logger.info(f"Processed {student_name}: {len(available_embeddings)}/3 views successful")
        return profile
    
    def build_database(self) -> int:
        """
        Build the complete database from all student folders
        
        Returns:
            Number of students processed successfully
        """
        logger.info("Building known faces database...")
        
        if not self.known_faces_path.exists():
            logger.error(f"Known faces path not found: {self.known_faces_path}")
            return 0
        
        # Get all student folders
        student_folders = [f for f in self.known_faces_path.iterdir() if f.is_dir()]
        successful_count = 0
        
        for student_folder in student_folders:
            student_name = student_folder.name
            
            try:
                profile = self.process_student_folder(student_name)
                
                if profile and profile.combined_embedding is not None:
                    self.student_profiles[student_name] = profile
                    self._save_profile_to_db(profile)
                    successful_count += 1
                else:
                    logger.warning(f"Failed to create profile for {student_name}")
                    
            except Exception as e:
                logger.error(f"Error processing student {student_name}: {e}")
        
        # Save embeddings to pickle file
        self._save_embeddings_to_file()
        
        logger.info(f"Database build complete: {successful_count}/{len(student_folders)} students processed")
        return successful_count
    
    def _save_profile_to_db(self, profile: StudentProfile):
        """Save student profile to SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert numpy arrays to binary data
        def serialize_array(arr):
            return pickle.dumps(arr) if arr is not None else None
        
        cursor.execute('''
            INSERT OR REPLACE INTO students 
            (student_id, name, front_embedding, left_embedding, right_embedding,
             combined_embedding, front_landmarks, left_landmarks, right_landmarks,
             quality_scores, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            profile.student_id,
            profile.name,
            serialize_array(profile.front_embedding),
            serialize_array(profile.left_embedding),
            serialize_array(profile.right_embedding),
            serialize_array(profile.combined_embedding),
            serialize_array(profile.front_landmarks),
            serialize_array(profile.left_landmarks),
            serialize_array(profile.right_landmarks),
            json.dumps(profile.quality_scores),
            profile.created_at,
            profile.updated_at
        ))
        
        conn.commit()
        conn.close()
    
    def _save_embeddings_to_file(self):
        """Save all embeddings to a pickle file for fast loading"""
        embeddings_data = {
            'student_embeddings': {},
            'metadata': {
                'total_students': len(self.student_profiles),
                'created_at': datetime.now().isoformat(),
                'extractor_config': asdict(self.extractor.config)
            }
        }
        
        for name, profile in self.student_profiles.items():
            if profile.combined_embedding is not None:
                embeddings_data['student_embeddings'][name] = {
                    'student_id': profile.student_id,
                    'combined_embedding': profile.combined_embedding,
                    'front_embedding': profile.front_embedding,
                    'left_embedding': profile.left_embedding,
                    'right_embedding': profile.right_embedding,
                    'quality_scores': profile.quality_scores
                }
        
        with open(self.embeddings_path, 'wb') as f:
            pickle.dump(embeddings_data, f)
        
        logger.info(f"Embeddings saved to {self.embeddings_path}")
    
    def load_embeddings_from_file(self) -> bool:
        """Load embeddings from pickle file"""
        if not os.path.exists(self.embeddings_path):
            return False
        
        try:
            with open(self.embeddings_path, 'rb') as f:
                data = pickle.load(f)
            
            self.student_profiles = {}
            for name, student_data in data['student_embeddings'].items():
                profile = StudentProfile(
                    student_id=student_data['student_id'],
                    name=name,
                    combined_embedding=student_data['combined_embedding'],
                    front_embedding=student_data.get('front_embedding'),
                    left_embedding=student_data.get('left_embedding'),
                    right_embedding=student_data.get('right_embedding'),
                    quality_scores=student_data.get('quality_scores', {})
                )
                self.student_profiles[name] = profile
            
            logger.info(f"Loaded {len(self.student_profiles)} student profiles from {self.embeddings_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return False
    
    def get_student_stats(self) -> Dict:
        """Get statistics about the student database"""
        stats = {
            'total_students': len(self.student_profiles),
            'students_with_all_views': 0,
            'students_with_partial_views': 0,
            'average_quality_scores': {}
        }
        
        quality_sums = {'front': 0, 'left': 0, 'right': 0}
        quality_counts = {'front': 0, 'left': 0, 'right': 0}
        
        for profile in self.student_profiles.values():
            view_count = sum([
                1 for emb in [profile.front_embedding, profile.left_embedding, profile.right_embedding]
                if emb is not None
            ])
            
            if view_count == 3:
                stats['students_with_all_views'] += 1
            elif view_count > 0:
                stats['students_with_partial_views'] += 1
            
            # Accumulate quality scores
            if profile.quality_scores:
                for view, score in profile.quality_scores.items():
                    if view in quality_sums:
                        quality_sums[view] += score
                        quality_counts[view] += 1
        
        # Calculate average quality scores
        for view in quality_sums:
            if quality_counts[view] > 0:
                stats['average_quality_scores'][view] = quality_sums[view] / quality_counts[view]
        
        return stats
    
    def add_new_student(self, student_name, student_dir):
        """Add a new student to the face database"""
        try:
            from facial_features_extractor import FacialFeaturesExtractor
            import cv2
            import numpy as np
            
            # Initialize feature extractor if not already done
            if not hasattr(self, 'feature_extractor'):
                self.feature_extractor = FacialFeaturesExtractor()
            
            # Process the three images
            embeddings_list = []
            landmarks_dict = {}
            quality_scores = {}
            
            for position in ['F', 'L', 'R']:  # front, left, right
                image_path = os.path.join(student_dir, f"{position}.jpg")
                if os.path.exists(image_path):
                    # Load and process image
                    image = cv2.imread(image_path)
                    if image is not None:
                        # Extract features
                        faces = self.feature_extractor.detect_faces(image)
                        
                        if faces:
                            face = faces[0]  # Take the first face
                            embedding = self.feature_extractor.get_face_embedding(image, face)
                            landmarks = self.feature_extractor.get_face_landmarks(image, face)
                            
                            if embedding is not None:
                                embeddings_list.append(embedding)
                                landmarks_dict[position] = landmarks
                                quality_scores[position] = face.get('confidence', 0.0)
                                
                                logger.info(f"Processed {position} view for {student_name}")
            
            if embeddings_list:
                # Create combined embedding (average of all views)
                combined_embedding = np.mean(embeddings_list, axis=0)
                
                # Create student profile
                profile = StudentProfile(
                    name=student_name,
                    student_id=f"STU_{student_name.replace(' ', '_').upper()}",
                    combined_embedding=combined_embedding,
                    quality_scores=quality_scores
                )
                
                # Set individual embeddings if available
                if len(embeddings_list) >= 1:
                    profile.front_embedding = embeddings_list[0]
                if len(embeddings_list) >= 2:
                    profile.left_embedding = embeddings_list[1]
                if len(embeddings_list) >= 3:
                    profile.right_embedding = embeddings_list[2]
                
                # Add to database
                self.student_profiles[student_name] = profile
                
                # Save to pickle file
                self.save_embeddings_to_file()
                
                logger.info(f"Added new student {student_name} to face database with {len(embeddings_list)} views")
                return True
            else:
                logger.error(f"No valid face embeddings found for {student_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding new student {student_name}: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    # Initialize the database system
    db = KnownFacesDatabase()
    
    # Try to load existing embeddings first
    if not db.load_embeddings_from_file():
        print("No existing embeddings found. Building database from scratch...")
        success_count = db.build_database()
        print(f"Successfully processed {success_count} students")
    else:
        print("Loaded existing embeddings from file")
    
    # Display statistics
    stats = db.get_student_stats()
    print("\nDatabase Statistics:")
    print(f"Total students: {stats['total_students']}")
    print(f"Students with all 3 views: {stats['students_with_all_views']}")
    print(f"Students with partial views: {stats['students_with_partial_views']}")
    
    if stats['average_quality_scores']:
        print("\nAverage Quality Scores:")
        for view, score in stats['average_quality_scores'].items():
            print(f"  {view.capitalize()}: {score:.3f}")
    
    print("\nKnown Faces Database ready for attendance monitoring!")
