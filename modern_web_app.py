"""
Modern Attendance Monitoring System
Minimalistic, mobile-friendly UI with MongoDB integration and human feedback
"""

import os
import shutil
import tempfile
import uuid
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

from attendance_processor import process_classroom_video, VideoAttendanceProcessor, VideoProcessingConfig
from known_faces_database import KnownFacesDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB connection
MONGODB_URL = "mongodb+srv://nihaaly41:7849@attendence.cfgt2xs.mongodb.net/"
DATABASE_NAME = "attendance_system"

# Create FastAPI app
app = FastAPI(
    title="Modern Attendance Monitor",
    description="AI-powered attendance monitoring with human feedback",
    version="2.0.0"
)

# Enable CORS for mobile access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
processing_jobs: Dict[str, Dict[str, Any]] = {}
thread_pool = ThreadPoolExecutor(max_workers=2)
mongo_client: AsyncIOMotorClient = None

# Pydantic models
class AttendanceCorrection(BaseModel):
    session_id: str
    student_name: str
    original_status: str
    corrected_status: str
    reason: Optional[str] = None
    corrected_by: Optional[str] = "manual"

class ProcessingStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Create directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

@app.on_event("startup")
async def startup_db_client():
    """Initialize MongoDB connection during startup"""
    global mongo_client
    try:
        mongo_client = AsyncIOMotorClient(MONGODB_URL)
        # Test connection
        await mongo_client.admin.command('ping')
        logger.info("Connected to MongoDB successfully")
    except Exception as e:
        logger.warning(f"MongoDB connection failed: {e}. Using in-memory storage.")
        mongo_client = None

@app.on_event("shutdown")
async def shutdown_db_client():
    """Close MongoDB connection during shutdown"""
    global mongo_client
    if mongo_client:
        mongo_client.close()
        logger.info("MongoDB connection closed")

def get_mongo_client():
    """Get the global MongoDB client"""
    global mongo_client
    return mongo_client

def serialize_mongo_doc(doc):
    """Convert MongoDB document to JSON-serializable format"""
    if doc is None:
        return None
    if isinstance(doc, dict):
        serialized = {}
        for key, value in doc.items():
            if isinstance(value, ObjectId):
                serialized[key] = str(value)
            elif isinstance(value, dict):
                serialized[key] = serialize_mongo_doc(value)
            elif isinstance(value, list):
                serialized[key] = [serialize_mongo_doc(item) if isinstance(item, dict) else str(item) if isinstance(item, ObjectId) else item for item in value]
            else:
                serialized[key] = value
        return serialized
    return doc

def process_classroom_video_with_progress(video_path: str, job_id: str):
    """Process classroom video with progress updates"""
    # Load faces database
    faces_db = KnownFacesDatabase()
    if not faces_db.load_embeddings_from_file():
        logger.info("Building faces database from known_faces_optimized...")
        faces_db.build_database()
    
    # Configure processing
    config = VideoProcessingConfig(
        frame_skip_interval=15,
        min_detection_confidence=0.7,
        face_recognition_threshold=0.65,
        min_detections_for_presence=5,
        max_frames_to_process=None,
        output_frame_interval=300
    )
    
    # Create processor
    processor = VideoAttendanceProcessor(faces_db, config)
    
    # Open video to get frame count
    import cv2
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # Update progress periodically during processing
    def update_progress_callback(frame_count, total_frames):
        if job_id in processing_jobs:
            video_progress = (frame_count / total_frames) * 0.6
            total_progress = 0.2 + video_progress
            
            processing_jobs[job_id]["progress"] = min(total_progress, 0.8)
            processing_jobs[job_id]["message"] = f"Processing frame {frame_count}/{total_frames}..."
    
    # Process with progress updates
    results = process_video_with_callback(processor, video_path, update_progress_callback)
    
    return results

def process_video_with_callback(processor, video_path: str, progress_callback):
    """Process video with progress callback"""
    logger.info(f"Processing video: {video_path}")
    
    # Open video
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Process frames
    frame_count = 0
    processed_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = frame_count / fps if fps > 0 else frame_count
        
        # Skip frames for efficiency
        if frame_count % processor.config.frame_skip_interval != 0:
            continue
        
        processed_count += 1
        
        # Process frame
        frame_results = processor.process_frame(frame, timestamp)
        
        # Store frame results
        processor.frame_results.append({
            'frame_number': frame_count,
            'timestamp': timestamp,
            'detections': frame_results
        })
        
        # Update progress every 2 processed frames
        if processed_count % 2 == 0:
            progress_callback(frame_count, total_frames)
    
    cap.release()
    
    # Generate final results
    attendance_results = processor.generate_attendance_results()
    
    return attendance_results

def process_video_background(job_id: str, video_path: str, original_filename: str):
    """Background task to process video"""
    try:
        logger.info(f"Starting background processing for job {job_id}")
        
        # Update status with granular progress
        processing_jobs[job_id]["status"] = "processing"
        processing_jobs[job_id]["message"] = "Initializing AI models..."
        processing_jobs[job_id]["progress"] = 0.1
        
        time.sleep(0.5)
        processing_jobs[job_id]["progress"] = 0.15
        processing_jobs[job_id]["message"] = "Loading face database..."
        
        time.sleep(0.5)
        processing_jobs[job_id]["progress"] = 0.2
        processing_jobs[job_id]["message"] = "Starting video analysis..."
        
        # Process the video
        results = process_classroom_video_with_progress(video_path, job_id)
        
        # Final steps
        processing_jobs[job_id]["progress"] = 0.85
        processing_jobs[job_id]["message"] = "Analyzing results..."
        
        time.sleep(0.3)
        processing_jobs[job_id]["progress"] = 0.95
        processing_jobs[job_id]["message"] = "Saving to database..."
        
        # MongoDB save will be handled by the main thread when results are retrieved
        
        # Create final result
        final_result = create_attendance_result(job_id, results, original_filename)
        
        # Complete
        processing_jobs[job_id]["status"] = "completed"
        processing_jobs[job_id]["progress"] = 1.0
        processing_jobs[job_id]["message"] = "Processing complete!"
        processing_jobs[job_id]["completed_at"] = datetime.now()
        processing_jobs[job_id]["result"] = final_result
        
        logger.info(f"Background processing completed for job {job_id}")
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}")
        processing_jobs[job_id]["status"] = "error"
        processing_jobs[job_id]["error"] = str(e)
        processing_jobs[job_id]["message"] = f"Error: {str(e)}"

def create_attendance_result(job_id: str, results: Dict, original_filename: str) -> Dict:
    """Create structured attendance result"""
    present_students = [
        {
            "name": name,
            "confidence": result.confidence,
            "detections": result.detection_count,
            "first_seen": result.first_detected_time,
            "last_seen": result.last_detected_time
        }
        for name, result in results.items() 
        if result.status == 'present'
    ]
    
    uncertain_students = [
        {
            "name": name,
            "confidence": result.confidence,
            "detections": result.detection_count
        }
        for name, result in results.items() 
        if result.status == 'uncertain'
    ]
    
    absent_students = [
        name for name, result in results.items() 
        if result.status == 'absent'
    ]
    
    return {
        "session_id": job_id,
        "video_filename": original_filename,
        "total_students": len(results),
        "present_count": len(present_students),
        "absent_count": len(absent_students),
        "uncertain_count": len(uncertain_students),
        "attendance_rate": len(present_students) / len(results) * 100,
        "present_students": sorted(present_students, key=lambda x: x["name"]),
        "absent_students": sorted(absent_students),
        "uncertain_students": sorted(uncertain_students, key=lambda x: x["name"]),
        "timestamp": datetime.now(),
        "corrections": []
    }

async def save_attendance_session(session_id: str, results: Dict, filename: str):
    """Save attendance session to MongoDB"""
    client = get_mongo_client()
    if not client:
        return
    
    try:
        db = client[DATABASE_NAME]
        collection = db.attendance_sessions
        
        session_data = {
            "session_id": session_id,
            "filename": filename,
            "timestamp": datetime.now(timezone.utc),
            "results": {name: {
                "status": result["status"] if isinstance(result, dict) else result.status,
                "confidence": result["confidence"] if isinstance(result, dict) else result.confidence,
                "detections": result["detection_count"] if isinstance(result, dict) else result.detection_count,
                "first_seen": result["first_detected_time"] if isinstance(result, dict) else result.first_detected_time,
                "last_seen": result["last_detected_time"] if isinstance(result, dict) else result.last_detected_time
            } for name, result in results.items()},
            "corrections": []
        }
        
        result = await collection.insert_one(session_data)
        logger.info(f"Saved session {session_id} to MongoDB with ID: {result.inserted_id}")
        
    except Exception as e:
        logger.error(f"Error saving to MongoDB: {e}")

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the modern web interface"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>Attendance Monitor</title>
    <meta name="theme-color" content="#6366f1">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        :root {
            --background: 0 0% 100%;
            --foreground: 222.2 84% 4.9%;
            --card: 0 0% 100%;
            --card-foreground: 222.2 84% 4.9%;
            --popover: 0 0% 100%;
            --popover-foreground: 222.2 84% 4.9%;
            --primary: 221.2 83.2% 53.3%;
            --primary-foreground: 210 40% 98%;
            --secondary: 210 40% 96%;
            --secondary-foreground: 222.2 84% 4.9%;
            --muted: 210 40% 96%;
            --muted-foreground: 215.4 16.3% 46.9%;
            --accent: 210 40% 96%;
            --accent-foreground: 222.2 84% 4.9%;
            --destructive: 0 84.2% 60.2%;
            --destructive-foreground: 210 40% 98%;
            --border: 214.3 31.8% 91.4%;
            --input: 214.3 31.8% 91.4%;
            --ring: 221.2 83.2% 53.3%;
            --radius: 0.5rem;
            --success: 142.1 76.2% 36.3%;
            --warning: 47.9 95.8% 53.1%;
            --shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: hsl(var(--background));
            color: hsl(var(--foreground));
            line-height: 1.5;
            -webkit-font-smoothing: antialiased;
            font-feature-settings: "rlig" 1, "calt" 1;
        }
        
        .container {
            min-height: 100vh;
            max-width: 640px;
            margin: 0 auto;
            background: var(--surface);
            box-shadow: var(--shadow-lg);
        }
        
        .header {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            color: white;
            padding: 2rem 1.5rem 1.5rem;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        
        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="20" cy="20" r="2" fill="white" opacity="0.1"/><circle cx="80" cy="40" r="1" fill="white" opacity="0.1"/><circle cx="40" cy="80" r="1.5" fill="white" opacity="0.1"/></svg>');
            pointer-events: none;
        }
        
        .header h1 {
            font-size: 1.75rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            position: relative;
        }
        
        .header p {
            opacity: 0.9;
            font-size: 0.9rem;
            position: relative;
        }
        
        .content {
            padding: 1.5rem;
        }
        
        .upload-card {
            background: var(--surface);
            border: 2px dashed var(--border);
            border-radius: 1rem;
            padding: 2rem 1rem;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
            margin-bottom: 2rem;
        }
        
        .upload-card:hover {
            border-color: var(--primary);
            background: #f8faff;
        }
        
        .upload-card.dragover {
            border-color: var(--primary);
            background: #f0f4ff;
            transform: scale(1.02);
        }
        
        .upload-icon {
            width: 3rem;
            height: 3rem;
            background: var(--primary);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 1rem;
            font-size: 1.5rem;
        }
        
        .upload-text {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text);
            margin-bottom: 0.5rem;
        }
        
        .upload-subtext {
            color: var(--text-light);
            font-size: 0.875rem;
        }
        
        #fileInput {
            display: none;
        }
        
        .progress-card {
            display: none;
            background: var(--surface);
            border-radius: 1rem;
            padding: 1.5rem;
            box-shadow: var(--shadow);
            margin-bottom: 2rem;
        }
        
        .progress-bar {
            background: var(--border);
            border-radius: 1rem;
            height: 0.5rem;
            overflow: hidden;
            margin-bottom: 1rem;
        }
        
        .progress-fill {
            background: linear-gradient(90deg, var(--primary), var(--primary-dark));
            height: 100%;
            width: 0%;
            transition: width 0.3s ease;
            border-radius: 1rem;
        }
        
        .progress-text {
            font-size: 0.875rem;
            color: var(--text-light);
            text-align: center;
        }
        
        .results-card {
            display: none;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .stat-card {
            background: var(--surface);
            border-radius: 1rem;
            padding: 1.5rem;
            text-align: center;
            box-shadow: var(--shadow);
            border-left: 4px solid var(--primary);
        }
        
        .stat-number {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        
        .stat-label {
            font-size: 0.75rem;
            color: var(--text-light);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 500;
        }
        
        .stat-card.present .stat-number { color: var(--success); }
        .stat-card.absent .stat-number { color: var(--error); }
        .stat-card.uncertain .stat-number { color: var(--warning); }
        
        .section-card {
            background: var(--surface);
            border-radius: 1rem;
            margin-bottom: 1rem;
            box-shadow: var(--shadow);
            overflow: hidden;
        }
        
        .section-header {
            padding: 1rem 1.5rem;
            font-weight: 600;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border-bottom: 1px solid var(--border);
        }
        
        .section-header.present { background: #f0fdf4; color: var(--success); }
        .section-header.absent { background: #fef2f2; color: var(--error); }
        .section-header.uncertain { background: #fefbeb; color: var(--warning); }
        
        .student-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--border);
        }
        
        .student-item:last-child {
            border-bottom: none;
        }
        
        .student-info {
            flex: 1;
        }
        
        .student-name {
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        
        .student-details {
            font-size: 0.75rem;
            color: var(--text-light);
        }
        
        .correction-btn {
            background: var(--success);
            color: white;
            border: none;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            font-size: 0.75rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .correction-btn:hover {
            background: #059669;
            transform: translateY(-1px);
        }
        
        .correction-btn:active {
            transform: translateY(0);
        }
        
        .correction-btn.corrected {
            background: #10b981;
            color: white;
            cursor: not-allowed;
        }
        
        .correction-btn.corrected-warning {
            background: #f59e0b;
            color: white;
            cursor: not-allowed;
        }
        
        .correction-btn.corrected-error {
            background: #ef4444;
            color: white;
            cursor: not-allowed;
        }
        
        /* Results header with download buttons */
        .results-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--primary-color);
        }
        
        .results-actions {
            display: flex;
            gap: 0.5rem;
        }
        
        .download-btn {
            background: var(--primary-color);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.3s ease;
        }
        
        .download-btn:hover {
            background: var(--primary-dark);
            transform: translateY(-2px);
        }
        
        /* History card styles */
        .history-card {
            background: white;
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            margin-top: 1rem;
            border: 1px solid #e5e7eb;
        }
        
        .history-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--secondary-color);
        }
        
        .refresh-btn {
            background: var(--secondary-color);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.3s ease;
        }
        
        .refresh-btn:hover {
            background: #059669;
            transform: translateY(-2px);
        }
        
        .history-content {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .history-session {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 0.75rem;
            transition: all 0.3s ease;
        }
        
        .history-session:hover {
            background: #f1f5f9;
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .session-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }
        
        .session-title {
            font-weight: 600;
            color: var(--text-dark);
            font-size: 1rem;
        }
        
        .session-date {
            color: var(--text-light);
            font-size: 0.85rem;
        }
        
        .session-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
            gap: 0.5rem;
            margin-top: 0.5rem;
        }
        
        .session-stat {
            text-align: center;
            padding: 0.25rem;
            border-radius: 6px;
            font-size: 0.8rem;
        }
        
        .session-stat.present {
            background: #dcfce7;
            color: #166534;
        }
        
        .session-stat.absent {
            background: #fee2e2;
            color: #991b1b;
        }
        
        .session-stat.uncertain {
            background: #fef3c7;
            color: #92400e;
        }
        
        .session-stat.corrections {
            background: #e0e7ff;
            color: #3730a3;
        }
        
        .session-actions {
            display: flex;
            gap: 0.5rem;
            margin-top: 0.75rem;
        }
        
        .session-download-btn {
            background: transparent;
            border: 1px solid var(--primary-color);
            color: var(--primary-color);
            padding: 0.25rem 0.75rem;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.8rem;
            transition: all 0.3s ease;
        }
        
        .session-download-btn:hover {
            background: var(--primary-color);
            color: white;
        }
        
        .loading-message {
            text-align: center;
            color: var(--text-light);
            padding: 2rem;
            font-style: italic;
        }
        
        /* Improved button states */
        .correction-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none !important;
        }
        
        .correction-btn.processing {
            background: #6b7280;
            color: white;
            cursor: not-allowed;
        }
        
        /* Dual buttons for uncertain students */
        .dual-buttons {
            display: flex;
            gap: 0.5rem;
        }
        
        .dual-buttons .correction-btn {
            flex: 1;
            font-size: 0.8rem;
            padding: 0.4rem 0.8rem;
        }
        
        .mark-present-btn {
            background: var(--success-color);
        }
        
        .mark-present-btn:hover {
            background: #059669;
        }
        
        .mark-absent-btn {
            background: var(--error-color);
        }
        
        .mark-absent-btn:hover {
            background: #dc2626;
        }
        
        /* Shadcn-style components */
        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            white-space: nowrap;
            border-radius: calc(var(--radius) - 2px);
            font-size: 0.875rem;
            font-weight: 500;
            transition: colors 0.2s;
            border: none;
            cursor: pointer;
            outline: none;
            text-decoration: none;
        }
        
        .btn:focus-visible {
            outline: 2px solid hsl(var(--ring));
            outline-offset: 2px;
        }
        
        .btn:disabled {
            pointer-events: none;
            opacity: 0.5;
        }
        
        .btn-primary {
            background: hsl(var(--primary));
            color: hsl(var(--primary-foreground));
            height: 2.25rem;
            padding: 0 1rem;
        }
        
        .btn-primary:hover {
            background: hsl(var(--primary) / 0.9);
        }
        
        .btn-secondary {
            background: hsl(var(--secondary));
            color: hsl(var(--secondary-foreground));
            height: 2.25rem;
            padding: 0 1rem;
        }
        
        .btn-secondary:hover {
            background: hsl(var(--secondary) / 0.8);
        }
        
        .btn-destructive {
            background: hsl(var(--destructive));
            color: hsl(var(--destructive-foreground));
            height: 2.25rem;
            padding: 0 1rem;
        }
        
        .btn-destructive:hover {
            background: hsl(var(--destructive) / 0.9);
        }
        
        .btn-outline {
            border: 1px solid hsl(var(--border));
            background: hsl(var(--background));
            color: hsl(var(--foreground));
            height: 2.25rem;
            padding: 0 1rem;
        }
        
        .btn-outline:hover {
            background: hsl(var(--accent));
            color: hsl(var(--accent-foreground));
        }
        
        .btn-ghost {
            background: transparent;
            color: hsl(var(--foreground));
            height: 2.25rem;
            padding: 0 1rem;
        }
        
        .btn-ghost:hover {
            background: hsl(var(--accent));
            color: hsl(var(--accent-foreground));
        }
        
        .btn-sm {
            height: 2rem;
            border-radius: calc(var(--radius) - 2px);
            padding: 0 0.75rem;
            font-size: 0.8125rem;
        }
        
        .btn-lg {
            height: 2.75rem;
            border-radius: var(--radius);
            padding: 0 2rem;
            font-size: 1rem;
        }
        
        /* Card component */
        .card {
            border-radius: var(--radius);
            border: 1px solid hsl(var(--border));
            background: hsl(var(--card));
            color: hsl(var(--card-foreground));
            box-shadow: var(--shadow);
        }
        
        .card-header {
            display: flex;
            flex-direction: column;
            space-y: 0.375rem;
            padding: 1.5rem;
            padding-bottom: 0;
        }
        
        .card-title {
            font-size: 1.5rem;
            font-weight: 600;
            line-height: 1;
            letter-spacing: -0.025em;
        }
        
        .card-description {
            font-size: 0.875rem;
            color: hsl(var(--muted-foreground));
        }
        
        .card-content {
            padding: 1.5rem;
            padding-top: 0;
        }
        
        .card-footer {
            display: flex;
            align-items: center;
            padding: 1.5rem;
            padding-top: 0;
        }
        
        /* Badge component */
        .badge {
            display: inline-flex;
            align-items: center;
            border-radius: 9999px;
            padding: 0.125rem 0.625rem;
            font-size: 0.75rem;
            font-weight: 600;
            line-height: 1;
            transition: colors 0.2s;
            border: 1px solid transparent;
        }
        
        .badge-default {
            border-color: transparent;
            background: hsl(var(--primary));
            color: hsl(var(--primary-foreground));
        }
        
        .badge-secondary {
            border-color: transparent;
            background: hsl(var(--secondary));
            color: hsl(var(--secondary-foreground));
        }
        
        .badge-destructive {
            border-color: transparent;
            background: hsl(var(--destructive));
            color: hsl(var(--destructive-foreground));
        }
        
        .badge-outline {
            color: hsl(var(--foreground));
            border-color: hsl(var(--border));
        }
        
        .badge-success {
            border-color: transparent;
            background: hsl(var(--success));
            color: white;
        }
        
        .badge-warning {
            border-color: transparent;
            background: hsl(var(--warning));
            color: hsl(var(--foreground));
        }
        
        /* Modal/Dialog */
        .modal-overlay {
            position: fixed;
            inset: 0;
            z-index: 50;
            background: rgba(0, 0, 0, 0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1rem;
            animation: fadeIn 0.2s ease-out;
        }
        
        .modal-content {
            background: hsl(var(--background));
            border: 1px solid hsl(var(--border));
            border-radius: var(--radius);
            box-shadow: var(--shadow-lg);
            max-width: 32rem;
            width: 100%;
            max-height: 85vh;
            overflow-y: auto;
            animation: slideIn 0.2s ease-out;
        }
        
        .modal-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1.5rem;
            border-bottom: 1px solid hsl(var(--border));
        }
        
        .modal-title {
            font-size: 1.125rem;
            font-weight: 600;
            line-height: 1;
        }
        
        .modal-close {
            opacity: 0.7;
            background: transparent;
            border: none;
            border-radius: 0.375rem;
            width: 1.5rem;
            height: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: opacity 0.2s;
        }
        
        .modal-close:hover {
            opacity: 1;
        }
        
        .modal-body {
            padding: 1.5rem;
        }
        
        .modal-footer {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            padding: 1.5rem;
            padding-top: 0;
        }
        
        @media (min-width: 640px) {
            .modal-footer {
                flex-direction: row;
                justify-content: flex-end;
            }
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        @keyframes slideIn {
            from { 
                opacity: 0;
                transform: scale(0.95) translateY(-10px);
            }
            to { 
                opacity: 1;
                transform: scale(1) translateY(0);
            }
        }
        
        /* History session clickable styles */
        .history-session {
            background: hsl(var(--card));
            border: 1px solid hsl(var(--border));
            border-radius: var(--radius);
            padding: 1rem;
            margin-bottom: 0.75rem;
            transition: all 0.2s ease;
            cursor: pointer;
        }
        
        .history-session:hover {
            background: hsl(var(--muted));
            border-color: hsl(var(--primary));
            transform: translateY(-1px);
            box-shadow: var(--shadow-lg);
        }
        
        .history-session.active {
            border-color: hsl(var(--primary));
            background: hsl(var(--primary) / 0.05);
        }
        
        /* Utility classes */
        .flex { display: flex; }
        .grid { display: grid; }
        .space-x-2 > * + * { margin-left: 0.5rem; }
        .space-y-2 > * + * { margin-top: 0.5rem; }
        .space-y-4 > * + * { margin-top: 1rem; }
        .gap-3 { gap: 0.75rem; }
        .gap-4 { gap: 1rem; }
        .grid-cols-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
        .justify-between { justify-content: space-between; }
        .justify-center { justify-content: center; }
        .items-center { align-items: center; }
        .items-start { align-items: flex-start; }
        .text-center { text-align: center; }
        .text-sm { font-size: 0.875rem; }
        .text-xs { font-size: 0.75rem; }
        .text-base { font-size: 1rem; }
        .text-2xl { font-size: 1.5rem; }
        .font-semibold { font-weight: 600; }
        .font-medium { font-weight: 500; }
        .font-bold { font-weight: 700; }
        .mb-3 { margin-bottom: 0.75rem; }
        .mb-4 { margin-bottom: 1rem; }
        .mt-1 { margin-top: 0.25rem; }
        .p-2 { padding: 0.5rem; }
        .rounded { border-radius: 0.25rem; }
        .border { border-width: 1px; }
        .bg-white { background-color: white; }
        .bg-green-50 { background-color: #f0fdf4; }
        .bg-red-50 { background-color: #fef2f2; }
        .bg-yellow-50 { background-color: #fefce8; }
        .border-green-200 { border-color: #bbf7d0; }
        .border-red-200 { border-color: #fecaca; }
        .border-yellow-200 { border-color: #fef08a; }
        .text-green-600 { color: #16a34a; }
        .text-red-600 { color: #dc2626; }
        .text-yellow-600 { color: #ca8a04; }
        .text-blue-600 { color: #2563eb; }
        .text-muted-foreground { color: hsl(var(--muted-foreground)); }
        .flex-1 { flex: 1 1 0%; }
        
        /* Tab Navigation */
        .tabs-container {
            margin: 2rem 0;
        }
        
        .tabs {
            display: flex;
            background: hsl(var(--muted));
            border-radius: var(--radius);
            padding: 0.25rem;
            overflow-x: auto;
        }
        
        .tab-btn {
            flex: 1;
            min-width: 120px;
            padding: 0.75rem 1rem;
            border: none;
            background: transparent;
            color: hsl(var(--muted-foreground));
            border-radius: calc(var(--radius) - 2px);
            cursor: pointer;
            transition: all 0.2s;
            font-weight: 500;
            white-space: nowrap;
        }
        
        .tab-btn:hover {
            background: hsl(var(--background));
            color: hsl(var(--foreground));
        }
        
        .tab-btn.active {
            background: hsl(var(--background));
            color: hsl(var(--foreground));
            box-shadow: var(--shadow);
        }
        
        /* Tab Content */
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        /* Student Management */
        .add-student-form {
            background: hsl(var(--card));
            border: 1px solid hsl(var(--border));
            border-radius: var(--radius);
            padding: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .form-group {
            margin-bottom: 1rem;
        }
        
        .form-label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 500;
            color: hsl(var(--foreground));
        }
        
        .form-input {
            width: 100%;
            padding: 0.5rem 0.75rem;
            border: 1px solid hsl(var(--border));
            border-radius: calc(var(--radius) - 2px);
            background: hsl(var(--background));
            color: hsl(var(--foreground));
            font-size: 0.875rem;
        }
        
        .form-input:focus {
            outline: none;
            border-color: hsl(var(--ring));
            box-shadow: 0 0 0 2px hsl(var(--ring) / 0.2);
        }
        
        .photo-upload-group {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .photo-upload {
            border: 2px dashed hsl(var(--border));
            border-radius: var(--radius);
            padding: 1rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .photo-upload:hover {
            border-color: hsl(var(--primary));
            background: hsl(var(--primary) / 0.05);
        }
        
        .photo-upload.has-file {
            border-color: hsl(var(--success));
            background: hsl(var(--success) / 0.1);
        }
        
        .photo-preview {
            width: 100px;
            height: 100px;
            object-fit: cover;
            border-radius: var(--radius);
            margin-bottom: 0.5rem;
        }
        
        .students-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .student-card {
            background: hsl(var(--card));
            border: 1px solid hsl(var(--border));
            border-radius: var(--radius);
            padding: 1rem;
            transition: all 0.2s;
        }
        
        .student-card:hover {
            box-shadow: var(--shadow-lg);
            transform: translateY(-2px);
        }
        
        .student-photos {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }
        
        .student-photo {
            width: 60px;
            height: 60px;
            object-fit: cover;
            border-radius: var(--radius);
            border: 1px solid hsl(var(--border));
        }
        
        .analytics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .analytics-card {
            background: hsl(var(--card));
            border: 1px solid hsl(var(--border));
            border-radius: var(--radius);
            padding: 1.5rem;
            text-align: center;
        }
        
        .analytics-value {
            font-size: 2rem;
            font-weight: 700;
            color: hsl(var(--primary));
            margin-bottom: 0.5rem;
        }
        
        .analytics-label {
            color: hsl(var(--muted-foreground));
            font-size: 0.875rem;
        }
        
        .error-toast {
            position: fixed;
            top: 1rem;
            left: 50%;
            transform: translateX(-50%);
            background: var(--error);
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 0.5rem;
            font-size: 0.875rem;
            font-weight: 500;
            box-shadow: var(--shadow-lg);
            z-index: 1000;
            opacity: 0;
            animation: slideIn 0.3s ease forwards;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-50%) translateY(-100%);
            }
            to {
                opacity: 1;
                transform: translateX(-50%) translateY(0);
            }
        }
        
        @keyframes slideOut {
            from {
                opacity: 1;
                transform: translateX(-50%) translateY(0);
            }
            to {
                opacity: 0;
                transform: translateX(-50%) translateY(-100%);
            }
        }
        
        .fab {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            width: 3.5rem;
            height: 3.5rem;
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 50%;
            font-size: 1.5rem;
            cursor: pointer;
            box-shadow: var(--shadow-lg);
            transition: all 0.3s ease;
            z-index: 100;
        }
        
        .fab:hover {
            background: var(--primary-dark);
            transform: scale(1.1);
        }
        
        @media (max-width: 480px) {
            .container {
                min-height: 100vh;
            }
            
            .header {
                padding: 1.5rem 1rem 1rem;
            }
            
            .content {
                padding: 1rem;
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
                gap: 0.75rem;
            }
            
            .stat-card {
                padding: 1rem;
            }
            
            .fab {
                bottom: 1rem;
                right: 1rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎓 University Attendance Management System</h1>
            <p>Complete solution for classroom attendance tracking and student management</p>
        </div>
        
        <!-- Navigation Tabs -->
        <div class="tabs-container">
            <div class="tabs">
                <button class="tab-btn active" onclick="switchTab('attendance')">📹 Take Attendance</button>
                <button class="tab-btn" onclick="switchTab('students')">👥 Manage Students</button>
                <button class="tab-btn" onclick="switchTab('classes')">🏛️ Classes</button>
                <button class="tab-btn" onclick="switchTab('analytics')">📊 Analytics</button>
                <button class="tab-btn" onclick="switchTab('history')">📚 History</button>
            </div>
        </div>
        
        <div class="content">
            <!-- Take Attendance Tab -->
            <div id="attendance-tab" class="tab-content active">
                <div class="upload-card" id="uploadCard" onclick="document.getElementById('fileInput').click()">
                    <div class="upload-icon">📹</div>
                    <div class="upload-text">Upload Classroom Video</div>
                    <div class="upload-subtext">Tap here or drag and drop your video</div>
                    <input type="file" id="fileInput" accept="video/*" onchange="handleFileSelect(event)">
                </div>
            
            <div class="progress-card" id="progressCard">
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <div class="progress-text" id="progressText">Preparing upload...</div>
            </div>
            
            <div class="results-card" id="resultsCard">
                <div class="results-header">
                    <h2>Current Session Results</h2>
                    <div class="results-actions">
                        <button class="download-btn" id="downloadCsvBtn" onclick="downloadReport('csv')" style="display: none;">
                            📊 Download CSV
                        </button>
                        <button class="download-btn" id="downloadJsonBtn" onclick="downloadReport('json')" style="display: none;">
                            📋 Download JSON
                        </button>
                    </div>
                </div>
                
                <div class="stats-grid" id="statsGrid">
                    <!-- Stats will be populated by JavaScript -->
                </div>
                
                <div id="studentSections">
                    <!-- Student sections will be populated by JavaScript -->
                </div>
            </div>
            
            </div>
            
            <!-- Manage Students Tab -->
            <div id="students-tab" class="tab-content">
                <div class="add-student-form">
                    <h3 class="card-title mb-4">➕ Add New Student</h3>
                    
                    <form id="addStudentForm" onsubmit="addNewStudent(event)">
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                            <div class="form-group">
                                <label class="form-label">Student Name</label>
                                <input type="text" name="name" class="form-input" required placeholder="Enter full name">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Student ID</label>
                                <input type="text" name="student_id" class="form-input" required placeholder="Enter student ID">
                            </div>
                        </div>
                        
                        <div class="form-group">
                            <label class="form-label">Class</label>
                            <input type="text" name="class_name" class="form-input" required placeholder="Enter class name (e.g., CS-101, Math-202)">
                        </div>
                        
                        <div class="form-group">
                            <label class="form-label">Student Photos (Required: Left, Right, Front views)</label>
                            <div class="photo-upload-group">
                                <div class="photo-upload" onclick="document.getElementById('leftPhoto').click()">
                                    <div id="leftPreview">
                                        <div>📷</div>
                                        <div>Left View</div>
                                        <small>Click to upload</small>
                                    </div>
                                    <input type="file" id="leftPhoto" name="left_photo" accept="image/*" style="display: none" onchange="previewPhoto(this, 'leftPreview')">
                                </div>
                                
                                <div class="photo-upload" onclick="document.getElementById('rightPhoto').click()">
                                    <div id="rightPreview">
                                        <div>📷</div>
                                        <div>Right View</div>
                                        <small>Click to upload</small>
                                    </div>
                                    <input type="file" id="rightPhoto" name="right_photo" accept="image/*" style="display: none" onchange="previewPhoto(this, 'rightPreview')">
                                </div>
                                
                                <div class="photo-upload" onclick="document.getElementById('frontPhoto').click()">
                                    <div id="frontPreview">
                                        <div>📷</div>
                                        <div>Front View</div>
                                        <small>Click to upload</small>
                                    </div>
                                    <input type="file" id="frontPhoto" name="front_photo" accept="image/*" style="display: none" onchange="previewPhoto(this, 'frontPreview')">
                                </div>
                            </div>
                        </div>
                        
                        <button type="submit" class="btn btn-primary btn-lg">Add Student</button>
                    </form>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">👥 All Students</div>
                        <div class="card-description">Manage your student database</div>
                    </div>
                    <div class="card-content">
                        <div class="flex justify-between items-center mb-4">
                            <select id="classFilter" class="form-input" style="width: auto;" onchange="filterStudentsByClass()">
                                <option value="">All Classes</option>
                            </select>
                            <button class="btn btn-outline btn-sm" onclick="loadAllStudents()">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/>
                                    <path d="M21 3v5h-5"/>
                                    <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/>
                                    <path d="M3 21v-5h5"/>
                                </svg>
                                Refresh
                            </button>
                        </div>
                        
                        <div id="studentsGrid" class="students-grid">
                            <div class="loading-message">Loading students...</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Classes Tab -->
            <div id="classes-tab" class="tab-content">
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">🏛️ Class Management</div>
                        <div class="card-description">Overview of all classes and their students</div>
                    </div>
                    <div class="card-content">
                        <div id="classesGrid" class="analytics-grid">
                            <div class="loading-message">Loading classes...</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Analytics Tab -->
            <div id="analytics-tab" class="tab-content">
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">📊 Attendance Analytics</div>
                        <div class="card-description">Insights and trends from attendance data</div>
                    </div>
                    <div class="card-content">
                        <div class="flex justify-between items-center mb-4">
                            <select id="analyticsClassFilter" class="form-input" style="width: auto;" onchange="loadAnalytics()">
                                <option value="">All Classes</option>
                            </select>
                            <select id="analyticsPeriod" class="form-input" style="width: auto;" onchange="loadAnalytics()">
                                <option value="7">Last 7 days</option>
                                <option value="30" selected>Last 30 days</option>
                                <option value="90">Last 90 days</option>
                            </select>
                        </div>
                        
                        <div id="analyticsContent">
                            <div class="loading-message">Loading analytics...</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- History Tab -->
            <div id="history-tab" class="tab-content">
                <div class="card" id="historyCard">
                    <div class="card-header">
                        <div class="card-title">📚 Attendance History</div>
                        <div class="card-description">Click on any session to view and edit attendance records</div>
                    </div>
                    
                    <div class="card-content">
                        <div class="flex justify-between items-center mb-4">
                            <button class="btn btn-outline btn-sm" onclick="loadAttendanceHistory()">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/>
                                    <path d="M21 3v5h-5"/>
                                    <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/>
                                    <path d="M3 21v-5h5"/>
                                </svg>
                                Refresh
                            </button>
                        </div>
                        
                        <div id="historyContent">
                            <div class="loading-message">Loading attendance history...</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Edit Session Modal -->
            <div class="modal-overlay" id="editSessionModal" style="display: none;" onclick="closeEditModal(event)">
                <div class="modal-content" onclick="event.stopPropagation()">
                    <div class="modal-header">
                        <h3 class="modal-title">Edit Attendance Session</h3>
                        <button class="modal-close" onclick="closeEditModal()">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M18 6 6 18"/>
                                <path d="M6 6l12 12"/>
                            </svg>
                        </button>
                    </div>
                    
                    <div class="modal-body" id="editModalContent">
                        <!-- Modal content will be populated by JavaScript -->
                    </div>
                    
                    <div class="modal-footer">
                        <button class="btn btn-outline" onclick="closeEditModal()">Cancel</button>
                        <button class="btn btn-primary" onclick="saveSessionChanges()">Save Changes</button>
                    </div>
                </div>
            </div>
        </div>
        
        <button class="fab" onclick="startNewUpload()" id="newUploadFab" style="display: none;">
            ➕
        </button>
    </div>

    <script>
        let currentJobId = null;
        let pollInterval = null;
        let currentResults = null;
        
        // File upload handling
        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                uploadVideo(file);
            }
        }
        
        // Drag and drop handling
        const uploadCard = document.getElementById('uploadCard');
        
        uploadCard.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadCard.classList.add('dragover');
        });
        
        uploadCard.addEventListener('dragleave', () => {
            uploadCard.classList.remove('dragover');
        });
        
        uploadCard.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadCard.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                uploadVideo(files[0]);
            }
        });
        
        async function uploadVideo(file) {
            if (!file.type.startsWith('video/')) {
                showError('Please select a video file');
                return;
            }
            
            const formData = new FormData();
            formData.append('video', file);
            
            // Show progress
            document.getElementById('uploadCard').style.display = 'none';
            document.getElementById('progressCard').style.display = 'block';
            document.getElementById('resultsCard').style.display = 'none';
            
            try {
                const response = await fetch('/upload-video', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error('Upload failed');
                }
                
                const result = await response.json();
                currentJobId = result.job_id;
                
                startPolling();
                
            } catch (error) {
                console.error('Upload error:', error);
                showError('Failed to upload video. Please try again.');
                resetToUpload();
            }
        }
        
        function startPolling() {
            if (pollInterval) {
                clearInterval(pollInterval);
            }
            
            pollInterval = setInterval(async () => {
                try {
                    const response = await fetch(`/job-status/${currentJobId}`);
                    const status = await response.json();
                    
                    updateProgress(status);
                    
                    if (status.status === 'completed') {
                        clearInterval(pollInterval);
                        currentResults = status.result;
                        showResults(status.result);
                    } else if (status.status === 'error') {
                        clearInterval(pollInterval);
                        showError(status.error || 'Processing failed');
                        resetToUpload();
                    }
                } catch (error) {
                    console.error('Polling error:', error);
                }
            }, 500);
        }
        
        function updateProgress(status) {
            const progressFill = document.getElementById('progressFill');
            const progressText = document.getElementById('progressText');
            
            const percentage = Math.round(status.progress * 100);
            progressFill.style.width = percentage + '%';
            
            let statusIcon = status.status === 'processing' ? '⚡' : '📊';
            progressText.textContent = `${statusIcon} ${status.message} (${percentage}%)`;
        }
        
        function showResults(result) {
            document.getElementById('progressCard').style.display = 'none';
            document.getElementById('resultsCard').style.display = 'block';
            document.getElementById('newUploadFab').style.display = 'block';
            
            // Update stats
            const statsGrid = document.getElementById('statsGrid');
            statsGrid.innerHTML = `
                <div class="stat-card present">
                    <div class="stat-number">${result.present_count}</div>
                    <div class="stat-label">Present</div>
                </div>
                <div class="stat-card absent">
                    <div class="stat-number">${result.absent_count}</div>
                    <div class="stat-label">Absent</div>
                </div>
                <div class="stat-card uncertain">
                    <div class="stat-number">${result.uncertain_count}</div>
                    <div class="stat-label">Uncertain</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">${Math.round(result.attendance_rate)}%</div>
                    <div class="stat-label">Attendance</div>
                </div>
            `;
            
            // Show student sections
            const studentSections = document.getElementById('studentSections');
            studentSections.innerHTML = '';
            
            // Present students
            if (result.present_students.length > 0) {
                studentSections.innerHTML += createStudentSection(
                    'present', 
                    `✅ Present (${result.present_students.length})`,
                    result.present_students.map(s => ({
                        name: s.name,
                        details: `${Math.round(s.confidence * 100)}% confidence • ${s.detections} detections`,
                        showButton: false
                    }))
                );
            }
            
            // Uncertain students
            if (result.uncertain_students.length > 0) {
                studentSections.innerHTML += createStudentSection(
                    'uncertain',
                    `⚠️ Uncertain (${result.uncertain_students.length})`,
                    result.uncertain_students.map(s => ({
                        name: s.name,
                        details: `${Math.round(s.confidence * 100)}% confidence • ${s.detections} detections`,
                        showButton: true
                    }))
                );
            }
            
            // Absent students
            if (result.absent_students.length > 0) {
                studentSections.innerHTML += createStudentSection(
                    'absent',
                    `❌ Absent (${result.absent_students.length})`,
                    result.absent_students.map(name => ({
                        name: name,
                        details: 'Not detected in video',
                        showButton: true
                    }))
                );
            }
        }
        
        function createStudentSection(type, title, students) {
            return `
                <div class="section-card">
                    <div class="section-header ${type}">${title}</div>
                    ${students.map(student => `
                        <div class="student-item">
                            <div class="student-info">
                                <div class="student-name">${student.name}</div>
                                <div class="student-details">${student.details}</div>
                            </div>
                            ${student.showButton ? (type === 'uncertain' ? `
                                <div class="dual-buttons">
                                    <button class="correction-btn mark-present-btn" onclick="markStudent('${student.name}', '${type}', 'present')">
                                        ✓ Present
                                    </button>
                                    <button class="correction-btn mark-absent-btn" onclick="markStudent('${student.name}', '${type}', 'absent')">
                                        ✗ Absent
                                    </button>
                                </div>
                            ` : `
                                <button class="correction-btn mark-present-btn" onclick="markStudent('${student.name}', '${type}', 'present')">
                                    Mark Present
                                </button>
                            `) : ''}
                        </div>
                    `).join('')}
                </div>
            `;
        }
        
        async function markStudent(studentName, originalStatus, newStatus) {
            const button = event.target;
            
            // Disable button and show processing state
            button.disabled = true;
            button.classList.add('processing');
            const originalText = button.textContent;
            button.textContent = 'Processing...';
            
            try {
                const response = await fetch('/correct-attendance', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        session_id: currentJobId,
                        student_name: studentName,
                        original_status: originalStatus,
                        corrected_status: newStatus,
                        reason: 'manual_correction'
                    })
                });
                
                if (response.ok) {
                    console.log('Response is OK (200), processing success...');
                    
                    let data;
                    try {
                        data = await response.json();
                        console.log('Correction response data:', data);
                    } catch (jsonError) {
                        console.error('JSON parse error (but response was OK):', jsonError);
                        data = { status: 'success', message: 'Correction applied successfully' };
                    }
                    
                    // Update UI to show correction
                    button.classList.remove('processing');
                    button.textContent = newStatus === 'present' ? 'Marked Present ✓' : 'Marked Absent ✗';
                    button.classList.add('corrected');
                    
                    // If this is a dual-button setup, disable both buttons
                    if (button.parentElement.classList.contains('dual-buttons')) {
                        const siblingButton = button.parentElement.querySelector('.correction-btn:not(.corrected)');
                        if (siblingButton) {
                            siblingButton.disabled = true;
                            siblingButton.classList.add('corrected');
                            siblingButton.textContent = 'Corrected';
                        }
                    }
                    
                    // Update stats
                    updateStatsAfterCorrection(originalStatus, newStatus);
                    
                    console.log('✅ Correction UI updated successfully for:', studentName);
                } else {
                    // Handle HTTP error responses
                    let errorMessage = 'Failed to save correction';
                    try {
                        const errorData = await response.json();
                        errorMessage = errorData.message || errorMessage;
                    } catch (e) {
                        errorMessage = response.statusText || errorMessage;
                    }
                    throw new Error(errorMessage);
                }
            } catch (error) {
                console.error('Correction error:', error);
                
                // Reset button state on error
                button.disabled = false;
                button.classList.remove('processing');
                button.textContent = originalText;
                button.classList.add('corrected-error');
                
                showError('Failed to save correction: ' + error.message);
            }
        }
        
        function updateStatsAfterCorrection(originalStatus, newStatus = 'present') {
            const statsGrid = document.getElementById('statsGrid');
            const statCards = statsGrid.querySelectorAll('.stat-card');
            
            if (statCards.length < 4) return; // Ensure we have all stat cards
            
            // Get current counts
            const presentCard = statCards[0];
            const absentCard = statCards[1];
            const uncertainCard = statCards[2];
            const attendanceCard = statCards[3];
            
            let presentCount = parseInt(presentCard.querySelector('.stat-number').textContent);
            let absentCount = parseInt(absentCard.querySelector('.stat-number').textContent);
            let uncertainCount = parseInt(uncertainCard.querySelector('.stat-number').textContent);
            
            // Decrease original status count
            if (originalStatus === 'absent') {
                absentCount = Math.max(0, absentCount - 1);
                absentCard.querySelector('.stat-number').textContent = absentCount;
            } else if (originalStatus === 'uncertain') {
                uncertainCount = Math.max(0, uncertainCount - 1);
                uncertainCard.querySelector('.stat-number').textContent = uncertainCount;
            }
            
            // Increase new status count
            if (newStatus === 'present') {
                presentCount += 1;
                presentCard.querySelector('.stat-number').textContent = presentCount;
            } else if (newStatus === 'absent') {
                absentCount += 1;
                absentCard.querySelector('.stat-number').textContent = absentCount;
            }
            
            // Update attendance rate
            const totalStudents = currentResults.total_students;
            const newRate = Math.round((presentCount / totalStudents) * 100);
            attendanceCard.querySelector('.stat-number').textContent = newRate + '%';
        }
        
        function showError(message) {
            const errorToast = document.createElement('div');
            errorToast.className = 'error-toast';
            errorToast.textContent = message;
            
            document.body.appendChild(errorToast);
            
            setTimeout(() => {
                errorToast.style.animation = 'slideOut 0.3s ease forwards';
                setTimeout(() => {
                    document.body.removeChild(errorToast);
                }, 300);
            }, 3000);
        }
        
        function startNewUpload() {
            resetToUpload();
        }
        
        function resetToUpload() {
            document.getElementById('uploadCard').style.display = 'block';
            document.getElementById('progressCard').style.display = 'none';
            document.getElementById('resultsCard').style.display = 'none';
            document.getElementById('newUploadFab').style.display = 'none';
            
            // Reset file input
            document.getElementById('fileInput').value = '';
            
            // Clear job tracking
            currentJobId = null;
            currentResults = null;
            if (pollInterval) {
                clearInterval(pollInterval);
                pollInterval = null;
            }
        }
        
        // Load attendance history on page load
        document.addEventListener('DOMContentLoaded', function() {
            loadAttendanceHistory();
        });
        
        async function loadAttendanceHistory() {
            try {
                const historyContent = document.getElementById('historyContent');
                historyContent.innerHTML = '<div class="loading-message">Loading attendance history...</div>';
                
                const response = await fetch('/attendance-history');
                const data = await response.json();
                
                if (data.sessions && data.sessions.length > 0) {
                    historyContent.innerHTML = data.sessions.map(session => createHistorySessionHTML(session)).join('');
                } else {
                    historyContent.innerHTML = '<div class="loading-message">No attendance history found. Process your first video to see history here.</div>';
                }
            } catch (error) {
                console.error('Error loading attendance history:', error);
                document.getElementById('historyContent').innerHTML = '<div class="loading-message">Error loading attendance history.</div>';
            }
        }
        
        function createHistorySessionHTML(session) {
            const date = new Date(session.timestamp).toLocaleString();
            return `
                <div class="history-session" onclick="openEditModal('${session.session_id}')">
                    <div class="flex justify-between items-start mb-3">
                        <div>
                            <div class="font-semibold text-base">${session.filename}</div>
                            <div class="text-sm text-muted-foreground">${date}</div>
                        </div>
                        <div class="flex items-center space-x-2">
                            <span class="badge badge-outline">${session.attendance_rate}%</span>
                        </div>
                    </div>
                    
                    <div class="grid grid-cols-4 gap-3 mb-3">
                        <div class="text-center">
                            <div class="badge badge-success">${session.present_count}</div>
                            <div class="text-xs text-muted-foreground mt-1">Present</div>
                        </div>
                        <div class="text-center">
                            <div class="badge badge-destructive">${session.absent_count}</div>
                            <div class="text-xs text-muted-foreground mt-1">Absent</div>
                        </div>
                        <div class="text-center">
                            <div class="badge badge-warning">${session.uncertain_count}</div>
                            <div class="text-xs text-muted-foreground mt-1">Uncertain</div>
                        </div>
                        <div class="text-center">
                            <div class="badge badge-secondary">${session.corrections_count}</div>
                            <div class="text-xs text-muted-foreground mt-1">Corrections</div>
                        </div>
                    </div>
                    
                    <div class="flex justify-between items-center" onclick="event.stopPropagation()">
                        <div class="text-xs text-muted-foreground">Click to edit attendance</div>
                        <div class="flex space-x-2">
                            <button class="btn btn-ghost btn-sm" onclick="downloadSessionReport('${session.session_id}', 'csv')">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                                    <polyline points="7,10 12,15 17,10"/>
                                    <line x1="12" x2="12" y1="15" y2="3"/>
                                </svg>
                                CSV
                            </button>
                            <button class="btn btn-ghost btn-sm" onclick="downloadSessionReport('${session.session_id}', 'json')">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                                    <polyline points="7,10 12,15 17,10"/>
                                    <line x1="12" x2="12" y1="15" y2="3"/>
                                </svg>
                                JSON
                            </button>
                        </div>
                    </div>
                </div>
            `;
        }
        
        function downloadReport(format) {
            if (!currentJobId) {
                showError('No session available for download');
                return;
            }
            downloadSessionReport(currentJobId, format);
        }
        
        function downloadSessionReport(sessionId, format) {
            try {
                const url = `/download-report/${sessionId}?format=${format}`;
                const link = document.createElement('a');
                link.href = url;
                link.download = `attendance_${sessionId}.${format}`;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            } catch (error) {
                console.error('Error downloading report:', error);
                showError('Failed to download report');
            }
        }
        
        // Show download buttons when results are displayed
        function showDownloadButtons() {
            document.getElementById('downloadCsvBtn').style.display = 'inline-block';
            document.getElementById('downloadJsonBtn').style.display = 'inline-block';
        }
        
        // Update the displayResults function to show download buttons
        const originalDisplayResults = displayResults;
        displayResults = function(result) {
            originalDisplayResults(result);
            showDownloadButtons();
            // Refresh history to show the new session
            setTimeout(loadAttendanceHistory, 1000);
        };
        
        // Tab Management
        function switchTab(tabName) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Remove active class from all tab buttons
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            
            // Show selected tab content
            document.getElementById(`${tabName}-tab`).classList.add('active');
            
            // Add active class to clicked tab button
            event.target.classList.add('active');
            
            // Load content based on tab
            switch(tabName) {
                case 'students':
                    loadAllStudents();
                    loadClassOptions();
                    break;
                case 'classes':
                    loadAllClasses();
                    break;
                case 'analytics':
                    loadAnalytics();
                    loadClassOptions('analyticsClassFilter');
                    break;
                case 'history':
                    loadAttendanceHistory();
                    break;
            }
        }
        
        // Student Management Functions
        async function addNewStudent(event) {
            event.preventDefault();
            
            const formData = new FormData(event.target);
            
            // Validate that all photos are uploaded
            const leftPhoto = document.getElementById('leftPhoto').files[0];
            const rightPhoto = document.getElementById('rightPhoto').files[0];
            const frontPhoto = document.getElementById('frontPhoto').files[0];
            
            if (!leftPhoto || !rightPhoto || !frontPhoto) {
                showError('Please upload all three photos (Left, Right, Front views)');
                return;
            }
            
            try {
                const response = await fetch('/add-student', {
                    method: 'POST',
                    body: formData
                });
                
                if (response.ok) {
                    const result = await response.json();
                    showSuccess(result.message);
                    
                    // Reset form
                    event.target.reset();
                    resetPhotoPreview('leftPreview');
                    resetPhotoPreview('rightPreview');
                    resetPhotoPreview('frontPreview');
                    
                    // Refresh students list
                    setTimeout(loadAllStudents, 1000);
                } else {
                    const error = await response.json();
                    showError(error.detail || 'Failed to add student');
                }
            } catch (error) {
                console.error('Error adding student:', error);
                showError('Failed to add student: ' + error.message);
            }
        }
        
        function previewPhoto(input, previewId) {
            const file = input.files[0];
            const preview = document.getElementById(previewId);
            
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    preview.innerHTML = `
                        <img src="${e.target.result}" class="photo-preview" alt="Preview">
                        <div>${input.name.replace('_', ' ').toUpperCase()}</div>
                        <small>Click to change</small>
                    `;
                    preview.parentElement.classList.add('has-file');
                };
                reader.readAsDataURL(file);
            }
        }
        
        function resetPhotoPreview(previewId) {
            const preview = document.getElementById(previewId);
            const uploadDiv = preview.parentElement;
            const viewName = previewId.replace('Preview', '');
            
            preview.innerHTML = `
                <div>📷</div>
                <div>${viewName.charAt(0).toUpperCase() + viewName.slice(1)} View</div>
                <small>Click to upload</small>
            `;
            uploadDiv.classList.remove('has-file');
        }
        
        async function loadAllStudents() {
            try {
                const response = await fetch('/students');
                const data = await response.json();
                
                const studentsGrid = document.getElementById('studentsGrid');
                
                if (data.students && data.students.length > 0) {
                    studentsGrid.innerHTML = data.students.map(student => `
                        <div class="student-card">
                            <div class="student-photos">
                                <img src="${student.photos.left}" class="student-photo" alt="Left view" onerror="this.src='data:image/svg+xml,<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"60\" height=\"60\" viewBox=\"0 0 60 60\"><rect width=\"60\" height=\"60\" fill=\"%23f0f0f0\"/><text x=\"50%\" y=\"50%\" text-anchor=\"middle\" dy=\".3em\" font-family=\"Arial\" font-size=\"12\" fill=\"%23999\">L</text></svg>'">
                                <img src="${student.photos.right}" class="student-photo" alt="Right view" onerror="this.src='data:image/svg+xml,<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"60\" height=\"60\" viewBox=\"0 0 60 60\"><rect width=\"60\" height=\"60\" fill=\"%23f0f0f0\"/><text x=\"50%\" y=\"50%\" text-anchor=\"middle\" dy=\".3em\" font-family=\"Arial\" font-size=\"12\" fill=\"%23999\">R</text></svg>'">
                                <img src="${student.photos.front}" class="student-photo" alt="Front view" onerror="this.src='data:image/svg+xml,<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"60\" height=\"60\" viewBox=\"0 0 60 60\"><rect width=\"60\" height=\"60\" fill=\"%23f0f0f0\"/><text x=\"50%\" y=\"50%\" text-anchor=\"middle\" dy=\".3em\" font-family=\"Arial\" font-size=\"12\" fill=\"%23999\">F</text></svg>'">
                            </div>
                            
                            <div class="flex justify-between items-start mb-2">
                                <div>
                                    <div class="font-semibold">${student.name}</div>
                                    <div class="text-sm text-muted-foreground">ID: ${student.student_id}</div>
                                </div>
                                <span class="badge badge-outline">${student.class_name}</span>
                            </div>
                            
                            <div class="text-xs text-muted-foreground mb-3">
                                Added: ${new Date(student.created_at).toLocaleDateString()}
                            </div>
                            
                            <div class="flex space-x-2">
                                <button class="btn btn-outline btn-sm flex-1" onclick="editStudent('${student.student_id}')">
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                        <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                                        <path d="M18.5 2.5a2.12 2.12 0 0 1 3 3L12 15l-4 1 1-4Z"/>
                                    </svg>
                                    Edit
                                </button>
                                <button class="btn btn-destructive btn-sm" onclick="deleteStudent('${student.student_id}', '${student.name}')">
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                        <path d="M3 6h18"/>
                                        <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/>
                                        <path d="M8 6V4c0-1 1-2 2-2h4c-1 0-2 1-2 2v2"/>
                                    </svg>
                                    Delete
                                </button>
                            </div>
                        </div>
                    `).join('');
                } else {
                    studentsGrid.innerHTML = '<div class="loading-message">No students found. Add some students to get started!</div>';
                }
            } catch (error) {
                console.error('Error loading students:', error);
                document.getElementById('studentsGrid').innerHTML = '<div class="loading-message">Error loading students.</div>';
            }
        }
        
        async function deleteStudent(studentId, studentName) {
            if (!confirm(`Are you sure you want to delete ${studentName}? This action cannot be undone.`)) {
                return;
            }
            
            try {
                const response = await fetch(`/student/${studentId}`, {
                    method: 'DELETE'
                });
                
                if (response.ok) {
                    showSuccess(`Student ${studentName} deleted successfully`);
                    loadAllStudents();
                } else {
                    const error = await response.json();
                    showError(error.detail || 'Failed to delete student');
                }
            } catch (error) {
                console.error('Error deleting student:', error);
                showError('Failed to delete student: ' + error.message);
            }
        }
        
        async function loadAllClasses() {
            try {
                const response = await fetch('/classes');
                const data = await response.json();
                
                const classesGrid = document.getElementById('classesGrid');
                
                if (data.classes && data.classes.length > 0) {
                    classesGrid.innerHTML = data.classes.map(cls => `
                        <div class="analytics-card">
                            <div class="analytics-value">${cls.student_count}</div>
                            <div class="analytics-label">${cls.name}</div>
                            <button class="btn btn-outline btn-sm mt-3" onclick="switchToStudentsTab('${cls.name}')">
                                View Students
                            </button>
                        </div>
                    `).join('');
                } else {
                    classesGrid.innerHTML = '<div class="loading-message">No classes found. Add some students to create classes!</div>';
                }
            } catch (error) {
                console.error('Error loading classes:', error);
                document.getElementById('classesGrid').innerHTML = '<div class="loading-message">Error loading classes.</div>';
            }
        }
        
        function switchToStudentsTab(className) {
            switchTab('students');
            document.getElementById('classFilter').value = className;
            filterStudentsByClass();
        }
        
        async function loadClassOptions(selectId = 'classFilter') {
            try {
                const response = await fetch('/classes');
                const data = await response.json();
                
                const select = document.getElementById(selectId);
                const currentValue = select.value;
                
                // Keep "All Classes" option and add class options
                select.innerHTML = '<option value="">All Classes</option>' + 
                    data.classes.map(cls => `<option value="${cls.name}">${cls.name}</option>`).join('');
                
                // Restore previous selection
                select.value = currentValue;
            } catch (error) {
                console.error('Error loading class options:', error);
            }
        }
        
        function filterStudentsByClass() {
            const selectedClass = document.getElementById('classFilter').value;
            loadStudentsByClass(selectedClass);
        }
        
        async function loadStudentsByClass(className) {
            try {
                const url = className ? `/students?class_name=${encodeURIComponent(className)}` : '/students';
                const response = await fetch(url);
                const data = await response.json();
                
                const studentsGrid = document.getElementById('studentsGrid');
                
                if (data.students && data.students.length > 0) {
                    studentsGrid.innerHTML = data.students.map(student => `
                        <div class="student-card">
                            <div class="student-photos">
                                <img src="${student.photos.left}" class="student-photo" alt="Left view" onerror="this.src='data:image/svg+xml,<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"60\" height=\"60\" viewBox=\"0 0 60 60\"><rect width=\"60\" height=\"60\" fill=\"%23f0f0f0\"/><text x=\"50%\" y=\"50%\" text-anchor=\"middle\" dy=\".3em\" font-family=\"Arial\" font-size=\"12\" fill=\"%23999\">L</text></svg>'">
                                <img src="${student.photos.right}" class="student-photo" alt="Right view" onerror="this.src='data:image/svg+xml,<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"60\" height=\"60\" viewBox=\"0 0 60 60\"><rect width=\"60\" height=\"60\" fill=\"%23f0f0f0\"/><text x=\"50%\" y=\"50%\" text-anchor=\"middle\" dy=\".3em\" font-family=\"Arial\" font-size=\"12\" fill=\"%23999\">R</text></svg>'">
                                <img src="${student.photos.front}" class="student-photo" alt="Front view" onerror="this.src='data:image/svg+xml,<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"60\" height=\"60\" viewBox=\"0 0 60 60\"><rect width=\"60\" height=\"60\" fill=\"%23f0f0f0\"/><text x=\"50%\" y=\"50%\" text-anchor=\"middle\" dy=\".3em\" font-family=\"Arial\" font-size=\"12\" fill=\"%23999\">F</text></svg>'">
                            </div>
                            
                            <div class="flex justify-between items-start mb-2">
                                <div>
                                    <div class="font-semibold">${student.name}</div>
                                    <div class="text-sm text-muted-foreground">ID: ${student.student_id}</div>
                                </div>
                                <span class="badge badge-outline">${student.class_name}</span>
                            </div>
                            
                            <div class="text-xs text-muted-foreground mb-3">
                                Added: ${new Date(student.created_at).toLocaleDateString()}
                            </div>
                            
                            <div class="flex space-x-2">
                                <button class="btn btn-outline btn-sm flex-1" onclick="editStudent('${student.student_id}')">
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                        <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                                        <path d="M18.5 2.5a2.12 2.12 0 0 1 3 3L12 15l-4 1 1-4Z"/>
                                    </svg>
                                    Edit
                                </button>
                                <button class="btn btn-destructive btn-sm" onclick="deleteStudent('${student.student_id}', '${student.name}')">
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                        <path d="M3 6h18"/>
                                        <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/>
                                        <path d="M8 6V4c0-1 1-2 2-2h4c-1 0-2 1-2 2v2"/>
                                    </svg>
                                    Delete
                                </button>
                            </div>
                        </div>
                    `).join('');
                } else {
                    const message = className ? `No students found in class "${className}"` : 'No students found';
                    studentsGrid.innerHTML = `<div class="loading-message">${message}</div>`;
                }
            } catch (error) {
                console.error('Error loading students by class:', error);
                document.getElementById('studentsGrid').innerHTML = '<div class="loading-message">Error loading students.</div>';
            }
        }
        
        async function loadAnalytics() {
            const className = document.getElementById('analyticsClassFilter')?.value || '';
            const days = document.getElementById('analyticsPeriod')?.value || 30;
            
            try {
                const url = `/attendance-analytics?days=${days}${className ? `&class_name=${encodeURIComponent(className)}` : ''}`;
                const response = await fetch(url);
                const data = await response.json();
                
                const analyticsContent = document.getElementById('analyticsContent');
                
                if (data.error) {
                    analyticsContent.innerHTML = `<div class="loading-message">Error: ${data.error}</div>`;
                    return;
                }
                
                const summary = data.summary || {};
                const studentRankings = data.student_rankings || [];
                const lowAttendanceStudents = data.low_attendance_students || [];
                
                analyticsContent.innerHTML = `
                    <div class="analytics-grid mb-6">
                        <div class="analytics-card">
                            <div class="analytics-value">${summary.total_sessions || 0}</div>
                            <div class="analytics-label">Total Sessions</div>
                        </div>
                        <div class="analytics-card">
                            <div class="analytics-value">${(summary.overall_attendance_rate || 0).toFixed(1)}%</div>
                            <div class="analytics-label">Overall Attendance</div>
                        </div>
                        <div class="analytics-card">
                            <div class="analytics-value">${summary.total_present || 0}</div>
                            <div class="analytics-label">Total Present</div>
                        </div>
                        <div class="analytics-card">
                            <div class="analytics-value">${summary.total_absent || 0}</div>
                            <div class="analytics-label">Total Absent</div>
                        </div>
                    </div>
                    
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div class="card">
                            <div class="card-header">
                                <div class="card-title">🏆 Top Performers</div>
                                <div class="card-description">Students with highest attendance rates</div>
                            </div>
                            <div class="card-content">
                                ${studentRankings.slice(0, 10).map((student, index) => `
                                    <div class="flex justify-between items-center p-2 ${index % 2 === 0 ? 'bg-muted' : ''} rounded">
                                        <div>
                                            <div class="font-medium">${student[0]}</div>
                                            <div class="text-xs text-muted-foreground">${student[1].present}/${student[1].total_sessions} sessions</div>
                                        </div>
                                        <div class="badge badge-success">${student[1].attendance_rate.toFixed(1)}%</div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                        
                        <div class="card">
                            <div class="card-header">
                                <div class="card-title">⚠️ Low Attendance Alert</div>
                                <div class="card-description">Students needing attention (< 75%)</div>
                            </div>
                            <div class="card-content">
                                ${lowAttendanceStudents.length > 0 ? lowAttendanceStudents.map(student => `
                                    <div class="flex justify-between items-center p-2 bg-red-50 rounded mb-2">
                                        <div>
                                            <div class="font-medium">${student[0]}</div>
                                            <div class="text-xs text-muted-foreground">${student[1].present}/${student[1].total_sessions} sessions</div>
                                        </div>
                                        <div class="badge badge-destructive">${student[1].attendance_rate.toFixed(1)}%</div>
                                    </div>
                                `).join('') : '<div class="text-muted-foreground">All students have good attendance! 🎉</div>'}
                            </div>
                        </div>
                    </div>
                `;
            } catch (error) {
                console.error('Error loading analytics:', error);
                document.getElementById('analyticsContent').innerHTML = '<div class="loading-message">Error loading analytics.</div>';
            }
        }
        
        function editStudent(studentId) {
            // Placeholder for student editing functionality
            showError('Student editing feature coming soon!');
        }
        
        // Modal functionality
        let currentEditSession = null;
        let sessionChanges = {};
        
        async function openEditModal(sessionId) {
            try {
                const response = await fetch(`/session/${sessionId}`);
                if (!response.ok) throw new Error('Failed to fetch session details');
                
                const sessionData = await response.json();
                currentEditSession = sessionData;
                sessionChanges = {};
                
                document.getElementById('editModalContent').innerHTML = createEditModalContent(sessionData);
                document.getElementById('editSessionModal').style.display = 'flex';
                
                // Prevent body scroll
                document.body.style.overflow = 'hidden';
                
            } catch (error) {
                console.error('Error opening edit modal:', error);
                showError('Failed to load session details');
            }
        }
        
        function closeEditModal(event) {
            if (event && event.target !== event.currentTarget) return;
            
            document.getElementById('editSessionModal').style.display = 'none';
            document.body.style.overflow = 'auto';
            currentEditSession = null;
            sessionChanges = {};
        }
        
        function createEditModalContent(sessionData) {
            const date = new Date(sessionData.timestamp).toLocaleString();
            const results = sessionData.results || {};
            const corrections = sessionData.corrections || [];
            
            // Group students by status
            const studentsByStatus = {
                present: [],
                absent: [],
                uncertain: []
            };
            
            Object.entries(results).forEach(([name, data]) => {
                const correctedStatus = corrections.find(c => c.student_name === name)?.corrected_status;
                const status = correctedStatus || data.status;
                
                if (studentsByStatus[status]) {
                    studentsByStatus[status].push({
                        name,
                        originalStatus: data.status,
                        currentStatus: status,
                        confidence: data.confidence,
                        detections: data.detections,
                        wasCorrected: !!correctedStatus
                    });
                }
            });
            
            return `
                <div class="space-y-4">
                    <div class="card">
                        <div class="card-header">
                            <div class="card-title">${sessionData.filename}</div>
                            <div class="card-description">${date}</div>
                        </div>
                        <div class="card-content">
                            <div class="grid grid-cols-4 gap-4">
                                <div class="text-center">
                                    <div class="text-2xl font-bold text-green-600">${studentsByStatus.present.length}</div>
                                    <div class="text-sm text-muted-foreground">Present</div>
                                </div>
                                <div class="text-center">
                                    <div class="text-2xl font-bold text-red-600">${studentsByStatus.absent.length}</div>
                                    <div class="text-sm text-muted-foreground">Absent</div>
                                </div>
                                <div class="text-center">
                                    <div class="text-2xl font-bold text-yellow-600">${studentsByStatus.uncertain.length}</div>
                                    <div class="text-sm text-muted-foreground">Uncertain</div>
                                </div>
                                <div class="text-center">
                                    <div class="text-2xl font-bold text-blue-600">${corrections.length}</div>
                                    <div class="text-sm text-muted-foreground">Corrections</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    ${Object.entries(studentsByStatus).map(([status, students]) => 
                        students.length > 0 ? createEditStudentSection(status, students) : ''
                    ).join('')}
                </div>
            `;
        }
        
        function createEditStudentSection(status, students) {
            const statusConfig = {
                present: { title: '✅ Present Students', class: 'border-green-200 bg-green-50' },
                absent: { title: '❌ Absent Students', class: 'border-red-200 bg-red-50' },
                uncertain: { title: '❓ Uncertain Students', class: 'border-yellow-200 bg-yellow-50' }
            };
            
            const config = statusConfig[status];
            
            return `
                <div class="card ${config.class}">
                    <div class="card-header">
                        <div class="card-title text-sm">${config.title}</div>
                    </div>
                    <div class="card-content">
                        <div class="space-y-2">
                            ${students.map(student => `
                                <div class="flex items-center justify-between p-2 bg-white rounded border">
                                    <div class="flex-1">
                                        <div class="font-medium">${student.name}</div>
                                        <div class="text-xs text-muted-foreground">
                                            Confidence: ${(student.confidence * 100).toFixed(1)}% | 
                                            Detections: ${student.detections}
                                            ${student.wasCorrected ? ' | ✏️ Corrected' : ''}
                                        </div>
                                    </div>
                                    <div class="flex space-x-2">
                                        <button class="btn btn-outline btn-sm ${student.currentStatus === 'present' ? 'btn-primary' : ''}" 
                                                onclick="changeStudentStatus('${student.name}', 'present')">
                                            Present
                                        </button>
                                        <button class="btn btn-outline btn-sm ${student.currentStatus === 'absent' ? 'btn-destructive' : ''}" 
                                                onclick="changeStudentStatus('${student.name}', 'absent')">
                                            Absent
                                        </button>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                </div>
            `;
        }
        
        function changeStudentStatus(studentName, newStatus) {
            sessionChanges[studentName] = newStatus;
            
            // Update the UI to reflect the change
            const modal = document.getElementById('editModalContent');
            modal.innerHTML = createEditModalContent(currentEditSession);
        }
        
        async function saveSessionChanges() {
            if (Object.keys(sessionChanges).length === 0) {
                showError('No changes to save');
                return;
            }
            
            try {
                // Apply each correction
                for (const [studentName, newStatus] of Object.entries(sessionChanges)) {
                    const response = await fetch('/correct-attendance', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            session_id: currentEditSession.session_id,
                            student_name: studentName,
                            original_status: currentEditSession.results[studentName].status,
                            corrected_status: newStatus,
                            reason: 'manual_edit_from_history'
                        })
                    });
                    
                    if (!response.ok) {
                        throw new Error(`Failed to update ${studentName}`);
                    }
                }
                
                // Show success message
                showSuccess(`Successfully updated ${Object.keys(sessionChanges).length} student(s)`);
                
                // Close modal and refresh history
                closeEditModal();
                setTimeout(loadAttendanceHistory, 500);
                
            } catch (error) {
                console.error('Error saving changes:', error);
                showError('Failed to save changes: ' + error.message);
            }
        }
        
        function showSuccess(message) {
            const toast = document.createElement('div');
            toast.className = 'error-toast';
            toast.style.background = 'hsl(var(--success))';
            toast.style.color = 'white';
            toast.textContent = message;
            
            document.body.appendChild(toast);
            
            setTimeout(() => {
                toast.style.animation = 'slideOut 0.3s ease forwards';
                setTimeout(() => {
                    document.body.removeChild(toast);
                }, 300);
            }, 3000);
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.post("/upload-video")
async def upload_video(background_tasks: BackgroundTasks, video: UploadFile = File(...)):
    """Upload and process classroom video"""
    if not video.content_type or not video.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="Please upload a video file")
    
    job_id = str(uuid.uuid4())
    
    try:
        upload_path = f"uploads/{job_id}_{video.filename}"
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        processing_jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "progress": 0.0,
            "message": "Video uploaded successfully. Processing will begin shortly...",
            "created_at": datetime.now(),
            "video_path": upload_path,
            "original_filename": video.filename
        }
        
        background_tasks.add_task(process_video_background, job_id, upload_path, video.filename)
        
        return {"job_id": job_id, "message": "Video uploaded successfully. Processing started."}
        
    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """Get processing status for a job"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    # If job just completed and hasn't been saved to MongoDB yet, save it now
    if job["status"] == "completed" and job.get("result") and not job.get("saved_to_db", False):
        try:
            # Extract original filename from job data
            original_filename = job.get("original_filename", "unknown.mp4")
            
            # Debug: Log the result structure
            logger.info(f"Job result structure: {type(job['result'])}")
            logger.info(f"Job result keys: {job['result'].keys() if isinstance(job['result'], dict) else 'Not a dict'}")
            
            # Convert result format for saving - handle both dict and object formats
            results_dict = {}
            result = job["result"]
            
            # Handle different result structures
            if isinstance(result, dict):
                # Dictionary format
                for category in ["present_students", "uncertain_students", "absent_students"]:
                    if category in result:
                        status = category.split("_")[0]  # present, uncertain, absent
                        for student in result[category]:
                            if isinstance(student, dict):
                                results_dict[student["name"]] = {
                                    'status': status,
                                    'confidence': student.get("confidence", 0.0),
                                    'detection_count': student.get("detections", 0),
                                    'first_detected_time': 0,
                                    'last_detected_time': 0
                                }
                            elif isinstance(student, str):
                                # Handle case where student is just a name string
                                results_dict[student] = {
                                    'status': status,
                                    'confidence': 0.0,
                                    'detection_count': 0,
                                    'first_detected_time': 0,
                                    'last_detected_time': 0
                                }
            else:
                # Object format - try to access attributes
                for attr_name in ["present_students", "uncertain_students", "absent_students"]:
                    if hasattr(result, attr_name):
                        students = getattr(result, attr_name)
                        status = attr_name.split("_")[0]
                        for student in students:
                            if isinstance(student, dict):
                                results_dict[student["name"]] = {
                                    'status': status,
                                    'confidence': student.get("confidence", 0.0),
                                    'detection_count': student.get("detections", 0),
                                    'first_detected_time': 0,
                                    'last_detected_time': 0
                                }
            
            if results_dict:
                await save_attendance_session(job_id, results_dict, original_filename)
                job["saved_to_db"] = True
                logger.info(f"Saved completed job {job_id} to MongoDB with {len(results_dict)} students")
            else:
                logger.warning(f"No valid student data found in job {job_id} result")
            
        except Exception as e:
            logger.error(f"Failed to save job {job_id} to MongoDB: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
    
    return ProcessingStatus(
        job_id=job["job_id"],
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
        result=job.get("result"),
        error=job.get("error")
    )

@app.post("/correct-attendance")
async def correct_attendance(correction: AttendanceCorrection):
    """Apply manual attendance correction"""
    try:
        client = get_mongo_client()
        if not client:
            logger.error("MongoDB client not available")
            return JSONResponse(content={"message": "MongoDB not available, correction saved locally", "status": "warning"})
            
        db = client[DATABASE_NAME]
        
        # First check if the session exists
        session = await db.attendance_sessions.find_one({"session_id": correction.session_id})
        if not session:
            logger.warning(f"Session {correction.session_id} not found in database")
            return JSONResponse(content={"message": "Session not found in database", "status": "warning"})
        
        # Serialize the session to ensure JSON compatibility
        session = serialize_mongo_doc(session)
        
        # Update session with correction
        result = await db.attendance_sessions.update_one(
            {"session_id": correction.session_id},
            {
                "$push": {
                    "corrections": {
                        "student_name": correction.student_name,
                        "original_status": correction.original_status,
                        "corrected_status": correction.corrected_status,
                        "reason": correction.reason,
                        "corrected_by": correction.corrected_by,
                        "timestamp": datetime.now(timezone.utc)
                    }
                }
            }
        )
        
        if result.modified_count > 0:
            logger.info(f"Applied correction for {correction.student_name} in session {correction.session_id}")
            return JSONResponse(content={"message": "Correction applied successfully", "status": "success"})
        else:
            logger.warning(f"Failed to update session {correction.session_id}")
            return JSONResponse(content={"message": "Failed to update session", "status": "error"})
        
    except Exception as e:
        logger.error(f"Error applying correction: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return JSONResponse(content={"message": f"Error: {str(e)}", "status": "error"})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        db = KnownFacesDatabase()
        db_loaded = db.load_embeddings_from_file()
        student_count = len(db.student_profiles) if db_loaded else 0
    except Exception as e:
        db_loaded = False
        student_count = 0
    
    # Test MongoDB connection
    client = get_mongo_client()
    mongo_status = "connected" if client else "disconnected"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "face_database_loaded": db_loaded,
        "known_students": student_count,
        "mongodb_status": mongo_status,
        "active_jobs": len([j for j in processing_jobs.values() if j["status"] == "processing"])
    }

@app.get("/attendance-history")
async def get_attendance_history():
    """Get attendance history from MongoDB"""
    try:
        client = get_mongo_client()
        if not client:
            return {"sessions": [], "message": "MongoDB not available"}
        
        db = client[DATABASE_NAME]
        collection = db.attendance_sessions
        
        # Get last 10 sessions, sorted by timestamp descending
        cursor = collection.find({}).sort("timestamp", -1).limit(10)
        sessions = []
        
        async for session in cursor:
            # Serialize the session data
            session_data = serialize_mongo_doc(session)
            
            # Calculate summary stats
            results = session_data.get("results", {})
            total_students = len(results)
            present_count = sum(1 for r in results.values() if r.get("status") == "present")
            absent_count = sum(1 for r in results.values() if r.get("status") == "absent")
            uncertain_count = sum(1 for r in results.values() if r.get("status") == "uncertain")
            
            # Count corrections
            corrections_count = len(session_data.get("corrections", []))
            
            sessions.append({
                "session_id": session_data["session_id"],
                "filename": session_data["filename"],
                "timestamp": session_data["timestamp"],
                "total_students": total_students,
                "present_count": present_count,
                "absent_count": absent_count,
                "uncertain_count": uncertain_count,
                "corrections_count": corrections_count,
                "attendance_rate": round((present_count / total_students * 100), 1) if total_students > 0 else 0
            })
        
        return {"sessions": sessions}
        
    except Exception as e:
        logger.error(f"Error fetching attendance history: {e}")
        return {"sessions": [], "error": str(e)}

@app.get("/session/{session_id}")
async def get_session_details(session_id: str):
    """Get detailed session data for editing"""
    try:
        client = get_mongo_client()
        if not client:
            raise HTTPException(status_code=503, detail="MongoDB not available")
        
        db = client[DATABASE_NAME]
        collection = db.attendance_sessions
        
        session = await collection.find_one({"session_id": session_id})
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Serialize the session data
        session_data = serialize_mongo_doc(session)
        
        return session_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching session details: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching session: {str(e)}")

@app.post("/add-student")
async def add_student(
    name: str = Form(...),
    class_name: str = Form(...),
    student_id: str = Form(...),
    left_photo: UploadFile = File(...),
    right_photo: UploadFile = File(...),
    front_photo: UploadFile = File(...)
):
    """Add a new student with L/R/F photos"""
    try:
        # Create student directory
        student_dir = os.path.join("known_faces_optimized", name)
        os.makedirs(student_dir, exist_ok=True)
        
        # Save the three photos
        photos = {
            'l': left_photo,
            'r': right_photo,
            'f': front_photo
        }
        
        for position, photo in photos.items():
            if photo and photo.filename:
                file_path = os.path.join(student_dir, f"{position}.jpg")
                with open(file_path, "wb") as buffer:
                    content = await photo.read()
                    buffer.write(content)
        
        # Add to face database
        face_db.add_new_student(name, student_dir)
        
        # Save to MongoDB
        client = get_mongo_client()
        if client:
            db = client[DATABASE_NAME]
            students_collection = db.students
            
            student_doc = {
                "name": name,
                "student_id": student_id,
                "class_name": class_name,
                "photos": {
                    "left": f"{student_dir}/l.jpg",
                    "right": f"{student_dir}/r.jpg",
                    "front": f"{student_dir}/f.jpg"
                },
                "created_at": datetime.now(timezone.utc),
                "active": True
            }
            
            await students_collection.insert_one(student_doc)
        
        return JSONResponse({
            "success": True,
            "message": f"Student {name} added successfully to class {class_name}",
            "student_id": student_id
        })
        
    except Exception as e:
        logger.error(f"Error adding student: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add student: {str(e)}")

@app.get("/students")
async def get_students(class_name: str = None):
    """Get all students or students from a specific class"""
    try:
        client = get_mongo_client()
        if not client:
            raise HTTPException(status_code=503, detail="Database not available")
        
        db = client[DATABASE_NAME]
        students_collection = db.students
        
        query = {"active": True}
        if class_name:
            query["class_name"] = class_name
        
        students = await students_collection.find(query).to_list(length=None)
        
        # Serialize the results
        serialized_students = [serialize_mongo_doc(student) for student in students]
        
        return {"students": serialized_students}
        
    except Exception as e:
        logger.error(f"Error fetching students: {e}")
        return {"students": [], "error": str(e)}

@app.get("/classes")
async def get_classes():
    """Get all available classes"""
    try:
        client = get_mongo_client()
        if not client:
            raise HTTPException(status_code=503, detail="Database not available")
        
        db = client[DATABASE_NAME]
        students_collection = db.students
        
        # Get unique class names
        classes = await students_collection.distinct("class_name", {"active": True})
        
        # Get student count per class
        class_stats = []
        for class_name in classes:
            count = await students_collection.count_documents({
                "class_name": class_name,
                "active": True
            })
            class_stats.append({
                "name": class_name,
                "student_count": count
            })
        
        return {"classes": class_stats}
        
    except Exception as e:
        logger.error(f"Error fetching classes: {e}")
        return {"classes": [], "error": str(e)}

@app.delete("/student/{student_id}")
async def delete_student(student_id: str):
    """Soft delete a student"""
    try:
        client = get_mongo_client()
        if not client:
            raise HTTPException(status_code=503, detail="Database not available")
        
        db = client[DATABASE_NAME]
        students_collection = db.students
        
        result = await students_collection.update_one(
            {"student_id": student_id},
            {"$set": {"active": False, "deleted_at": datetime.now(timezone.utc)}}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Student not found")
        
        return JSONResponse({"success": True, "message": "Student deleted successfully"})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting student: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete student: {str(e)}")

@app.get("/attendance-analytics")
async def get_attendance_analytics(class_name: str = None, days: int = 30):
    """Get attendance analytics and trends"""
    try:
        client = get_mongo_client()
        if not client:
            raise HTTPException(status_code=503, detail="Database not available")
        
        db = client[DATABASE_NAME]
        sessions_collection = db.attendance_sessions
        
        # Date range filter
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        query = {"timestamp": {"$gte": start_date}}
        
        if class_name:
            query["class_name"] = class_name
        
        sessions = await sessions_collection.find(query).to_list(length=None)
        
        # Calculate analytics
        total_sessions = len(sessions)
        total_students = 0
        total_present = 0
        total_absent = 0
        total_uncertain = 0
        
        daily_stats = {}
        student_attendance = {}
        
        for session in sessions:
            date_key = session['timestamp'].strftime('%Y-%m-%d')
            
            if date_key not in daily_stats:
                daily_stats[date_key] = {
                    'sessions': 0,
                    'present': 0,
                    'absent': 0,
                    'uncertain': 0
                }
            
            daily_stats[date_key]['sessions'] += 1
            daily_stats[date_key]['present'] += session.get('present_count', 0)
            daily_stats[date_key]['absent'] += session.get('absent_count', 0)
            daily_stats[date_key]['uncertain'] += session.get('uncertain_count', 0)
            
            total_present += session.get('present_count', 0)
            total_absent += session.get('absent_count', 0)
            total_uncertain += session.get('uncertain_count', 0)
            
            # Track individual student attendance
            results = session.get('results', {})
            corrections = session.get('corrections', [])
            
            for student_name, data in results.items():
                if student_name not in student_attendance:
                    student_attendance[student_name] = {
                        'present': 0,
                        'absent': 0,
                        'uncertain': 0,
                        'total_sessions': 0
                    }
                
                # Check for corrections
                corrected_status = None
                for correction in corrections:
                    if correction['student_name'] == student_name:
                        corrected_status = correction['corrected_status']
                        break
                
                final_status = corrected_status or data['status']
                student_attendance[student_name][final_status] += 1
                student_attendance[student_name]['total_sessions'] += 1
        
        # Calculate attendance rates for each student
        for student_name, stats in student_attendance.items():
            if stats['total_sessions'] > 0:
                stats['attendance_rate'] = (stats['present'] / stats['total_sessions']) * 100
            else:
                stats['attendance_rate'] = 0
        
        # Sort students by attendance rate
        sorted_students = sorted(
            student_attendance.items(),
            key=lambda x: x[1]['attendance_rate'],
            reverse=True
        )
        
        return {
            "summary": {
                "total_sessions": total_sessions,
                "total_present": total_present,
                "total_absent": total_absent,
                "total_uncertain": total_uncertain,
                "overall_attendance_rate": (total_present / (total_present + total_absent + total_uncertain)) * 100 if (total_present + total_absent + total_uncertain) > 0 else 0
            },
            "daily_stats": daily_stats,
            "student_rankings": sorted_students[:20],  # Top 20 students
            "low_attendance_students": [s for s in sorted_students if s[1]['attendance_rate'] < 75][-10:]  # Bottom 10
        }
        
    except Exception as e:
        logger.error(f"Error fetching analytics: {e}")
        return {"error": str(e)}

@app.get("/download-report/{session_id}")
async def download_report(session_id: str, format: str = "csv"):
    """Download attendance report for a specific session"""
    try:
        client = get_mongo_client()
        if not client:
            raise HTTPException(status_code=503, detail="MongoDB not available")
        
        db = client[DATABASE_NAME]
        collection = db.attendance_sessions
        
        session = await collection.find_one({"session_id": session_id})
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = serialize_mongo_doc(session)
        
        if format.lower() == "csv":
            return generate_csv_report(session)
        elif format.lower() == "json":
            return generate_json_report(session)
        else:
            raise HTTPException(status_code=400, detail="Unsupported format. Use 'csv' or 'json'")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

def generate_csv_report(session):
    """Generate CSV report for a session"""
    import io
    import csv
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        "Student Name", "Status", "Confidence", "Detections", 
        "First Seen", "Last Seen", "Corrected", "Correction Reason"
    ])
    
    # Get corrections mapping
    corrections = {c["student_name"]: c for c in session.get("corrections", [])}
    
    # Data rows
    for name, result in session.get("results", {}).items():
        correction = corrections.get(name, {})
        writer.writerow([
            name,
            correction.get("corrected_status", result.get("status", "unknown")),
            result.get("confidence", 0),
            result.get("detections", 0),
            result.get("first_seen", ""),
            result.get("last_seen", ""),
            "Yes" if name in corrections else "No",
            correction.get("reason", "")
        ])
    
    output.seek(0)
    
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=attendance_{session['session_id']}.csv"}
    )

def generate_json_report(session):
    """Generate JSON report for a session"""
    report = {
        "session_info": {
            "session_id": session["session_id"],
            "filename": session["filename"],
            "timestamp": session["timestamp"]
        },
        "summary": {
            "total_students": len(session.get("results", {})),
            "present_count": sum(1 for r in session.get("results", {}).values() if r.get("status") == "present"),
            "absent_count": sum(1 for r in session.get("results", {}).values() if r.get("status") == "absent"),
            "uncertain_count": sum(1 for r in session.get("results", {}).values() if r.get("status") == "uncertain"),
            "corrections_count": len(session.get("corrections", []))
        },
        "students": session.get("results", {}),
        "corrections": session.get("corrections", [])
    }
    
    from fastapi.responses import JSONResponse
    return JSONResponse(
        content=report,
        headers={"Content-Disposition": f"attachment; filename=attendance_{session['session_id']}.json"}
    )

if __name__ == "__main__":
    # Ensure face database is ready
    try:
        db = KnownFacesDatabase()
        if not db.load_embeddings_from_file():
            logger.warning("Face database not found. Please run the database builder first.")
        else:
            logger.info(f"Face database loaded with {len(db.student_profiles)} students")
    except Exception as e:
        logger.error(f"Error loading face database: {e}")
    
    # Run the web application
    uvicorn.run(
        "modern_web_app:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
