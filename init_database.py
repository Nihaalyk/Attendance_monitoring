#!/usr/bin/env python3
"""
Database initialization script for the Attendance Monitoring System
Creates collections and indexes in MongoDB Atlas
"""

import asyncio
import logging
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorClient

# MongoDB configuration
MONGODB_URL = "mongodb+srv://nihaaly41:7849@attendence.cfgt2xs.mongodb.net/"
DATABASE_NAME = "attendance_system"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def init_database():
    """Initialize the MongoDB database with proper collections and indexes"""
    try:
        # Connect to MongoDB
        client = AsyncIOMotorClient(MONGODB_URL)
        db = client[DATABASE_NAME]
        
        # Test connection
        await client.admin.command('ping')
        logger.info(f"Connected to MongoDB Atlas successfully")
        
        # Create collections if they don't exist
        collections = await db.list_collection_names()
        
        # 1. Attendance Sessions Collection
        if "attendance_sessions" not in collections:
            await db.create_collection("attendance_sessions")
            logger.info("Created attendance_sessions collection")
        
        # Create indexes for attendance_sessions
        await db.attendance_sessions.create_index("session_id", unique=True)
        await db.attendance_sessions.create_index("timestamp")
        await db.attendance_sessions.create_index("filename")
        logger.info("Created indexes for attendance_sessions")
        
        # 2. Student Profiles Collection (for future use)
        if "student_profiles" not in collections:
            await db.create_collection("student_profiles")
            logger.info("Created student_profiles collection")
        
        # Create indexes for student_profiles
        await db.student_profiles.create_index("student_id", unique=True)
        await db.student_profiles.create_index("name")
        await db.student_profiles.create_index("class_id")
        logger.info("Created indexes for student_profiles")
        
        # 3. Classes Collection (for future use)
        if "classes" not in collections:
            await db.create_collection("classes")
            logger.info("Created classes collection")
        
        # Create indexes for classes
        await db.classes.create_index("class_id", unique=True)
        await db.classes.create_index("class_name")
        logger.info("Created indexes for classes")
        
        # Insert sample data structure documentation
        sample_session = {
            "_id": "sample_session_structure",
            "session_id": "example-session-id",
            "filename": "example_video.mp4",
            "timestamp": datetime.now(timezone.utc),
            "results": {
                "student_name": {
                    "status": "present|absent|uncertain",
                    "confidence": 0.85,
                    "detections": 5,
                    "first_seen": 0,
                    "last_seen": 100
                }
            },
            "corrections": [
                {
                    "student_name": "example_student",
                    "original_status": "uncertain",
                    "corrected_status": "present",
                    "reason": "manual_correction",
                    "corrected_by": "teacher",
                    "timestamp": datetime.now(timezone.utc)
                }
            ],
            "metadata": {
                "total_students": 22,
                "present_count": 15,
                "absent_count": 5,
                "uncertain_count": 2,
                "processing_time": 45.2
            }
        }
        
        # Check if sample exists and update/insert
        existing = await db.attendance_sessions.find_one({"_id": "sample_session_structure"})
        if existing:
            await db.attendance_sessions.replace_one({"_id": "sample_session_structure"}, sample_session)
        else:
            await db.attendance_sessions.insert_one(sample_session)
        
        logger.info("Updated sample session structure")
        
        # Display collection stats
        session_count = await db.attendance_sessions.count_documents({})
        logger.info(f"Database initialization complete!")
        logger.info(f"Collections created: attendance_sessions, student_profiles, classes")
        logger.info(f"Current attendance sessions: {session_count}")
        
        # Close connection
        client.close()
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(init_database())
