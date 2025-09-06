# 🎓 Smart Attendance Monitoring System

A comprehensive, AI-powered university attendance management system with advanced facial recognition, real-time processing, and professional web interface.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## ✨ Features

### 🎯 Core Functionality
- **📹 Video-based Attendance**: Upload classroom videos for automatic attendance tracking
- **🤖 AI Face Recognition**: Advanced facial recognition using InsightFace and MediaPipe
- **👥 Student Management**: Add/edit students with L/R/F photo views for optimal recognition
- **🏛️ Class Management**: Organize students by classes with comprehensive overview
- **📊 Analytics Dashboard**: Real-time insights, trends, and performance metrics
- **📚 Historical Records**: Clickable history with editing capabilities for past sessions

### 🎨 Professional Interface
- **Modern UI/UX**: Professional shadcn-inspired design with responsive layout
- **Mobile-Friendly**: Optimized for phones, tablets, and desktop devices
- **Real-time Updates**: Live progress tracking and smooth animations
- **Interactive Elements**: Clickable history, modal dialogs, and intuitive navigation

### 📈 Advanced Analytics
- **Performance Tracking**: Top performers and low attendance alerts
- **Trend Analysis**: Customizable time periods (7/30/90 days)
- **Class-specific Insights**: Filter analytics by individual classes
- **Export Capabilities**: Download reports in CSV/JSON formats

### 🔧 Technical Excellence
- **MongoDB Integration**: Persistent data storage with MongoDB Atlas
- **Asynchronous Processing**: Non-blocking video processing with real-time updates
- **Professional Error Handling**: Comprehensive error management and user feedback
- **Scalable Architecture**: Built for university-scale deployments

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- MongoDB Atlas account (free tier available)
- Webcam or smartphone for capturing student photos

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Nihaalyk/Attendance_monitoring.git
   cd Attendance_monitoring
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure MongoDB**
   - Create a MongoDB Atlas account at [mongodb.com](https://www.mongodb.com/atlas)
   - Create a new cluster and get your connection string
   - Update the MongoDB URL in `modern_web_app.py` (line 30):
   ```python
   MONGODB_URL = "your_mongodb_connection_string_here"
   ```

5. **Initialize the database**
   ```bash
   python init_database.py
   ```

6. **Run the application**
   ```bash
   python modern_web_app.py
   ```

7. **Access the application**
   - Open your browser and go to `http://localhost:8001`
   - For mobile access, use Cloudflare Tunnel (see deployment section)

## 📖 Usage Guide

### 1. Adding Students
1. Navigate to the **"Manage Students"** tab
2. Fill in student details (Name, ID, Class)
3. Upload three photos: Left view, Right view, and Front view
4. Click **"Add Student"** to register

### 2. Taking Attendance
1. Go to the **"Take Attendance"** tab
2. Upload a classroom video (MP4 format recommended)
3. Wait for AI processing to complete
4. Review results and make manual corrections if needed
5. Download reports as needed

### 3. Viewing Analytics
1. Visit the **"Analytics"** tab
2. Select class and time period filters
3. View performance metrics, top performers, and attendance trends
4. Export data for further analysis

### 4. Managing History
1. Access the **"History"** tab
2. Click on any past session to view details
3. Edit attendance records using the modal interface
4. Download historical reports

## 🏗️ Architecture

### System Components
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Frontend  │    │   FastAPI       │    │   MongoDB       │
│   (HTML/CSS/JS) │◄──►│   Backend       │◄──►│   Atlas         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Face Recognition│
                    │ Engine (AI/ML)  │
                    └─────────────────┘
```

### Key Technologies
- **Backend**: FastAPI with async support
- **Database**: MongoDB Atlas for scalable data storage
- **AI/ML**: InsightFace, MediaPipe, OpenCV
- **Frontend**: Modern HTML5/CSS3/JavaScript with shadcn design
- **Deployment**: Cloudflare Tunnel for secure remote access

## 📁 Project Structure

```
Attendance_monitoring/
├── modern_web_app.py          # Main application server
├── facial_features_extractor.py # AI face recognition engine
├── known_faces_database.py    # Student database management
├── init_database.py          # Database initialization script
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Project configuration
├── known_faces_optimized/   # Student photo database
│   ├── StudentName1/
│   │   ├── L.jpg           # Left view photo
│   │   ├── R.jpg           # Right view photo
│   │   └── F.jpg           # Front view photo
│   └── ...
├── .gitignore              # Git ignore rules
└── README.md              # This file
```

## 🌐 Deployment

### Local Development
```bash
python modern_web_app.py
# Access at http://localhost:8001
```

### Remote Access (Cloudflare Tunnel)
```bash
# Install Cloudflare tunnel
# Visit: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/

# Create tunnel
cloudflared tunnel --url http://localhost:8001
```

### Production Deployment
For production deployment, consider:
- Using a proper WSGI server (Gunicorn, Uvicorn)
- Setting up SSL certificates
- Implementing rate limiting and security measures
- Using environment variables for sensitive configuration

## 🔧 Configuration

### MongoDB Setup
1. Create MongoDB Atlas account
2. Create a new cluster
3. Get connection string
4. Update `MONGODB_URL` in `modern_web_app.py`

### Environment Variables (Optional)
Create a `.env` file for sensitive configuration:
```env
MONGODB_URL=your_mongodb_connection_string
SECRET_KEY=your_secret_key
DEBUG=false
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **InsightFace** for advanced face recognition capabilities
- **FastAPI** for the excellent web framework
- **MongoDB** for reliable data storage
- **shadcn/ui** for design inspiration
- **OpenCV** and **MediaPipe** for computer vision tools

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/Nihaalyk/Attendance_monitoring/issues) page
2. Create a new issue with detailed information
3. Contact the maintainer

## 🔄 Changelog

### v1.0.0 (Current)
- ✅ Complete university attendance management system
- ✅ Advanced facial recognition with L/R/F photo support
- ✅ Professional web interface with shadcn design
- ✅ MongoDB integration with full CRUD operations
- ✅ Real-time analytics and reporting
- ✅ Historical data management with editing capabilities
- ✅ Mobile-responsive design
- ✅ Export functionality (CSV/JSON)

---

**Made with ❤️ for educational institutions worldwide**

*Streamline your attendance management with the power of AI and modern web technologies.*