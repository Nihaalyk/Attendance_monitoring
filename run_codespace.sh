#!/bin/bash

echo "🎓 Starting Smart Attendance System on GitHub Codespaces..."

# Check if MongoDB URL is set
if [ -z "$MONGODB_URL" ]; then
    echo "⚠️  MongoDB URL not set. Please set it in Codespaces secrets or environment."
    echo "   Go to your repository → Settings → Secrets and variables → Codespaces"
    echo "   Add: MONGODB_URL = your_mongodb_atlas_connection_string"
    echo ""
    echo "   Using default MongoDB URL for now (you'll need to update it)"
fi

# Start the FastAPI server in background
echo "🚀 Starting FastAPI server..."
python modern_web_app.py &
SERVER_PID=$!

# Wait for server to start
echo "⏳ Waiting for server to start..."
sleep 10

# Check if server is running
if curl -s http://localhost:8001/health > /dev/null; then
    echo "✅ Server started successfully!"
else
    echo "❌ Server failed to start. Check logs above."
    exit 1
fi

# Start Cloudflare tunnel
echo "📡 Creating Cloudflare tunnel..."
echo "🌐 This will provide a public URL for your attendance system..."

# Create tunnel and capture the URL
cloudflared tunnel --url http://localhost:8001 > tunnel.log 2>&1 &
TUNNEL_PID=$!

# Wait for tunnel to establish
sleep 5

# Extract and display the tunnel URL
echo ""
echo "🎉 =================================="
echo "🎓 Smart Attendance System is READY!"
echo "🎉 =================================="
echo ""

# Try to extract URL from tunnel log
TUNNEL_URL=$(grep -o 'https://.*\.trycloudflare\.com' tunnel.log | head -1)

if [ ! -z "$TUNNEL_URL" ]; then
    echo "🌐 PUBLIC URL: $TUNNEL_URL"
    echo ""
    echo "📱 Access from any device using the URL above!"
    echo "🎯 Features available:"
    echo "   • Upload classroom videos for attendance"
    echo "   • Add/manage students with photos"
    echo "   • View analytics and reports"
    echo "   • Edit historical attendance"
    echo ""
else
    echo "📡 Cloudflare tunnel is starting..."
    echo "🔍 Check tunnel.log for the public URL"
    echo ""
    echo "💡 You can also access locally at: http://localhost:8001"
    echo ""
fi

echo "📊 System Status:"
echo "   • FastAPI Server: Running (PID: $SERVER_PID)"
echo "   • Cloudflare Tunnel: Running (PID: $TUNNEL_PID)"
echo "   • MongoDB: $(python -c "
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os

async def check_mongo():
    try:
        mongo_url = os.getenv('MONGODB_URL', 'mongodb+srv://nihaaly41:7849@attendence.cfgt2xs.mongodb.net/')
        client = AsyncIOMotorClient(mongo_url)
        await client.admin.command('ping')
        print('Connected ✅')
    except:
        print('Not Connected ❌ (Update MONGODB_URL)')
    finally:
        if 'client' in locals():
            client.close()

asyncio.run(check_mongo())
")"

echo ""
echo "🛑 To stop the system: Ctrl+C or run 'pkill -f python && pkill -f cloudflared'"
echo ""

# Keep script running and show logs
echo "📋 Live Logs (Ctrl+C to stop):"
echo "================================"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down system..."
    kill $SERVER_PID 2>/dev/null
    kill $TUNNEL_PID 2>/dev/null
    exit 0
}

trap cleanup INT TERM

# Follow logs
tail -f tunnel.log &
wait
