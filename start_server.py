"""
Startup script that ensures imports work by adding project root to sys.path
Run the API server with proper Python path configuration
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Now import and run the API
if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("  Lane Detection API Server - 2026 Edition")
    print("=" * 60)
    print()
    print("Starting server...")
    print("API will be available at: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print()
    print("Press CTRL+C to stop")
    print("=" * 60)
    print()
    
    # Import the app after adding to path
    from web.api import app
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
