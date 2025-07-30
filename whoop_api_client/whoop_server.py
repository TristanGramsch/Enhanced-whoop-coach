from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
import httpx
import os
from urllib.parse import urlencode
import json
from dotenv import load_dotenv
from whoop_data_fetcher import WHOOPDataFetcher
import asyncio
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="WHOOP Data Server")

# WHOOP API Configuration
WHOOP_CLIENT_ID = os.getenv("WHOOP_CLIENT_ID")  # Set your Client ID
WHOOP_CLIENT_SECRET = os.getenv("WHOOP_CLIENT_SECRET")  # Set your Client Secret
REDIRECT_URI = "http://localhost:8000/callback"
WHOOP_AUTH_URL = "https://api.prod.whoop.com/oauth/oauth2/auth"
WHOOP_TOKEN_URL = "https://api.prod.whoop.com/oauth/oauth2/token"
WHOOP_API_BASE = "https://api.prod.whoop.com/developer/v2"  # Updated to v2

# Store tokens (in production, use proper storage)
user_tokens = {}
TOKEN_PATH = Path("token.json")

# Load token from file if it exists
if TOKEN_PATH.exists():
    with open(TOKEN_PATH, "r") as f:
        user_tokens["current_user"] = json.load(f)

@app.get("/")
async def home():
    """Home page with login link"""
    html_content = f"""
    <html>
        <head>
            <title>WHOOP Data Server</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .button {{ padding: 12px 24px; font-size: 16px; margin: 10px; text-decoration: none; border-radius: 5px; display: inline-block; }}
                .primary {{ background-color: #ff6b35; color: white; }}
                .secondary {{ background-color: #007bff; color: white; }}
                .success {{ background-color: #28a745; color: white; }}
                .warning {{ background-color: #ffc107; color: black; }}
                h1 {{ color: #333; }}
                .api-note {{ background-color: #e7f3ff; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #007bff; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>WHOOP Data Server (API v2)</h1>
                <a href="/login" class="button primary">Connect to WHOOP</a>
                <h3>Comprehensive Data Export:</h3>
                <a href="/fetch-data" class="button success">Fetch ALL Data to /data folder</a>
                <h3>Debug:</h3>
                <a href="/tokens" class="button warning">View Token Status</a>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/login")
async def login():
    """Redirect to WHOOP OAuth authorization"""
    if not WHOOP_CLIENT_ID:
        raise HTTPException(status_code=500, detail="WHOOP_CLIENT_ID not configured")
    
    params = {
        "response_type": "code",
        "client_id": WHOOP_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        # Updated scopes for v2 API
        "scope": "read:profile read:body_measurement read:workout read:recovery read:sleep read:cycles",
        "state": "random_state_string"  # Use a proper random state in production
    }
    
    auth_url = f"{WHOOP_AUTH_URL}?{urlencode(params)}"
    return RedirectResponse(url=auth_url)

@app.get("/callback")
async def callback(request: Request):
    """Handle OAuth callback from WHOOP"""
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    error = request.query_params.get("error")
    
    if error:
        raise HTTPException(status_code=400, detail=f"OAuth error: {error}")
    
    if not code:
        raise HTTPException(status_code=400, detail="No authorization code received")
    
    # Exchange code for tokens
    token_data = {
        "grant_type": "authorization_code",
        "client_id": WHOOP_CLIENT_ID,
        "client_secret": WHOOP_CLIENT_SECRET,
        "code": code,
        "redirect_uri": REDIRECT_URI
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(WHOOP_TOKEN_URL, data=token_data)
        
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Token exchange failed: {response.text}")
        
        tokens = response.json()
        user_tokens["current_user"] = tokens  # Store tokens (use proper user identification in production)
        # Save token to file
        with open(TOKEN_PATH, "w") as f:
            json.dump(tokens, f)
    
    return HTMLResponse(content="""
    <html>
        <head>
            <title>WHOOP Connection Success</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
                .container { max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .success { color: #28a745; }
                .button { padding: 12px 24px; font-size: 16px; margin: 10px; text-decoration: none; border-radius: 5px; display: inline-block; background-color: #007bff; color: white; }
            </style>
        </head>
        <body>
            <div class="container">
                <h2 class="success">Successfully connected to WHOOP!</h2>
                <p>You can now access your WHOOP data using API v2 endpoints.</p>
                <p>Ready to fetch comprehensive data to your local /data folder!</p>
                <a href="/" class="button">Go back to home</a>
                <a href="/fetch-data" class="button" style="background-color: #28a745;">Fetch All Data Now</a>
            </div>
        </body>
    </html>
    """)

async def get_whoop_data(endpoint: str):
    """Helper function to make authenticated requests to WHOOP API v2"""
    if "current_user" not in user_tokens:
        raise HTTPException(status_code=401, detail="Not authenticated. Please login first.")
    
    access_token = user_tokens["current_user"]["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{WHOOP_API_BASE}{endpoint}", headers=headers)
        
        if response.status_code == 401:
            raise HTTPException(status_code=401, detail="Token expired. Please login again.")
        elif response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"API error: {response.text}")
        
        return response.json()

@app.get("/profile")
async def get_profile():
    """Get user profile data (v2 API)"""
    data = await get_whoop_data("/user/profile/basic")
    return data

@app.get("/body-measurements")
async def get_body_measurements():
    """Get user body measurements (v2 API)"""
    data = await get_whoop_data("/user/measurement/body")
    return data

@app.get("/cycles")
async def get_cycles():
    """Get cycles data (v2 API)"""
    data = await get_whoop_data("/cycle?limit=10")
    return data

@app.get("/workouts")
async def get_workouts():
    """Get workout data (v2 API)"""
    data = await get_whoop_data("/activity/workout?limit=10")
    return data

@app.get("/recovery")
async def get_recovery():
    """Get recovery data (v2 API)"""
    data = await get_whoop_data("/recovery?limit=10")
    return data

@app.get("/sleep")
async def get_sleep():
    """Get sleep data (v2 API)"""
    data = await get_whoop_data("/activity/sleep?limit=10")
    return data

# Global variable to track fetch status
fetch_status = {"running": False, "last_run": None, "status": "idle"}

async def run_data_fetch(access_token: str):
    """Background task to run comprehensive data fetch"""
    global fetch_status
    
    try:
        fetch_status["running"] = True
        fetch_status["status"] = "running"
        
        fetcher = WHOOPDataFetcher(access_token)
        await fetcher.fetch_all_data()
        
        fetch_status["status"] = "completed"
        fetch_status["last_run"] = "success"
        
    except Exception as e:
        fetch_status["status"] = f"error: {str(e)}"
        fetch_status["last_run"] = "error"
        print(f"Data fetch error: {e}")
    finally:
        fetch_status["running"] = False

@app.get("/fetch-data")
async def fetch_comprehensive_data(background_tasks: BackgroundTasks):
    """Fetch all WHOOP data and save to /data folder"""
    global fetch_status
    
    if "current_user" not in user_tokens:
        raise HTTPException(status_code=401, detail="Not authenticated. Please login first.")
    
    if fetch_status["running"]:
        return HTMLResponse(content=f"""
        <html>
            <head>
                <title>Data Fetch In Progress</title>
                <meta http-equiv="refresh" content="5">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                    .container {{ max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
                    .spinner {{ border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 40px; height: 40px; animation: spin 2s linear infinite; margin: 20px auto; }}
                    @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>Data Fetch In Progress...</h2>
                    <div class="spinner"></div>
                    <p>Status: {fetch_status["status"]}</p>
                    <p>This page will auto-refresh every 5 seconds.</p>
                    <p>The comprehensive data fetch may take several minutes depending on your data history.</p>
                    <a href="/">Go back to home</a>
                </div>
            </body>
        </html>
        """)
    
    access_token = user_tokens["current_user"]["access_token"]
    
    # Start background task
    background_tasks.add_task(run_data_fetch, access_token)
    
    return HTMLResponse(content="""
    <html>
        <head>
            <title>Data Fetch Started</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
                .container { max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
                .button { padding: 12px 24px; font-size: 16px; margin: 10px; text-decoration: none; border-radius: 5px; display: inline-block; background-color: #007bff; color: white; }
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Comprehensive Data Fetch Started!</h2>
                <p>Your WHOOP data is being fetched and will be saved to the <code>/data</code> folder.</p>
                <p>This includes:</p>
                <ul>
                    <li>👤 User Profile & Body Measurements</li>
                    <li>🔄 Physiological Cycles (last 2 years)</li>
                    <li>😴 Recovery Data (last 2 years)</li>
                    <li>🛌 Sleep Data (last 2 years)</li>
                    <li>💪 Workout Data (last 2 years)</li>
                </ul>
                <p><strong>⏱️ Expected time:</strong> 3-10 minutes depending on your data history</p>
                <p><strong>📁 Data location:</strong> <code>Enhanced-whoop-coach/data/</code></p>
                
                <a href="/fetch-data" class="button">Check Status</a>
                <a href="/" class="button">Go back to home</a>
            </div>
        </body>
    </html>
    """)

@app.get("/tokens")
async def get_tokens():
    """Debug endpoint to view current tokens"""
    if "current_user" not in user_tokens:
        return {"message": "No tokens stored"}
    
    # Don't expose actual tokens in production
    return {
        "has_access_token": bool(user_tokens["current_user"].get("access_token")),
        "has_refresh_token": bool(user_tokens["current_user"].get("refresh_token")),
        "token_type": user_tokens["current_user"].get("token_type"),
        "api_version": "v2",
        "scopes": "read:profile read:body_measurement read:workout read:recovery read:sleep read:cycles"
    }

if __name__ == "__main__":
    import uvicorn
    
    print("   🏃‍♀️ Starting WHOOP FastAPI server (API v2)...")
    print("   Make sure to set your environment variables:")
    print("   export WHOOP_CLIENT_ID='your_client_id'")
    print("   export WHOOP_CLIENT_SECRET='your_client_secret'")
    print("🌐 Server will be available at: http://localhost:8000")
    print("📊 Ready to fetch comprehensive WHOOP data to /data folder!")
    
    uvicorn.run(app, host="0.0.0.0", port=8000) 