from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
import httpx
import os
from urllib.parse import urlencode
import json

app = FastAPI(title="WHOOP Data Server")

# WHOOP API Configuration
WHOOP_CLIENT_ID = os.getenv("WHOOP_CLIENT_ID")  # Set your Client ID
WHOOP_CLIENT_SECRET = os.getenv("WHOOP_CLIENT_SECRET")  # Set your Client Secret
REDIRECT_URI = "http://localhost:8000/callback"
WHOOP_AUTH_URL = "https://api.prod.whoop.com/oauth/oauth2/auth"
WHOOP_TOKEN_URL = "https://api.prod.whoop.com/oauth/oauth2/token"
WHOOP_API_BASE = "https://api.prod.whoop.com/developer/v1"

# Store tokens (in production, use proper storage)
user_tokens = {}

@app.get("/")
async def home():
    """Home page with login link"""
    html_content = f"""
    <html>
        <head>
            <title>WHOOP Data Server</title>
        </head>
        <body>
            <h1>WHOOP Data Server</h1>
            <p>Connect your WHOOP account to access your data.</p>
            <a href="/login">
                <button style="padding: 10px 20px; font-size: 16px;">
                    Connect to WHOOP
                </button>
            </a>
            <br><br>
            <a href="/profile">View Profile</a> | 
            <a href="/workouts">View Workouts</a> | 
            <a href="/recovery">View Recovery</a> | 
            <a href="/sleep">View Sleep</a>
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
        "scope": "read:profile read:workout read:recovery read:sleep",
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
    
    return HTMLResponse(content="""
    <html>
        <body>
            <h2>✅ Successfully connected to WHOOP!</h2>
            <p>You can now access your WHOOP data.</p>
            <a href="/">Go back to home</a>
        </body>
    </html>
    """)

async def get_whoop_data(endpoint: str):
    """Helper function to make authenticated requests to WHOOP API"""
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
    """Get user profile data"""
    data = await get_whoop_data("/user/profile/basic")
    return data

@app.get("/workouts")
async def get_workouts():
    """Get workout data"""
    data = await get_whoop_data("/activity/workout")
    return data

@app.get("/recovery")
async def get_recovery():
    """Get recovery data"""
    data = await get_whoop_data("/recovery")
    return data

@app.get("/sleep")
async def get_sleep():
    """Get sleep data"""
    data = await get_whoop_data("/activity/sleep")
    return data

@app.get("/tokens")
async def get_tokens():
    """Debug endpoint to view current tokens"""
    if "current_user" not in user_tokens:
        return {"message": "No tokens stored"}
    
    # Don't expose actual tokens in production
    return {
        "has_access_token": bool(user_tokens["current_user"].get("access_token")),
        "has_refresh_token": bool(user_tokens["current_user"].get("refresh_token")),
        "token_type": user_tokens["current_user"].get("token_type")
    }

if __name__ == "__main__":
    import uvicorn
    
    print("   Starting WHOOP FastAPI server...")
    print("   Make sure to set your environment variables:")
    print("   export WHOOP_CLIENT_ID='your_client_id'")
    print("   export WHOOP_CLIENT_SECRET='your_client_secret'")
    print("🌐 Server will be available at: http://localhost:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000) 