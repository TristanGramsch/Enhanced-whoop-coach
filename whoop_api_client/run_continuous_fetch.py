#!/usr/bin/env python3
"""
Continuous WHOOP Data Fetcher
Runs the WHOOP API client continuously to fetch new data at regular intervals.
"""

import asyncio
import json
import sys
from pathlib import Path
from whoop_data_fetcher import WHOOPDataFetcher

def load_token():
    """Load access token from token.json"""
    token_file = Path("token.json")
    if not token_file.exists():
        print("❌ No token.json found. Please run the server first to authenticate.")
        sys.exit(1)
    
    with open(token_file, 'r') as f:
        token_data = json.load(f)
    
    return token_data.get("access_token")

async def main():
    """Main function to run continuous fetching"""
    print("🚀 Starting continuous WHOOP data fetching...")
    
    # Load access token
    access_token = load_token()
    if not access_token:
        print("❌ No access token found in token.json")
        sys.exit(1)
    
    # Create fetcher instance
    fetcher = WHOOPDataFetcher(access_token)
    
    print("📊 Initial data fetch...")
    try:
        # Do an initial fetch to get any new data
        await fetcher.fetch_all_data()
        
        # Start continuous fetching every 30 minutes
        print("\n🔄 Starting continuous fetching mode...")
        await fetcher.continuous_fetch(interval_minutes=30)
        
    except KeyboardInterrupt:
        print("\n🛑 Continuous fetching stopped by user")
    except Exception as e:
        print(f"\n❌ Error in continuous fetching: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())