import httpx
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import asyncio
from pathlib import Path

class WHOOPDataFetcher:
    def __init__(self, access_token: str, base_url: str = "https://api.prod.whoop.com/developer/v2"):
        self.access_token = access_token
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {access_token}"}
        self.data_dir = Path("../data")
        self.data_dir.mkdir(exist_ok=True)
        
    async def fetch_paginated_data(self, endpoint: str, params: Optional[Dict] = None) -> List[Dict]:
        """Fetch all pages of data from a paginated endpoint"""
        all_records = []
        next_token = None
        page = 1
        
        if params is None:
            params = {}
            
        async with httpx.AsyncClient(timeout=30.0) as client:
            while True:
                current_params = params.copy()
                if next_token:
                    current_params["nextToken"] = next_token
                    
                print(f"  Fetching page {page} from {endpoint}")
                
                try:
                    response = await client.get(
                        f"{self.base_url}{endpoint}",
                        headers=self.headers,
                        params=current_params
                    )
                    
                    if response.status_code == 429:
                        print("  Rate limited, waiting 60 seconds...")
                        await asyncio.sleep(60)
                        continue
                        
                    response.raise_for_status()
                    data = response.json()
                    
                    # Handle both collection format and direct data format
                    if "records" in data:
                        records = data["records"]
                        next_token = data.get("next_token")
                    else:
                        records = [data] if data else []
                        next_token = None
                        
                    all_records.extend(records)
                    
                    print(f"    Retrieved {len(records)} records")
                    
                    if not next_token or not records:
                        break
                        
                    page += 1
                    
                    # Small delay to be respectful to API
                    await asyncio.sleep(0.5)
                    
                except httpx.HTTPStatusError as e:
                    print(f"    HTTP error {e.response.status_code}: {e.response.text}")
                    break
                except Exception as e:
                    print(f"    Error: {e}")
                    break
                    
        return all_records
    
    async def fetch_single_data(self, endpoint: str) -> Optional[Dict]:
        """Fetch single record from non-paginated endpoint"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(
                    f"{self.base_url}{endpoint}",
                    headers=self.headers
                )
                
                if response.status_code == 429:
                    print("  Rate limited, waiting 60 seconds...")
                    await asyncio.sleep(60)
                    return await self.fetch_single_data(endpoint)
                    
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPStatusError as e:
                print(f"    HTTP error {e.response.status_code}: {e.response.text}")
                return None
            except Exception as e:
                print(f"    Error: {e}")
                return None
    
    def save_data(self, data: Dict | List, filename: str, subfolder: str = ""):
        """Save data to JSON file"""
        if subfolder:
            save_dir = self.data_dir / subfolder
            save_dir.mkdir(exist_ok=True)
        else:
            save_dir = self.data_dir
            
        filepath = save_dir / f"{filename}.json"
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
        print(f"    Saved to {filepath}")
    
    async def fetch_user_data(self):
        """Fetch user profile and body measurements"""
        print("Fetching user profile data...")
        
        # Basic profile
        profile = await self.fetch_single_data("/user/profile/basic")
        if profile:
            self.save_data(profile, "user_profile", "user")
            
        # Body measurements  
        body_measurements = await self.fetch_single_data("/user/measurement/body")
        if body_measurements:
            self.save_data(body_measurements, "body_measurements", "user")
    
    async def fetch_cycles_data(self):
        """Fetch all cycles data"""
        print("Fetching cycles data...")
        
        # Get cycles with a reasonable time range (last 2 years)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # 2 years
        
        params = {
            "limit": 25,  # Max allowed
            "start": start_date.isoformat() + "Z",
            "end": end_date.isoformat() + "Z"
        }
        
        cycles = await self.fetch_paginated_data("/cycle", params)
        if cycles:
            self.save_data(cycles, "cycles", "cycles")
            
            # Save individual cycles and fetch associated sleep data
            for cycle in cycles:
                cycle_id = cycle["id"]
                self.save_data(cycle, f"cycle_{cycle_id}", "cycles/individual")
                
                # Get sleep for this cycle
                sleep = await self.fetch_single_data(f"/cycle/{cycle_id}/sleep")
                if sleep:
                    self.save_data(sleep, f"cycle_{cycle_id}_sleep", "cycles/sleep")
                    
                await asyncio.sleep(0.2)  # Be respectful to API
    
    async def fetch_recovery_data(self):
        """Fetch all recovery data"""
        print("Fetching recovery data...")
        
        # Get recovery with a reasonable time range (last 2 years)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        params = {
            "limit": 25,
            "start": start_date.isoformat() + "Z", 
            "end": end_date.isoformat() + "Z"
        }
        
        recoveries = await self.fetch_paginated_data("/recovery", params)
        if recoveries:
            self.save_data(recoveries, "recoveries", "recovery")
    
    async def fetch_sleep_data(self):
        """Fetch all sleep data"""
        print("Fetching sleep data...")
        
        # Get sleep with a reasonable time range (last 2 years)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        params = {
            "limit": 25,
            "start": start_date.isoformat() + "Z",
            "end": end_date.isoformat() + "Z"
        }
        
        sleep_activities = await self.fetch_paginated_data("/activity/sleep", params)
        if sleep_activities:
            self.save_data(sleep_activities, "sleep_activities", "sleep")
            
            # Save individual sleep records
            for sleep in sleep_activities:
                sleep_id = sleep["id"]
                self.save_data(sleep, f"sleep_{sleep_id}", "sleep/individual")
                await asyncio.sleep(0.1)
    
    async def fetch_workout_data(self):
        """Fetch all workout data"""
        print("Fetching workout data...")
        
        # Get workouts with a reasonable time range (last 2 years)
        end_date = datetime.now() 
        start_date = end_date - timedelta(days=730)
        
        params = {
            "limit": 25,
            "start": start_date.isoformat() + "Z",
            "end": end_date.isoformat() + "Z"
        }
        
        workouts = await self.fetch_paginated_data("/activity/workout", params)
        if workouts:
            self.save_data(workouts, "workouts", "workouts")
            
            # Save individual workout records
            for workout in workouts:
                workout_id = workout["id"]
                self.save_data(workout, f"workout_{workout_id}", "workouts/individual")
                await asyncio.sleep(0.1)
    
    async def fetch_all_data(self):
        """Fetch all WHOOP data"""
        print("Starting comprehensive WHOOP data fetch...")
        print(f"Data will be saved to: {self.data_dir.absolute()}")
        
        start_time = datetime.now()
        
        try:
            # Fetch all data types
            await self.fetch_user_data()
            await self.fetch_cycles_data()
            await self.fetch_recovery_data()
            await self.fetch_sleep_data()
            await self.fetch_workout_data()
            
            # Create summary
            end_time = datetime.now()
            duration = end_time - start_time
            
            summary = {
                "fetch_completed_at": end_time.isoformat(),
                "fetch_duration_seconds": duration.total_seconds(),
                "data_types_fetched": [
                    "user_profile",
                    "body_measurements", 
                    "cycles",
                    "recovery",
                    "sleep",
                    "workouts"
                ],
                "api_version": "v2",
                "notes": "Data fetched using WHOOP API v2. v1 API will be deprecated by October 1, 2025."
            }
            
            self.save_data(summary, "fetch_summary", "")
            
            print("Data fetch completed successfully.")
            print(f"Total time: {duration.total_seconds():.1f} seconds")
            print(f"Check the {self.data_dir} folder for all your WHOOP data.")
            
        except Exception as e:
            print(f"Error during data fetch: {e}")
            raise

async def main():
    """Main function to run the data fetcher"""
    # This will be called from the server with the access token
    print("WHOOP Data Fetcher requires an access token.")
    print("Please use the /fetch-data endpoint on the running server.")

if __name__ == "__main__":
    asyncio.run(main()) 