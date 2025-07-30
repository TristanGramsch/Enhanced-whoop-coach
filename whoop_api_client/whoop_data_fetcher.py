import httpx
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import asyncio
from pathlib import Path
import time

class WHOOPDataFetcher:
    def __init__(self, access_token: str, base_url: str = "https://api.prod.whoop.com/developer/v2"):
        self.access_token = access_token
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {access_token}"}
        self.data_dir = Path("../data")
        self.data_dir.mkdir(exist_ok=True)
        self.last_fetch_times = {}  # Track last fetch times for each data type
        self.fetch_intervals = {
            "user_profile": 24 * 60 * 60,  # 24 hours
            "cycles": 6 * 60 * 60,         # 6 hours  
            "recovery": 6 * 60 * 60,       # 6 hours
            "sleep": 6 * 60 * 60,          # 6 hours
            "workouts": 2 * 60 * 60        # 2 hours
        }
        self.load_last_fetch_times()
        
    def load_last_fetch_times(self):
        """Load last fetch times from file"""
        fetch_times_file = self.data_dir / "last_fetch_times.json"
        if fetch_times_file.exists():
            with open(fetch_times_file, 'r') as f:
                self.last_fetch_times = json.load(f)
                # Convert strings back to timestamps
                for key in self.last_fetch_times:
                    if isinstance(self.last_fetch_times[key], str):
                        self.last_fetch_times[key] = datetime.fromisoformat(self.last_fetch_times[key]).timestamp()
    
    def save_last_fetch_times(self):
        """Save last fetch times to file"""
        fetch_times_file = self.data_dir / "last_fetch_times.json"
        # Convert timestamps to ISO strings for JSON serialization
        times_to_save = {}
        for key, timestamp in self.last_fetch_times.items():
            if isinstance(timestamp, (int, float)):
                times_to_save[key] = datetime.fromtimestamp(timestamp).isoformat()
            else:
                times_to_save[key] = timestamp
        
        with open(fetch_times_file, 'w') as f:
            json.dump(times_to_save, f, indent=2)
    
    def should_fetch(self, data_type: str) -> bool:
        """Check if data type should be fetched based on intervals"""
        current_time = time.time()
        last_fetch = self.last_fetch_times.get(data_type, 0)
        interval = self.fetch_intervals.get(data_type, 3600)  # Default 1 hour
        
        return (current_time - last_fetch) >= interval
    
    def get_latest_data_timestamp(self, data_type: str) -> Optional[datetime]:
        """Get the timestamp of the latest data for incremental fetches"""
        try:
            if data_type == "cycles":
                cycles_file = self.data_dir / "cycles" / "cycles.json"
                if cycles_file.exists():
                    with open(cycles_file, 'r') as f:
                        cycles = json.load(f)
                        if cycles:
                            # Get the most recent cycle's updated_at timestamp
                            latest = max(cycles, key=lambda x: x.get('updated_at', ''))
                            return datetime.fromisoformat(latest['updated_at'].replace('Z', '+00:00'))
            
            elif data_type == "recovery":
                recovery_file = self.data_dir / "recovery" / "recoveries.json"
                if recovery_file.exists():
                    with open(recovery_file, 'r') as f:
                        recoveries = json.load(f)
                        if recoveries:
                            latest = max(recoveries, key=lambda x: x.get('updated_at', ''))
                            return datetime.fromisoformat(latest['updated_at'].replace('Z', '+00:00'))
            
            elif data_type == "sleep":
                sleep_file = self.data_dir / "sleep" / "sleep_activities.json"
                if sleep_file.exists():
                    with open(sleep_file, 'r') as f:
                        sleep_activities = json.load(f)
                        if sleep_activities:
                            latest = max(sleep_activities, key=lambda x: x.get('updated_at', ''))
                            return datetime.fromisoformat(latest['updated_at'].replace('Z', '+00:00'))
            
            elif data_type == "workouts":
                workouts_file = self.data_dir / "workouts" / "workouts.json"
                if workouts_file.exists():
                    with open(workouts_file, 'r') as f:
                        workouts = json.load(f)
                        if workouts:
                            latest = max(workouts, key=lambda x: x.get('updated_at', ''))
                            return datetime.fromisoformat(latest['updated_at'].replace('Z', '+00:00'))
            
        except Exception as e:
            print(f"Error getting latest timestamp for {data_type}: {e}")
        
        return None

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
                    
                print(f"  📄 Fetching page {page} from {endpoint}")
                
                try:
                    response = await client.get(
                        f"{self.base_url}{endpoint}",
                        headers=self.headers,
                        params=current_params
                    )
                    
                    if response.status_code == 429:
                        print("  ⏸️  Rate limited, waiting 60 seconds...")
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
                    
                    print(f"    ✅ Got {len(records)} records")
                    
                    if not next_token or not records:
                        break
                        
                    page += 1
                    
                    # Small delay to be respectful to API
                    await asyncio.sleep(0.5)
                    
                except httpx.HTTPStatusError as e:
                    print(f"    ❌ HTTP error {e.response.status_code}: {e.response.text}")
                    break
                except Exception as e:
                    print(f"    ❌ Error: {e}")
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
                    print("  ⏸️  Rate limited, waiting 60 seconds...")
                    await asyncio.sleep(60)
                    return await self.fetch_single_data(endpoint)
                    
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPStatusError as e:
                print(f"    ❌ HTTP error {e.response.status_code}: {e.response.text}")
                return None
            except Exception as e:
                print(f"    ❌ Error: {e}")
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
            
        print(f"    💾 Saved to {filepath}")
    
    async def fetch_user_data(self):
        """Fetch user profile and body measurements"""
        if not self.should_fetch("user_profile"):
            print("⏭️  Skipping user profile - fetched recently")
            return
            
        print("🔵 Fetching User Profile Data...")
        
        # Basic profile
        profile = await self.fetch_single_data("/user/profile/basic")
        if profile:
            self.save_data(profile, "user_profile", "user")
            
        # Body measurements  
        body_measurements = await self.fetch_single_data("/user/measurement/body")
        if body_measurements:
            self.save_data(body_measurements, "body_measurements", "user")
        
        self.last_fetch_times["user_profile"] = time.time()
        self.save_last_fetch_times()
    
    async def fetch_cycles_data(self):
        """Fetch cycles data with incremental updates"""
        if not self.should_fetch("cycles"):
            print("⏭️  Skipping cycles - fetched recently")
            return
            
        print("🔵 Fetching Cycles Data...")
        
        # For incremental fetching, start from the latest data we have
        latest_timestamp = self.get_latest_data_timestamp("cycles")
        
        # Get cycles with a reasonable time range
        end_date = datetime.now()
        if latest_timestamp:
            # Fetch from 1 day before latest to catch any updates
            start_date = latest_timestamp - timedelta(days=1)
            print(f"  📅 Incremental fetch from {start_date.isoformat()}")
        else:
            # Full fetch for the last 2 years
            start_date = end_date - timedelta(days=730)
            print(f"  📅 Full fetch from {start_date.isoformat()}")
        
        params = {
            "limit": 25,
            "start": start_date.isoformat() + "Z",
            "end": end_date.isoformat() + "Z"
        }
        
        cycles = await self.fetch_paginated_data("/cycle", params)
        if cycles:
            # Merge with existing data to avoid duplicates
            existing_cycles = self.load_existing_data("cycles", "cycles")
            merged_cycles = self.merge_data_by_id(existing_cycles, cycles)
            
            self.save_data(merged_cycles, "cycles", "cycles")
            
            # Also save individual cycle records
            for cycle in cycles:
                cycle_id = cycle["id"]
                self.save_data(cycle, f"cycle_{cycle_id}", "cycles/individual")
                await asyncio.sleep(0.1)
        
        self.last_fetch_times["cycles"] = time.time()
        self.save_last_fetch_times()

    def load_existing_data(self, filename: str, subdirectory: str = "") -> List[Dict]:
        """Load existing data from file"""
        try:
            if subdirectory:
                file_path = self.data_dir / subdirectory / f"{filename}.json"
            else:
                file_path = self.data_dir / f"{filename}.json"
            
            if file_path.exists():
                with open(file_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading existing data from {file_path}: {e}")
        
        return []
    
    def merge_data_by_id(self, existing_data: List[Dict], new_data: List[Dict]) -> List[Dict]:
        """Merge new data with existing data, avoiding duplicates by ID"""
        existing_ids = {item.get('id') for item in existing_data if 'id' in item}
        
        # Add new items that don't exist
        merged = existing_data.copy()
        for item in new_data:
            if item.get('id') not in existing_ids:
                merged.append(item)
                print(f"  ➕ Added new item: {item.get('id')}")
            else:
                # Update existing item if it's newer
                for i, existing_item in enumerate(merged):
                    if existing_item.get('id') == item.get('id'):
                        if item.get('updated_at', '') > existing_item.get('updated_at', ''):
                            merged[i] = item
                            print(f"  🔄 Updated item: {item.get('id')}")
                        break
        
        return merged
    
    def merge_data_by_cycle_id(self, existing_data: List[Dict], new_data: List[Dict]) -> List[Dict]:
        """Merge new data with existing data, avoiding duplicates by cycle_id"""
        existing_cycle_ids = {item.get('cycle_id') for item in existing_data if 'cycle_id' in item}
        
        # Add new items that don't exist
        merged = existing_data.copy()
        for item in new_data:
            if item.get('cycle_id') not in existing_cycle_ids:
                merged.append(item)
                print(f"  ➕ Added new item for cycle: {item.get('cycle_id')}")
            else:
                # Update existing item if it's newer
                for i, existing_item in enumerate(merged):
                    if existing_item.get('cycle_id') == item.get('cycle_id'):
                        if item.get('updated_at', '') > existing_item.get('updated_at', ''):
                            merged[i] = item
                            print(f"  🔄 Updated item for cycle: {item.get('cycle_id')}")
                        break
        
        return merged

    async def fetch_recovery_data(self):
        """Fetch recovery data with incremental updates"""
        if not self.should_fetch("recovery"):
            print("⏭️  Skipping recovery - fetched recently")
            return
            
        print("🟡 Fetching Recovery Data...")
        
        # For incremental fetching, start from the latest data we have
        latest_timestamp = self.get_latest_data_timestamp("recovery")
        
        end_date = datetime.now()
        if latest_timestamp:
            start_date = latest_timestamp - timedelta(days=1)
            print(f"  📅 Incremental fetch from {start_date.isoformat()}")
        else:
            start_date = end_date - timedelta(days=730)
            print(f"  📅 Full fetch from {start_date.isoformat()}")
        
        params = {
            "limit": 25,
            "start": start_date.isoformat() + "Z", 
            "end": end_date.isoformat() + "Z"
        }
        
        recoveries = await self.fetch_paginated_data("/recovery", params)
        if recoveries:
            existing_recoveries = self.load_existing_data("recoveries", "recovery")
            merged_recoveries = self.merge_data_by_cycle_id(existing_recoveries, recoveries)
            
            self.save_data(merged_recoveries, "recoveries", "recovery")
        
        self.last_fetch_times["recovery"] = time.time()
        self.save_last_fetch_times()
    
    async def fetch_sleep_data(self):
        """Fetch sleep data with incremental updates"""
        if not self.should_fetch("sleep"):
            print("⏭️  Skipping sleep - fetched recently")
            return
            
        print("🟣 Fetching Sleep Data...")
        
        # For incremental fetching, start from the latest data we have
        latest_timestamp = self.get_latest_data_timestamp("sleep")
        
        end_date = datetime.now()
        if latest_timestamp:
            start_date = latest_timestamp - timedelta(days=1)
            print(f"  📅 Incremental fetch from {start_date.isoformat()}")
        else:
            start_date = end_date - timedelta(days=730)
            print(f"  📅 Full fetch from {start_date.isoformat()}")
        
        params = {
            "limit": 25,
            "start": start_date.isoformat() + "Z",
            "end": end_date.isoformat() + "Z"
        }
        
        sleep_activities = await self.fetch_paginated_data("/activity/sleep", params)
        if sleep_activities:
            existing_sleep = self.load_existing_data("sleep_activities", "sleep")
            merged_sleep = self.merge_data_by_id(existing_sleep, sleep_activities)
            
            self.save_data(merged_sleep, "sleep_activities", "sleep")
            
            # Save individual sleep records
            for sleep in sleep_activities:
                sleep_id = sleep["id"]
                self.save_data(sleep, f"sleep_{sleep_id}", "sleep/individual")
                await asyncio.sleep(0.1)
        
        self.last_fetch_times["sleep"] = time.time()
        self.save_last_fetch_times()
    
    async def fetch_workout_data(self):
        """Fetch workout data with incremental updates"""
        if not self.should_fetch("workouts"):
            print("⏭️  Skipping workouts - fetched recently")
            return
            
        print("🔴 Fetching Workout Data...")
        
        # For incremental fetching, start from the latest data we have
        latest_timestamp = self.get_latest_data_timestamp("workouts")
        
        end_date = datetime.now() 
        if latest_timestamp:
            start_date = latest_timestamp - timedelta(days=1)
            print(f"  📅 Incremental fetch from {start_date.isoformat()}")
        else:
            start_date = end_date - timedelta(days=730)
            print(f"  📅 Full fetch from {start_date.isoformat()}")
        
        params = {
            "limit": 25,
            "start": start_date.isoformat() + "Z",
            "end": end_date.isoformat() + "Z"
        }
        
        workouts = await self.fetch_paginated_data("/activity/workout", params)
        if workouts:
            existing_workouts = self.load_existing_data("workouts", "workouts")
            merged_workouts = self.merge_data_by_id(existing_workouts, workouts)
            
            self.save_data(merged_workouts, "workouts", "workouts")
            
            # Save individual workout records
            for workout in workouts:
                workout_id = workout["id"]
                self.save_data(workout, f"workout_{workout_id}", "workouts/individual")
                await asyncio.sleep(0.1)
        
        self.last_fetch_times["workouts"] = time.time()
        self.save_last_fetch_times()
    
    async def fetch_all_data(self):
        """Fetch all WHOOP data with redundancy prevention"""
        print("🚀 Starting WHOOP data fetch...")
        print(f"📁 Data will be saved to: {self.data_dir.absolute()}")
        
        start_time = datetime.now()
        
        try:
            # Check what needs to be fetched
            fetch_needed = []
            for data_type in ["user_profile", "cycles", "recovery", "sleep", "workouts"]:
                if self.should_fetch(data_type):
                    fetch_needed.append(data_type)
                else:
                    print(f"⏭️  Skipping {data_type} - fetched recently")
            
            if not fetch_needed:
                print("✅ All data is up to date!")
                return
            
            # Fetch only what's needed
            if "user_profile" in fetch_needed:
                await self.fetch_user_data()
            if "cycles" in fetch_needed:
                await self.fetch_cycles_data()
            if "recovery" in fetch_needed:
                await self.fetch_recovery_data()
            if "sleep" in fetch_needed:
                await self.fetch_sleep_data()
            if "workouts" in fetch_needed:
                await self.fetch_workout_data()
            
            # Create summary
            end_time = datetime.now()
            duration = end_time - start_time
            
            summary = {
                "fetch_completed_at": end_time.isoformat(),
                "fetch_duration_seconds": duration.total_seconds(),
                "data_types_fetched": fetch_needed,
                "api_version": "v2",
                "notes": "Data fetched using WHOOP API v2 with redundancy prevention."
            }
            
            self.save_data(summary, "fetch_summary", "")
            
            print(f"\n✅ Data fetch completed successfully!")
            print(f"⏱️  Total time: {duration.total_seconds():.1f} seconds")
            print(f"📊 Fetched: {', '.join(fetch_needed)}")
            
        except Exception as e:
            print(f"\n❌ Error during data fetch: {e}")
            raise

    async def continuous_fetch(self, interval_minutes: int = 30):
        """Run continuous data fetching at specified intervals"""
        print(f"🔄 Starting continuous data fetching every {interval_minutes} minutes...")
        
        while True:
            try:
                await self.fetch_all_data()
                print(f"⏰ Next fetch in {interval_minutes} minutes...")
                await asyncio.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                print("\n🛑 Continuous fetching stopped by user")
                break
            except Exception as e:
                print(f"❌ Error in continuous fetch: {e}")
                print(f"⏰ Retrying in {interval_minutes} minutes...")
                await asyncio.sleep(interval_minutes * 60)

async def main():
    """Main function to run the data fetcher"""
    # This will be called from the server with the access token
    print("WHOOP Data Fetcher requires an access token.")
    print("Please use the /fetch-data endpoint on the running server.")

if __name__ == "__main__":
    asyncio.run(main()) 