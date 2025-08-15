import os
import time
import requests


def main():
    target = os.environ.get("TARGET_URL", "http://whoop_api:8000/fetch-data")
    interval = int(os.environ.get("RUN_INTERVAL_SECONDS", "3600"))
    while True:
        try:
            print(f"[fetch_scheduler] POST {target}")
            # whoop_api endpoint uses GET per implementation; trigger via GET
            resp = requests.get(target, timeout=60)
            print(f"[fetch_scheduler] status={resp.status_code}")
        except Exception as e:
            print(f"[fetch_scheduler] error: {e}")
        time.sleep(interval)


if __name__ == "__main__":
    main()

