"""Parallel fetch of settled GB imbalance prices from Elexon Insights (system prices only)."""
import csv, json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

BASE = "https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/system-prices"
out_dir, first, last = sys.argv[1], date.fromisoformat(sys.argv[2]), date.fromisoformat(sys.argv[3])

def get(day):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"{BASE}/{day}?format=json", timeout=60) as r:
                return [(x["startTime"], x["settlementDate"], x["settlementPeriod"], x["systemSellPrice"], x["netImbalanceVolume"]) for x in json.load(r)["data"]]
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))

days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
with ThreadPoolExecutor(8) as pool:
    rows = sorted({row for chunk in pool.map(get, days) for row in chunk})
with open(f"{out_dir}/system_prices.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["start_utc", "settlement_date", "settlement_period", "price", "niv"])
    w.writerows(rows)
print(len(rows), "rows", rows[0][0], rows[-1][0])
