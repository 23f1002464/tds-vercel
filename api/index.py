from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
from collections import defaultdict

# --- Application Setup ---
app = FastAPI()

# --- CORS Middleware ---
# This allows the API to be called from any web frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Data Models ---
class AnalyticsRequest(BaseModel):
    regions: List[str]
    threshold_ms: int

# --- Data Loading and Processing ---

# It's more efficient to load data once when the app starts, not on every request.
# This data is now stored in a global variable.
TELEMETRY_DATA = [
    {"region": "apac", "service": "catalog", "latency_ms": 103.67, "uptime_pct": 99.269, "timestamp": 20250301},
    {"region": "apac", "service": "support", "latency_ms": 206.81, "uptime_pct": 98.88, "timestamp": 20250302},
    {"region": "apac", "service": "analytics", "latency_ms": 150.4, "uptime_pct": 97.798, "timestamp": 20250303},
    {"region": "apac", "service": "checkout", "latency_ms": 139.44, "uptime_pct": 97.891, "timestamp": 20250304},
    {"region": "apac", "service": "support", "latency_ms": 159.57, "uptime_pct": 97.519, "timestamp": 20250305},
    {"region": "apac", "service": "checkout", "latency_ms": 231.17, "uptime_pct": 97.365, "timestamp": 20250306},
    {"region": "apac", "service": "checkout", "latency_ms": 155.99, "uptime_pct": 97.822, "timestamp": 20250307},
    {"region": "apac", "service": "recommendations", "latency_ms": 177.31, "uptime_pct": 97.575, "timestamp": 20250308},
    {"region": "apac", "service": "payments", "latency_ms": 236.58, "uptime_pct": 97.27, "timestamp": 20250309},
    {"region": "apac", "service": "support", "latency_ms": 152.06, "uptime_pct": 98.879, "timestamp": 20250310},
    {"region": "apac", "service": "analytics", "latency_ms": 143.73, "uptime_pct": 98.738, "timestamp": 20250311},
    {"region": "apac", "service": "checkout", "latency_ms": 100.63, "uptime_pct": 98.563, "timestamp": 20250312},
    {"region": "emea", "service": "analytics", "latency_ms": 186.77, "uptime_pct": 97.498, "timestamp": 20250301},
    {"region": "emea", "service": "support", "latency_ms": 223.88, "uptime_pct": 99.071, "timestamp": 20250302},
    {"region": "emea", "service": "payments", "latency_ms": 225.29, "uptime_pct": 98.303, "timestamp": 20250303},
    {"region": "emea", "service": "payments", "latency_ms": 126.49, "uptime_pct": 97.432, "timestamp": 20250304},
    {"region": "emea", "service": "catalog", "latency_ms": 183.37, "uptime_pct": 99.029, "timestamp": 20250305},
    {"region": "emea", "service": "support", "latency_ms": 169.76, "uptime_pct": 98.48, "timestamp": 20250306},
    {"region": "emea", "service": "catalog", "latency_ms": 135.66, "uptime_pct": 97.909, "timestamp": 20250307},
    {"region": "emea", "service": "support", "latency_ms": 125.56, "uptime_pct": 97.164, "timestamp": 20250308},
    {"region": "emea", "service": "recommendations", "latency_ms": 219.31, "uptime_pct": 97.868, "timestamp": 20250309},
    {"region": "emea", "service": "payments", "latency_ms": 132, "uptime_pct": 99.2, "timestamp": 20250310},
    {"region": "emea", "service": "checkout", "latency_ms": 122.16, "uptime_pct": 98.539, "timestamp": 20250311},
    {"region": "emea", "service": "checkout", "latency_ms": 107.86, "uptime_pct": 98.703, "timestamp": 20250312},
    {"region": "amer", "service": "recommendations", "latency_ms": 162.82, "uptime_pct": 98.083, "timestamp": 20250301},
    {"region": "amer", "service": "recommendations", "latency_ms": 171.9, "uptime_pct": 97.271, "timestamp": 20250302},
    {"region": "amer", "service": "analytics", "latency_ms": 154.74, "uptime_pct": 98.954, "timestamp": 20250303},
    {"region": "amer", "service": "recommendations", "latency_ms": 130.84, "uptime_pct": 98.581, "timestamp": 20250304},
    {"region": "amer", "service": "payments", "latency_ms": 222.48, "uptime_pct": 98.746, "timestamp": 20250305},
    {"region": "amer", "service": "support", "latency_ms": 123.92, "uptime_pct": 99.387, "timestamp": 20250306},
    {"region": "amer", "service": "payments", "latency_ms": 183.61, "uptime_pct": 99.035, "timestamp": 20250307},
    {"region": "amer", "service": "analytics", "latency_ms": 157.98, "uptime_pct": 99.103, "timestamp": 20250308},
    {"region": "amer", "service": "recommendations", "latency_ms": 212.52, "uptime_pct": 98.047, "timestamp": 20250309},
    {"region": "amer", "service": "payments", "latency_ms": 215.67, "uptime_pct": 99.02, "timestamp": 20250310},
    {"region": "amer", "service": "recommendations", "latency_ms": 141.13, "uptime_pct": 99.242, "timestamp": 20250311},
    {"region": "amer", "service": "payments", "latency_ms": 134.05, "uptime_pct": 98.826, "timestamp": 20250312}
]

# Pre-process the data into a more efficient structure for lookups.
# This groups all data points by region, avoiding re-filtering on every request.
PROCESSED_DATA = defaultdict(lambda: {'latencies': [], 'uptimes': []})
for item in TELEMETRY_DATA:
    region = item['region']
    PROCESSED_DATA[region]['latencies'].append(item['latency_ms'])
    PROCESSED_DATA[region]['uptimes'].append(item['uptime_pct'])

# --- API Endpoints ---

@app.post("/api/latency")
async def analyze_latency(request: AnalyticsRequest):
    """
    Analyzes telemetry data for a given list of regions and returns key statistics.
    """
    results = {}
    
    for region in request.regions:
        # Check if we have data for the requested region
        if region not in PROCESSED_DATA:
            # You might want to skip, return an error, or return empty data.
            # Here, we'll return a zeroed-out entry.
            results[region] = {
                "avg_latency": 0,
                "p95_latency": 0,
                "avg_uptime": 0,
                "breaches": 0
            }
            continue

        # Use the pre-processed data for calculations
        latencies = PROCESSED_DATA[region]['latencies']
        uptimes = PROCESSED_DATA[region]['uptimes']
        
        # Using numpy is highly optimized for these kinds of numerical operations.
        # It's more reliable and faster than a custom percentile function.
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        avg_uptime = np.mean(uptimes)
        
        # Efficiently count breaches using a numpy array comparison
        breaches = np.sum(np.array(latencies) > request.threshold_ms)
        
        results[region] = {
            "avg_latency": round(avg_latency, 2),
            "p95_latency": round(p95_latency, 2),
            "avg_uptime": round(avg_uptime, 4),
            "breaches": int(breaches) # Convert from numpy int to standard python int
        }
    
    return results

@app.get("/")
async def root():
    """
    Root endpoint providing basic API information.
    """
    return {"message": "Latency Analytics API", "status": "running"}

