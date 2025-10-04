from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List
import statistics

app = FastAPI()

class AnalyticsRequest(BaseModel):
    regions: List[str]
    threshold_ms: int

# Manual CORS middleware
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS, PUT, DELETE"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    return response

# Load the telemetry data
def load_telemetry_data():
    return [
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

def calculate_percentile(data: List[float], percentile: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = (len(sorted_data) - 1) * percentile / 100
    lower_index = int(index)
    upper_index = lower_index + 1
    
    if upper_index >= len(sorted_data):
        return sorted_data[lower_index]
    
    weight = index - lower_index
    return sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight

@app.post("/api/latency")
async def analyze_latency(request: AnalyticsRequest):
    try:
        telemetry_data = load_telemetry_data()
        
        # Filter data for requested regions
        filtered_data = [item for item in telemetry_data if item["region"] in request.regions]
        
        # Calculate metrics per region
        results = {}
        
        for region in request.regions:
            region_data = [item for item in filtered_data if item["region"] == region]
            
            if not region_data:
                results[region] = {
                    "avg_latency": 0,
                    "p95_latency": 0,
                    "avg_uptime": 0,
                    "breaches": 0
                }
                continue
            
            # Extract metrics
            latencies = [item["latency_ms"] for item in region_data]
            uptimes = [item["uptime_pct"] for item in region_data]
            
            # Calculate statistics
            avg_latency = statistics.mean(latencies) if latencies else 0
            p95_latency = calculate_percentile(latencies, 95)
            avg_uptime = statistics.mean(uptimes) if uptimes else 0
            breaches = sum(1 for latency in latencies if latency > request.threshold_ms)
            
            results[region] = {
                "avg_latency": round(avg_latency, 2),
                "p95_latency": round(p95_latency, 2),
                "avg_uptime": round(avg_uptime, 4),
                "breaches": breaches
            }
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Handle OPTIONS preflight requests explicitly
@app.options("/api/latency")
async def options_latency():
    from fastapi.responses import JSONResponse
    response = JSONResponse(content={"message": "OK"})
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

@app.get("/")
async def root():
    return {"message": "Latency Analytics API", "status": "running"}