"""
main.py  –  AI-Powered Crime-Aware Safe Route Navigation System
================================================================
FastAPI backend:
  GET  /           → index page (input form)
  POST /route      → computes safe route, returns rendered map HTML
  GET  /health     → health check
"""

from __future__ import annotations

import os
import io
import math
import datetime
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import joblib
import folium
from folium.plugins import HeatMap
import httpx
import polyline as polyline_lib
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjQ3YTFjMzM4N2NlMDRmODU5ODViZTViMWMwOTQ2NDU5IiwiaCI6Im11cm11cjY0In0="
ORS_BASE    = "https://api.openrouteservice.org"
NOMINATIM   = "https://nominatim.openstreetmap.org/search"

BASE_DIR    = Path(__file__).parent
MODEL_PATH  = BASE_DIR / "crime_model.pkl"
DATA_PATH   = BASE_DIR / "crime_data.pkl"

MAX_ROUTE_KM = 80   # reject routes longer than this (ORS free-tier limit guard)
SAMPLE_PTS   = 30   # points sampled per route for risk estimation

# ─────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────

app = FastAPI(title="SafeRoute AI")

static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Load ML model + heatmap data once at startup
_model      = None
_crime_data = None   # list of [lat, lon, risk]

def _load_assets():
    global _model, _crime_data
    if MODEL_PATH.exists():
        _model = joblib.load(str(MODEL_PATH))
        print("✅  crime_model.pkl loaded")
    else:
        print("⚠️  crime_model.pkl NOT found – run train_model.py first")

    if DATA_PATH.exists():
        _crime_data = joblib.load(str(DATA_PATH))
        print(f"✅  crime_data.pkl loaded ({len(_crime_data)} points)")
    else:
        print("⚠️  crime_data.pkl NOT found – heatmap will be empty")

_load_assets()


# ─────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def _route_length_km(coords: list[tuple]) -> float:
    total = 0.0
    for i in range(len(coords) - 1):
        total += _haversine_km(coords[i][0], coords[i][1],
                                coords[i+1][0], coords[i+1][1])
    return total


def _sample_route(coords: list[tuple], n: int = SAMPLE_PTS) -> list[tuple]:
    """Return n evenly-spaced points from the route."""
    if len(coords) <= n:
        return coords
    idxs = np.linspace(0, len(coords) - 1, n, dtype=int)
    return [coords[i] for i in idxs]


def _predict_risk(lat: float, lon: float, hour: int, day: int, month: int) -> float:
    if _model is None:
        return 0.5   # fallback when model absent
    features = np.array([[lat, lon, hour, day, month]], dtype=float)
    risk = float(_model.predict(features)[0])
    return max(0.0, min(1.0, risk))


def _route_risk_score(coords: list[tuple], hour: int, day: int, month: int) -> float:
    sample = _sample_route(coords)
    risks  = [_predict_risk(lat, lon, hour, day, month) for lat, lon in sample]
    return float(np.mean(risks))


# ─────────────────────────────────────────────────────────────
# Geocoding
# ─────────────────────────────────────────────────────────────

async def _geocode(place: str) -> Optional[tuple[float, float]]:
    """Return (lat, lon) for a place name via Nominatim."""
    params = {
        "q":            place + ", Chicago, IL",
        "format":       "json",
        "limit":        1,
        "addressdetails": 0,
    }
    headers = {"User-Agent": "SafeRouteAI/1.0 (demo)"}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(NOMINATIM, params=params, headers=headers)
    if resp.status_code != 200 or not resp.json():
        # Try without city suffix
        params["q"] = place
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(NOMINATIM, params=params, headers=headers)
    data = resp.json()
    if not data:
        return None
    return float(data[0]["lat"]), float(data[0]["lon"])


# ─────────────────────────────────────────────────────────────
# Routing  (OpenRouteService)
# ─────────────────────────────────────────────────────────────

async def _fetch_routes(src: tuple, dst: tuple) -> list[list[tuple]]:
    """
    Call ORS /v2/directions/driving-car/geojson with alternative_routes.
    Returns list of decoded coordinate lists  [(lat,lon), …].
    """
    url = f"{ORS_BASE}/v2/directions/driving-car/geojson"
    headers = {
        "Authorization": ORS_API_KEY,
        "Content-Type":  "application/json",
    }
    body = {
        "coordinates": [
            [src[1], src[0]],   # ORS uses [lon, lat]
            [dst[1], dst[0]],
        ],
        "alternative_routes": {
            "target_count":          3,
            "weight_factor":         1.6,
            "share_factor":          0.6,
        },
        "geometry_simplify": False,
        "instructions":      False,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=body, headers=headers)

    if resp.status_code != 200:
        raise ValueError(f"ORS API error {resp.status_code}: {resp.text[:300]}")

    features = resp.json().get("features", [])
    routes: list[list[tuple]] = []

    for feat in features:
        geom = feat["geometry"]
        if geom["type"] == "LineString":
            # ORS returns [lon, lat] → flip to (lat, lon)
            coords = [(c[1], c[0]) for c in geom["coordinates"]]
            routes.append(coords)

    return routes


# ─────────────────────────────────────────────────────────────
# Map builder
# ─────────────────────────────────────────────────────────────

ROUTE_COLORS = ["#4FC3F7", "#81C784", "#FFB74D", "#CE93D8"]  # 4 palette slots
SAFE_COLOR   = "#00E676"   # bright green for safest
OTHER_ALPHA  = "#546E7A"   # muted grey-blue for non-safe routes


def _build_map(
    src:      tuple[float, float],
    dst:      tuple[float, float],
    routes:   list[list[tuple]],
    risks:    list[float],
    src_name: str,
    dst_name: str,
) -> str:
    """Render a Folium map and return HTML string."""

    # Centre map on midpoint
    mid_lat = (src[0] + dst[0]) / 2
    mid_lon = (src[1] + dst[1]) / 2
    fmap    = folium.Map(
        location=[mid_lat, mid_lon],
        zoom_start=13,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )

    # ── Crime heatmap ──────────────────────────────────────────
    if _crime_data:
        heat_data = [[r[0], r[1], r[2]] for r in _crime_data]
        HeatMap(
            heat_data,
            min_opacity=0.25,
            max_val=1.0,
            radius=14,
            blur=18,
            gradient={0.2: "#1a237e", 0.45: "#e65100",
                      0.7: "#b71c1c", 1.0: "#ff1744"},
        ).add_to(fmap)

    # ── Routes ────────────────────────────────────────────────
    best_idx = int(np.argmin(risks))

    for i, (coords, risk) in enumerate(zip(routes, risks)):
        is_best = (i == best_idx)
        color   = SAFE_COLOR if is_best else OTHER_ALPHA
        weight  = 6 if is_best else 3
        opacity = 1.0 if is_best else 0.45
        dash    = None if is_best else "8 6"

        tooltip = (
            f"{'✅ SAFEST ROUTE' if is_best else f'Route {i+1}'} | "
            f"Risk: {risk:.2%} | "
            f"Length: {_route_length_km(coords):.1f} km"
        )

        line = folium.PolyLine(
            locations=coords,
            color=color,
            weight=weight,
            opacity=opacity,
            dash_array=dash,
            tooltip=tooltip,
        )
        line.add_to(fmap)

        # Animated flow on safest route (fake via a thinner lighter line on top)
        if is_best:
            folium.PolyLine(
                locations=coords,
                color="#FFFFFF",
                weight=2,
                opacity=0.35,
                dash_array="4 12",
                tooltip=tooltip,
            ).add_to(fmap)

    # ── Markers ───────────────────────────────────────────────
    folium.Marker(
        location=src,
        tooltip=f"🟢 Start: {src_name}",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(fmap)

    folium.Marker(
        location=dst,
        tooltip=f"🔴 End: {dst_name}",
        icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
    ).add_to(fmap)

    # ── Risk legend ───────────────────────────────────────────
    legend_items = ""
    for i, (risk, coords) in enumerate(zip(risks, routes)):
        label = "✅ SAFEST" if i == best_idx else f"Route {i+1}"
        bar_w = int(risk * 120)
        legend_items += f"""
        <div style="margin:6px 0;display:flex;align-items:center;gap:8px">
          <span style="font-size:11px;min-width:90px;color:{'#00E676' if i==best_idx else '#90A4AE'}">{label}</span>
          <div style="background:#1e2a38;border-radius:3px;height:10px;width:120px;overflow:hidden">
            <div style="background:{'#00E676' if i==best_idx else '#e53935'};height:100%;width:{bar_w}px"></div>
          </div>
          <span style="font-size:10px;color:#78909C">{risk:.1%}</span>
        </div>"""

    legend_html = f"""
    <div id="legend" style="
        position:fixed;bottom:28px;left:28px;z-index:9999;
        background:rgba(13,17,23,0.92);
        border:1px solid #263238;border-radius:10px;
        padding:14px 18px;font-family:'Courier New',monospace;
        box-shadow:0 4px 24px rgba(0,0,0,0.5);min-width:250px">
      <div style="font-size:13px;font-weight:700;color:#00E676;margin-bottom:10px;
                  letter-spacing:1px;border-bottom:1px solid #1e3a2f;padding-bottom:6px">
        🛡 ROUTE RISK ANALYSIS
      </div>
      {legend_items}
      <div style="margin-top:10px;font-size:9px;color:#546E7A;border-top:1px solid #1e2a38;padding-top:6px">
        Risk scored by AI model · Lower = Safer
      </div>
    </div>"""

    fmap.get_root().html.add_child(folium.Element(legend_html))

    # Auto-fit bounds to all routes
    all_pts = [p for r in routes for p in r]
    if all_pts:
        lats = [p[0] for p in all_pts]
        lons = [p[1] for p in all_pts]
        fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    return fmap._repr_html_()


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {
        "status":      "ok",
        "model_loaded": _model is not None,
        "crime_points": len(_crime_data) if _crime_data else 0,
    }


@app.post("/route", response_class=HTMLResponse)
async def compute_route(
    request:     Request,
    source:      str = Form(...),
    destination: str = Form(...),
):
    now   = datetime.datetime.now()
    hour  = now.hour
    day   = now.weekday() + 1   # 1=Mon … 7=Sun
    month = now.month

    error_ctx = {
        "request":     request,
        "source":      source,
        "destination": destination,
        "error":       None,
        "map_html":    None,
        "route_info":  [],
        "hour":        hour,
        "day":         day,
        "month":       month,
    }

    # ── Geocode ───────────────────────────────────────────────
    try:
        src_coords = await _geocode(source)
        if not src_coords:
            error_ctx["error"] = f"Could not locate '{source}'. Try adding 'Chicago' to the name."
            return templates.TemplateResponse("map.html", error_ctx)

        dst_coords = await _geocode(destination)
        if not dst_coords:
            error_ctx["error"] = f"Could not locate '{destination}'. Try adding 'Chicago' to the name."
            return templates.TemplateResponse("map.html", error_ctx)
    except Exception as exc:
        error_ctx["error"] = f"Geocoding failed: {exc}"
        return templates.TemplateResponse("map.html", error_ctx)

    # ── Distance sanity check ─────────────────────────────────
    straight_km = _haversine_km(*src_coords, *dst_coords)
    if straight_km > MAX_ROUTE_KM:
        error_ctx["error"] = (
            f"Distance ({straight_km:.0f} km) exceeds the {MAX_ROUTE_KM} km limit. "
            "Please choose locations closer together."
        )
        return templates.TemplateResponse("map.html", error_ctx)

    if straight_km < 0.05:
        error_ctx["error"] = "Start and destination are the same (or too close)."
        return templates.TemplateResponse("map.html", error_ctx)

    # ── Fetch routes ──────────────────────────────────────────
    try:
        routes = await _fetch_routes(src_coords, dst_coords)
    except Exception as exc:
        error_ctx["error"] = f"Routing API error: {exc}"
        return templates.TemplateResponse("map.html", error_ctx)

    if not routes:
        error_ctx["error"] = "No routes found between these locations."
        return templates.TemplateResponse("map.html", error_ctx)

    # ── Score each route ──────────────────────────────────────
    risks = [_route_risk_score(r, hour, day, month) for r in routes]
    best_idx = int(np.argmin(risks))

    route_info = []
    for i, (r, risk) in enumerate(zip(routes, risks)):
        route_info.append({
            "index":   i + 1,
            "risk":    f"{risk:.1%}",
            "length":  f"{_route_length_km(r):.1f} km",
            "safest":  (i == best_idx),
        })

    # ── Build map ─────────────────────────────────────────────
    try:
        map_html = _build_map(
            src_coords, dst_coords,
            routes, risks,
            source, destination,
        )
    except Exception as exc:
        traceback.print_exc()
        error_ctx["error"] = f"Map rendering error: {exc}"
        return templates.TemplateResponse("map.html", error_ctx)

    return templates.TemplateResponse("map.html", {
        "request":     request,
        "source":      source,
        "destination": destination,
        "error":       None,
        "map_html":    map_html,
        "route_info":  route_info,
        "hour":        hour,
        "day":         day,
        "month":       month,
    })
