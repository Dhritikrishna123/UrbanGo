import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import math
import csv
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
import os

# Optional OSM imports (graceful fallback if not available)
try:
    import osmnx as ox
    import networkx as nx
    from shapely.geometry import LineString, Point
    from shapely.ops import linemerge
    _OSM_AVAILABLE = True
except Exception:
    _OSM_AVAILABLE = False

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

@dataclass
class BusStop:
    stop_id: str
    name: str
    lat: float
    lon: float
    is_major_hub: bool = False

@dataclass
class Route:
    route_id: str
    name: str
    stops: List[str]
    base_frequency_mins: int  # base frequency in minutes

class RealisticBusETAGenerator:
    def __init__(self, city_place: str = 'Jalandhar, Punjab, India'):
        # Force Jalandhar-only per requirements
        self.city_place = 'Jalandhar, Punjab, India'
        self.setup_kolkata_infrastructure()
        self.setup_temporal_patterns()
        self.setup_weather_patterns()
        self.setup_traffic_patterns()
        self.setup_segment_speed_profiles()
        self._init_operational_state()
        self._schedule_rows = []
        
    def setup_kolkata_infrastructure(self):
        """Setup Jalandhar bus network using OSM when available; fallback to synthetic Jalandhar."""
        if _OSM_AVAILABLE:
            try:
                self._setup_city_infrastructure_osm()
                return
            except Exception as e:
                print(f"[WARN] OSM-based setup failed, falling back to synthetic. Reason: {e}")
        # Jalandhar-only synthetic fallback with dense routes
        self._setup_city_infrastructure_synthetic()
    
    def setup_temporal_patterns(self):
        """Setup time-based patterns"""
        self.rush_hours = {
            'morning': (7, 10),    # 7-10 AM
            'evening': (17, 20),   # 5-8 PM
        }
        
        self.operating_hours = (5, 23)  # 5 AM to 11 PM
        
        # Traffic multipliers by hour
        self.traffic_multipliers = {
            5: 0.7, 6: 0.8, 7: 1.4, 8: 1.8, 9: 1.6, 10: 1.2, 11: 1.0,
            12: 1.1, 13: 1.0, 14: 1.0, 15: 1.1, 16: 1.2, 17: 1.5, 
            18: 1.9, 19: 1.7, 20: 1.4, 21: 1.1, 22: 0.9, 23: 0.8
        }
    
    def setup_weather_patterns(self):
        """Setup weather impact patterns"""
        self.weather_conditions = {
            'clear': {'probability': 0.4, 'speed_factor': 1.0, 'delay_factor': 1.0},
            'cloudy': {'probability': 0.25, 'speed_factor': 0.98, 'delay_factor': 1.05},
            'light_rain': {'probability': 0.2, 'speed_factor': 0.85, 'delay_factor': 1.3},
            'heavy_rain': {'probability': 0.1, 'speed_factor': 0.6, 'delay_factor': 1.8},
            'fog': {'probability': 0.05, 'speed_factor': 0.7, 'delay_factor': 1.4}
        }
        # Spatiotemporal weather model parameters
        self._weather_cache: Dict[Tuple[str, int, int], str] = {}
        self.weather_cell_km = 3.0  # grid size ~3km for urban coherence
        self.weather_hour_persistence = 0.85  # probability weather persists next hour
        self.weather_noise_prob = 0.05  # chance of localized deviation

    def _get_weather_base_probs(self, month: int) -> Tuple[list, list]:
        """Return (conditions, probabilities) for the given month."""
        # Base probabilities by season
        if 6 <= month <= 9:  # Monsoon
            probs = [0.2, 0.2, 0.35, 0.2, 0.05]
        elif month == 11 or month == 12 or month <= 2:  # Winter
            probs = [0.5, 0.3, 0.1, 0.02, 0.08]
        else:  # Shoulder/Summer
            probs = [0.6, 0.25, 0.12, 0.02, 0.01]
        conditions = list(self.weather_conditions.keys())
        return conditions, probs

    def _weather_cell_indices(self, lat: float, lon: float) -> Tuple[int, int]:
        """Map lat/lon to discrete weather grid cell indices."""
        # Degrees per km approximation at given latitude
        lat_rad = math.radians(lat)
        deg_per_km_lat = 1.0 / 110.574
        deg_per_km_lon = 1.0 / (111.320 * max(0.1, math.cos(lat_rad)))
        cell_deg_lat = self.weather_cell_km * deg_per_km_lat
        cell_deg_lon = self.weather_cell_km * deg_per_km_lon
        ilat = int(lat / cell_deg_lat)
        ilon = int(lon / cell_deg_lon)
        return ilat, ilon
    
    def setup_traffic_patterns(self):
        """Setup traffic and congestion patterns"""
        # Major hubs have higher congestion
        self.hub_congestion = {
            'HUB001': 1.5,  # Howrah
            'HUB002': 1.3,  # Sealdah  
            'HUB003': 1.4,  # Esplanade
            'HUB004': 1.2,  # Airport
            'HUB005': 1.3,  # Shyambazar
        }
    
    # ---------- OSM helpers (optional) ----------
    def _load_osm_graph(self):
        """Load city drivable street graph."""
        if not _OSM_AVAILABLE:
            raise RuntimeError("OSM libraries not available")
        ox.settings.log_console = False
        ox.settings.use_cache = True
        return ox.graph_from_place(self.city_place, network_type="drive")

    def _nearest_node(self, G, lat, lon):
        return ox.distance.nearest_nodes(G, lon, lat)

    def _route_nodes(self, G, waypoints):
        """Build a route path by chaining shortest paths across waypoint sequence."""
        full_path = []
        for i in range(len(waypoints) - 1):
            u = self._nearest_node(G, waypoints[i][0], waypoints[i][1])
            v = self._nearest_node(G, waypoints[i+1][0], waypoints[i+1][1])
            sub_path = ox.routing.shortest_path(G, u, v, weight="length")
            if not sub_path:
                raise RuntimeError("No path found between waypoints")
            if full_path and sub_path[0] == full_path[-1]:
                full_path.extend(sub_path[1:])
            else:
                full_path.extend(sub_path)
        return full_path

    def _route_linestring(self, G, path_nodes):
        """Convert node path to a merged LineString geometry."""
        gdf = ox.routing.route_to_gdf(G, path_nodes)
        geom = gdf.geometry.unary_union
        if geom.geom_type == "MultiLineString":
            geom = linemerge(geom)
        return geom

    def _points_every(self, line: LineString, spacing_m=400):
        """Yield points every ~spacing_m along the line."""
        total_len = line.length
        d = 0.0
        while d <= total_len:
            yield line.interpolate(d)
            d += spacing_m
        if d - spacing_m < total_len:
            yield line.interpolate(total_len)

    def _build_stops_from_corridor(self, G, corridor_name, waypoints, spacing_m=400, hub_labels=None):
        """
        Create stops along an OSM corridor.
        - hub_labels: list of tuples (name, lat, lon) to mark as hubs if close-by.
        """
        path_nodes = self._route_nodes(G, waypoints)
        line = self._route_linestring(G, path_nodes)
        stops_ids = []
        local_stops = {}
        idx = 0

        # densify stops
        for pt in self._points_every(line, spacing_m=spacing_m):
            idx += 1
            lat, lon = pt.y, pt.x
            stop_id = f"{corridor_name[:3].upper()}{idx:03d}"
            # default stop
            is_hub = False
            name = f"{corridor_name} Stop {idx}"

            # upgrade to hub if near any provided hub points (within ~120 m)
            if hub_labels:
                for hub_name, hlat, hlon in hub_labels:
                    if ox.distance.great_circle_vec(lat, lon, hlat, hlon) <= 0.12:  # km
                        is_hub = True
                        name = hub_name

            local_stops[stop_id] = BusStop(stop_id, name, lat, lon, is_hub)
            stops_ids.append(stop_id)

        return local_stops, stops_ids

    def _setup_city_infrastructure_synthetic(self):
        """Jalandhar-only synthetic fallback with dense stops (no OSM)."""
        city = 'jalandhar'
        self.stops = {}
        self.routes = {}
        # Define synthetic hubs and polyline corridors
        hubs = {
            'HUB001': ('Jalandhar Bus Stand', 31.3260, 75.5760),
            'HUB002': ('Jalandhar City Railway Station', 31.3269, 75.5785),
            'HUB003': ('BMC Chowk', 31.3238, 75.5820),
            'HUB004': ('Model Town Market', 31.3220, 75.5920),
            'HUB005': ('PAP Chowk', 31.3010, 75.6200),
            'HUB006': ('Maqsudan Chowk', 31.3560, 75.5580),
        }
        corridors = [
            ('RT001', 'Bus Stand – BMC – Model Town – Station (Loop)',
             [(31.3260, 75.5760), (31.3238, 75.5820), (31.3220, 75.5920), (31.3269, 75.5785), (31.3260, 75.5760)]),
            ('RT002', 'Bus Stand – Maqsudan Corridor',
             [(31.3260, 75.5760), (31.3400, 75.5660), (31.3560, 75.5580)]),
            ('RT003', 'Railway Station – BMC – PAP',
             [(31.3269, 75.5785), (31.3238, 75.5820), (31.3010, 75.6200)]),
        ]

        # Create densely spaced stops every ~250m by linear interpolation along corridor polylines
        def interpolate_points(points, spacing_km=0.25):
            out = []
            for i in range(len(points) - 1):
                lat1, lon1 = points[i]
                lat2, lon2 = points[i + 1]
                # approximate distance in km using haversine
                d = self.calculate_distance(lat1, lon1, lat2, lon2)
                n = max(1, int(d / spacing_km))
                for k in range(n):
                    frac = k / n
                    out.append((lat1 + frac * (lat2 - lat1), lon1 + frac * (lon2 - lon1)))
            out.append(points[-1])
            return out

        # Build stops and routes
        stop_counter = 0
        for route_id, name, poly in corridors:
            dense = interpolate_points(poly, spacing_km=random.uniform(0.22, 0.32))
            stop_ids = []
            for lat, lon in dense:
                stop_counter += 1
                sid = f"STP{stop_counter:04d}"
                # Mark hub if within ~150m of a hub point
                is_hub = False
                sname = f"{name} Stop {len(stop_ids)+1}"
                for hub_id, (hname, hlat, hlon) in hubs.items():
                    if self.calculate_distance(lat, lon, hlat, hlon) <= 0.15:
                        is_hub = True
                        sname = hname
                        break
                self.stops[sid] = BusStop(sid, sname, lat, lon, is_hub)
                stop_ids.append(sid)
            # Ensure 20–40 stops per route
            if len(stop_ids) < 20 and len(dense) > 1:
                # add midpoints to reach at least 20
                extra = []
                for i in range(len(dense) - 1):
                    latm = (dense[i][0] + dense[i+1][0]) / 2
                    lonm = (dense[i][1] + dense[i+1][1]) / 2
                    extra.append((latm, lonm))
                for lat, lon in extra[:20 - len(stop_ids)]:
                    stop_counter += 1
                    sid = f"STP{stop_counter:04d}"
                    self.stops[sid] = BusStop(sid, f"{name} Extra", lat, lon, False)
                    stop_ids.append(sid)

            self.routes[route_id] = Route(route_id, name, stop_ids, base_frequency_mins=random.choice([8, 10, 12, 15]))

        # Promote canonical hub IDs by replacing nearest generated stops
        for hub_id, (hname, hlat, hlon) in hubs.items():
            best_id, best_d = None, 1e9
            for sid, s in self.stops.items():
                d = self.calculate_distance(s.lat, s.lon, hlat, hlon)
                if d < best_d:
                    best_id, best_d = sid, d
            if best_id:
                bs = self.stops[best_id]
                self.stops[hub_id] = BusStop(hub_id, hname, bs.lat, bs.lon, True)
                for rid, route in self.routes.items():
                    route.stops = [hub_id if x == best_id else x for x in route.stops]
                if hub_id != best_id and best_id in self.stops:
                    del self.stops[best_id]

        # Fleet assignment scaled to route length
        self.buses = {}
        bus_count = 0
        for route_id, route in self.routes.items():
            nstops = len(route.stops)
            buses_for_route = max(8, nstops // 6)
            for _ in range(buses_for_route):
                bus_count += 1
                self.buses[f'BUS{bus_count:03d}'] = {
                    'primary_route': route_id,
                    'age_years': random.randint(1, 12),
                    'driver_type': random.choice(['fast', 'normal', 'slow']),
                    'condition': random.choice(['excellent', 'good', 'average', 'poor'])
                }

    def _setup_city_infrastructure_osm(self):
        """OSM-based routes and real bus stops for Jalandhar only."""
        print(f"[INFO] Building OSM-based routes for Jalandhar...")
        G = self._load_osm_graph()
        self.stops = {}
        self.routes = {}
        spacing = random.randint(200, 300)
        # Hubs and corridors for Jalandhar
        hubs = {
            'HUB001': ('Jalandhar Bus Stand', 31.3260, 75.5760),
            'HUB002': ('Jalandhar City Railway Station', 31.3269, 75.5785),
            'HUB003': ('BMC Chowk', 31.3238, 75.5820),
            'HUB004': ('Model Town Market', 31.3220, 75.5920),
            'HUB005': ('PAP Chowk', 31.3010, 75.6200),
            'HUB006': ('Maqsudan Chowk', 31.3560, 75.5580),
            'HUB007': ('Industrial Area', 31.3150, 75.6350),
        }
        RT001_wps = [
            (31.3260, 75.5760), (31.3238, 75.5820), (31.3220, 75.5920), (31.3269, 75.5785), (31.3260, 75.5760)
        ]
        RT002_wps = [
            (31.3260, 75.5760), (31.3400, 75.5660), (31.3560, 75.5580)
        ]
        RT003_wps = [
            (31.3269, 75.5785), (31.3238, 75.5820), (31.3010, 75.6200), (31.3150, 75.6350)
        ]
        RT004_wps = [
            (31.3260, 75.5760), (31.3180, 75.5860), (31.3220, 75.5920), (31.3300, 75.6000)
        ]
        corridor_defs = [
            ('RT001', 'Bus Stand – BMC – Model Town – Station (Loop)', RT001_wps,
             [('Jalandhar Bus Stand', 31.3260, 75.5760), ('BMC Chowk', 31.3238, 75.5820), ('Model Town Market', 31.3220, 75.5920), ('Jalandhar City Railway Station', 31.3269, 75.5785)]),
            ('RT002', 'Bus Stand – Maqsudan Corridor', RT002_wps,
             [('Jalandhar Bus Stand', 31.3260, 75.5760), ('Maqsudan Chowk', 31.3560, 75.5580)]),
            ('RT003', 'Railway Station – BMC – PAP – Industrial Area', RT003_wps,
             [('Jalandhar City Railway Station', 31.3269, 75.5785), ('PAP Chowk', 31.3010, 75.6200)]),
            ('RT004', 'Bus Stand – Model Town – Urban Estate', RT004_wps,
             [('Jalandhar Bus Stand', 31.3260, 75.5760), ('Model Town Market', 31.3220, 75.5920)]),
        ]

        # Fetch real bus stops (highway=bus_stop) and snap route stops to nearest real stops along corridors
        import geopandas as gpd
        # Build OSM-derived corridors
        for route_id, route_name, wpts, hub_labels in corridor_defs:
            local_stops, stop_ids = self._build_stops_from_corridor(G, route_name, wpts, spacing_m=spacing, hub_labels=hub_labels)
            # Query bus stops within city boundary
            # Use OSMnx pois if available; fallback to keeping generated points
            try:
                pois = ox.features_from_place(self.city_place, { 'highway': 'bus_stop' })
                busstops = pois.to_crs(ox.utils_geo.get_utm_crs(pois.unary_union.centroid.y, pois.unary_union.centroid.x))
                # Prepare corridor geometry
                path_nodes = self._route_nodes(G, wpts)
                line = self._route_linestring(G, path_nodes)
                line_proj = gpd.GeoSeries([line], crs='EPSG:4326').to_crs(busstops.crs).iloc[0]
                # Snap each generated stop to nearest real bus stop within 150m; otherwise keep as-is
                for sid in list(local_stops.keys()):
                    s = local_stops[sid]
                    pt_proj = gpd.GeoSeries([Point(s.lon, s.lat)], crs='EPSG:4326').to_crs(busstops.crs).iloc[0]
                    # filter candidate stops within 300m of corridor buffer
                    near = busstops[busstops.geometry.distance(line_proj) <= 300]
                    if not near.empty:
                        distances = near.geometry.distance(pt_proj)
                        idx_min = distances.idxmin()
                        if float(distances.loc[idx_min]) <= 150:  # within 150m
                            snapped = near.loc[idx_min].geometry
                            lat_sn, lon_sn = snapped.y, snapped.x
                            local_stops[sid] = BusStop(sid, s.name, lat_sn, lon_sn, s.is_major_hub)
            except Exception:
                pass

            for sid, stop in local_stops.items():
                if sid in self.stops:
                    sid = f"{sid}_X"
                    stop.stop_id = sid
                self.stops[sid] = stop

            self.routes[route_id] = Route(route_id, route_name, list(local_stops.keys()), base_frequency_mins=random.choice([8, 10, 12, 15]))

        # Upgrade exact hubs: replace nearest generated stop with canonical hub IDs
        for hub_id, (hub_name, hlat, hlon) in hubs.items():
            best_id, best_d = None, 1e9
            for sid, s in list(self.stops.items()):
                d = ox.distance.great_circle_vec(s.lat, s.lon, hlat, hlon)
                if d < best_d:
                    best_id, best_d = sid, d
            if best_id:
                bs = self.stops[best_id]
                self.stops[hub_id] = BusStop(hub_id, hub_name, bs.lat, bs.lon, True)
                for rid, route in self.routes.items():
                    route.stops = [hub_id if x == best_id else x for x in route.stops]
                if hub_id != best_id and best_id in self.stops:
                    del self.stops[best_id]

        # Fleet assignment scaled to route length
        self.buses = {}
        bus_count = 0
        for route_id, route in self.routes.items():
            nstops = len(route.stops)
            buses_for_route = max(8, nstops // 6)
            for _ in range(buses_for_route):
                bus_count += 1
                self.buses[f'BUS{bus_count:03d}'] = {
                    'primary_route': route_id,
                    'age_years': random.randint(1, 12),
                    'driver_type': random.choice(['fast', 'normal', 'slow']),
                    'condition': random.choice(['excellent', 'good', 'average', 'poor'])
                }

        print(f"[INFO] OSM-based network: {len(self.routes)} routes, {len(self.stops)} stops, {len(self.buses)} buses.")
    
    # ---------- Segment congestion, incidents, dwell, headway ----------
    def setup_segment_speed_profiles(self):
        """Precompute per-segment, per-hour congestion multipliers (>=1 means slower)."""
        self.segment_hour_multiplier: Dict[Tuple[str, str, str, int], float] = {}
        # Build for all routes and adjacent stops
        for route_id, route in getattr(self, 'routes', {}).items():
            for i in range(len(route.stops) - 1):
                s1 = route.stops[i]
                s2 = route.stops[i + 1]
                # Base segment difficulty from proximity to hubs
                hub_factor = (self.hub_congestion.get(s1, 1.0) + self.hub_congestion.get(s2, 1.0)) / 2.0
                for hour in range(0, 24):
                    traffic = self.traffic_multipliers.get(hour, 1.0)
                    # Inject per-segment randomness, slightly higher on peak
                    seg_noise = np.random.normal(1.0, 0.08) * (1.05 if hour in (8, 9, 18, 19) else 1.0)
                    multiplier = max(0.7, min(2.5, traffic * hub_factor * seg_noise))
                    self.segment_hour_multiplier[(route_id, s1, s2, hour)] = multiplier

    def _init_operational_state(self):
        """Initialize state containers for incidents and headway control."""
        self.incidents_by_date: Dict[str, list] = {}
        # headway control: nothing global needed beyond route headway minutes
        self.headway_control_enabled = True

    def generate_daily_incidents(self, date):
        """Generate incidents for the day that slow down certain segments.
        Stores list of dicts with keys: route_id, s1, s2, start, end, severity.
        """
        date_key = date.strftime('%Y-%m-%d')
        incidents = []
        rng = np.random.default_rng(abs(hash(date_key)) % (2**32))
        # Expected 2-5 incidents per day citywide
        num_inc = rng.integers(2, 6)
        for _ in range(int(num_inc)):
            route_id = rng.choice(list(self.routes.keys()))
            route = self.routes[route_id]
            if len(route.stops) < 2:
                continue
            idx = int(rng.integers(0, len(route.stops) - 1))
            s1 = route.stops[idx]
            s2 = route.stops[idx + 1]
            start_hour = int(rng.integers(6, 22))
            duration_h = float(rng.choice([0.5, 1, 1.5, 2, 3]))
            start = datetime.combine(date, datetime.min.time().replace(hour=start_hour)) + timedelta(minutes=int(rng.integers(0, 60)))
            end = start + timedelta(hours=duration_h)
            severity = float(rng.choice([1.2, 1.3, 1.5, 1.8, 2.2]))  # multiplier on slowdown
            incidents.append({'route_id': route_id, 's1': s1, 's2': s2, 'start': start, 'end': end, 'severity': severity, 'type': rng.choice(['accident', 'waterlogging', 'roadwork'])})
        self.incidents_by_date[date_key] = incidents

    def _incident_multiplier_for(self, route_id: str, s1: str, s2: str, t: datetime) -> Tuple[float, str]:
        date_key = t.strftime('%Y-%m-%d')
        for inc in self.incidents_by_date.get(date_key, []):
            if inc['route_id'] == route_id and inc['s1'] == s1 and inc['s2'] == s2 and inc['start'] <= t <= inc['end']:
                return inc['severity'], inc['type']
        return 1.0, None

    def _sample_board_alight(self, stop_id: str, t: datetime, weather: str, occupancy: int) -> Tuple[int, int]:
        """Sample boarding and alighting counts based on time, hub status, and weather."""
        is_hub = self.stops.get(stop_id, BusStop(stop_id, '', 0, 0)).is_major_hub
        hour = t.hour
        peak = 1.5 if hour in (8, 9, 18, 19) else (1.2 if 7 <= hour <= 21 else 0.6)
        weather_factor = 0.9 if weather in ('heavy_rain', 'fog') else (1.0 if weather in ('light_rain', 'cloudy') else 1.05)
        base = 4 if not is_hub else 10
        expected_board = max(0, int(np.random.poisson(base * peak * weather_factor)))
        # Alight fraction depends on occupancy
        alight_frac = min(0.35, 0.1 + (0.3 if is_hub else 0.2) * (occupancy / 60.0))
        expected_alight = int(np.random.binomial(occupancy, alight_frac))
        return expected_board, expected_alight

    def _compute_dwell_seconds(self, stop_id: str, board: int, alight: int, weather: str) -> int:
        """Compute dwell time seconds based on flows and conditions."""
        is_hub = self.stops.get(stop_id, BusStop(stop_id, '', 0, 0)).is_major_hub
        base = 15 if not is_hub else 25
        per_pax = 1.6 if is_hub else 1.2
        weather_penalty = 8 if weather in ('light_rain',) else 18 if weather in ('heavy_rain', 'fog') else 0
        dwell = base + int(per_pax * (board + alight)) + weather_penalty
        dwell = max(10, min(240, dwell))
        return dwell

    def _headway_hold_seconds(self, route_id: str, t: datetime, is_hub: bool) -> int:
        """Simple headway control: at hubs, hold to align departures to headway slots."""
        if not self.headway_control_enabled or not is_hub:
            return 0
        headway_min = max(6, self.routes[route_id].base_frequency_mins)
        minute_of_day = t.hour * 60 + t.minute
        next_slot_min = ((minute_of_day // headway_min) + 1) * headway_min
        hold = (next_slot_min - minute_of_day) * 60 - t.second
        if 0 < hold <= 180:
            return int(hold)
        return 0
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two coordinates"""
        R = 6371  # Earth's radius in km
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c
    
    def get_weather_condition(self, timestamp, lat: float = None, lon: float = None):
        """Get weather condition with spatial and hourly coherence.
        Falls back to seasonal sampling if lat/lon not provided.
        """
        month = timestamp.month
        conditions, base_probs = self._get_weather_base_probs(month)

        # If no spatial info, simple seasonal sample
        if lat is None or lon is None:
            return np.random.choice(conditions, p=base_probs)

        # Quantize to hour and grid cell
        hour_key = timestamp.strftime('%Y-%m-%d %H:00')
        ilat, ilon = self._weather_cell_indices(lat, lon)
        key = (hour_key, ilat, ilon)

        # If cached for this hour/cell, return
        if key in self._weather_cache:
            return self._weather_cache[key]

        # Check previous hour same cell for persistence
        prev_hour = (timestamp - timedelta(hours=1)).strftime('%Y-%m-%d %H:00')
        prev_key = (prev_hour, ilat, ilon)
        chosen: str
        if prev_key in self._weather_cache and random.random() < self.weather_hour_persistence:
            chosen = self._weather_cache[prev_key]
        else:
            # Sample from seasonal base probs, with occasional noise event skew
            if random.random() < self.weather_noise_prob:
                # Slightly skew towards rainy conditions during monsoon
                skew = np.array(base_probs, dtype=float)
                try:
                    lr_idx = conditions.index('light_rain')
                    hr_idx = conditions.index('heavy_rain')
                except ValueError:
                    lr_idx, hr_idx = 2, 3
                skew[lr_idx] *= 1.25
                skew[hr_idx] *= 1.2
                skew = skew / skew.sum()
                chosen = np.random.choice(conditions, p=skew)
            else:
                chosen = np.random.choice(conditions, p=base_probs)

        # Cache and return
        self._weather_cache[key] = chosen
        return chosen
    
    def add_gps_noise(self, lat, lon, noise_level=0.0001):
        """Add realistic GPS noise"""
        lat_noise = np.random.normal(0, noise_level)
        lon_noise = np.random.normal(0, noise_level)
        return lat + lat_noise, lon + lon_noise
    
    def simulate_bus_trip(self, bus_id, route_id, start_time, trip_duration_hours=2):
        """Simulate a complete bus trip with realistic GPS updates"""
        route = self.routes[route_id]
        bus_info = self.buses[bus_id]
        
        trip_data = []
        current_time = start_time
        end_time = start_time + timedelta(hours=trip_duration_hours)
        
        # Calculate trip parameters
        total_stops = len(route.stops)
        current_stop_idx = 0
        occupancy = random.randint(5, 30)
        
        # Base speed (km/h) adjusted by bus condition and driver
        base_speed = {
            'fast': 32, 'normal': 28, 'slow': 24
        }[bus_info['driver_type']]
        
        condition_factor = {
            'excellent': 1.1, 'good': 1.0, 'average': 0.9, 'poor': 0.8
        }[bus_info['condition']]
        
        base_speed *= condition_factor
        
        # Start trip
        while current_time < end_time and current_stop_idx < total_stops:
            current_stop_id = route.stops[current_stop_idx]
            current_stop = self.stops[current_stop_id]
            
            # Get weather condition (spatiotemporal coherence)
            weather = self.get_weather_condition(current_time, current_stop.lat, current_stop.lon)
            weather_factor = self.weather_conditions[weather]['speed_factor']
            
            # Get traffic factor
            hour = current_time.hour
            traffic_factor = self.traffic_multipliers.get(hour, 1.0)
            
            # Hub congestion
            hub_factor = self.hub_congestion.get(current_stop_id, 1.0)
            
            # Calculate current speed
            current_speed = base_speed * weather_factor / (traffic_factor * hub_factor)
            current_speed = max(5, current_speed)  # Minimum 5 km/h
            
            # Add some randomness
            current_speed *= np.random.normal(1.0, 0.15)
            current_speed = max(3, min(60, current_speed))  # Clamp between 3-60 km/h
            
            # Calculate to next stop
            if current_stop_idx < total_stops - 1:
                next_stop_id = route.stops[current_stop_idx + 1]
                next_stop = self.stops[next_stop_id]
                segment_distance = self.calculate_distance(
                    current_stop.lat, current_stop.lon,
                    next_stop.lat, next_stop.lon
                )
                # Segment-specific congestion multiplier
                seg_mult = self.segment_hour_multiplier.get((route_id, current_stop_id, next_stop_id, hour), 1.0)
                # Incident multiplier (>=1 slows down)
                inc_mult, inc_type = self._incident_multiplier_for(route_id, current_stop_id, next_stop_id, current_time)
                
                # ETA calculation (simplified)
                remaining_distance = 0
                for i in range(current_stop_idx + 1, total_stops):
                    if i < total_stops - 1:
                        stop1 = self.stops[route.stops[i]]
                        stop2 = self.stops[route.stops[i + 1]]
                        remaining_distance += self.calculate_distance(
                            stop1.lat, stop1.lon, stop2.lat, stop2.lon
                        )
                
                # Effective speed considering segment and incident
                effective_speed = max(3, current_speed / (seg_mult * inc_mult))
                eta_hours = remaining_distance / effective_speed
                predicted_eta_seconds = int(eta_hours * 3600)
                
                target_stop_id = route.stops[-1]  # Final destination
            else:
                segment_distance = 0
                predicted_eta_seconds = 0
                target_stop_id = current_stop_id
            
            # Add GPS noise to location
            noisy_lat, noisy_lon = self.add_gps_noise(current_stop.lat, current_stop.lon)
            
            # Generate record
            record = {
                'bus_id': bus_id,
                'route_id': route_id,
                'current_stop_id': current_stop_id,
                'target_stop_id': target_stop_id,
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'current_location_lat': round(noisy_lat, 6),
                'current_location_lon': round(noisy_lon, 6),
                'average_speed_last_segment': round(current_speed, 2),
                'segment_distance': round(segment_distance, 3),
                'weather_condition': weather,
                'day_of_week': current_time.weekday(),
                'hour_of_day': current_time.hour,
                'predicted_eta_seconds': predicted_eta_seconds,
                'confidence_score': round(np.random.normal(0.85, 0.1), 2),
                'delay_indicator': 'delayed' if current_speed < base_speed * 0.7 else 'on_time',
                'passenger_load': occupancy,
                'stop_dwell_time': 0,
                'traffic_level': 'high' if traffic_factor > 1.5 else 'medium' if traffic_factor > 1.2 else 'low'
            }
            # Add incident metadata if applicable
            if current_stop_idx < total_stops - 1 and inc_mult > 1.0:
                record['incident_type'] = inc_type
                record['incident_severity'] = inc_mult
            
            # Clamp confidence score
            record['confidence_score'] = max(0.1, min(1.0, record['confidence_score']))
            
            trip_data.append(record)
            
            # Move to next update (every 15-30 seconds)
            update_interval = random.randint(15, 30)
            current_time += timedelta(seconds=update_interval)
            
            # Simulate reaching next stop
            if segment_distance > 0:
                # Time to next stop using effective speed
                time_to_next_stop = (segment_distance / max(3, current_speed / (seg_mult * inc_mult))) * 3600
                # Boarding/alighting at the NEXT stop determines dwell there
                board, alight = self._sample_board_alight(next_stop_id, current_time + timedelta(seconds=time_to_next_stop), weather, occupancy)
                occupancy = max(0, min(60, occupancy + board - alight))
                dwell_time = self._compute_dwell_seconds(next_stop_id, board, alight, weather)
                # Simple headway control at hubs
                dwell_time += self._headway_hold_seconds(route_id, current_time + timedelta(seconds=time_to_next_stop + dwell_time), self.stops[next_stop_id].is_major_hub)
                # Apply jump to next stop occasionally
                if random.random() < 0.35:  # 35% chance to simulate traveling between stops
                    current_time += timedelta(seconds=time_to_next_stop + dwell_time)
                    current_stop_idx += 1
                    record['stop_dwell_time'] = dwell_time
        
        return trip_data
    
    def generate_daily_operations(self, date, num_trips_per_route=6):
        """Generate full day operations for all routes"""
        all_data = []
        # Generate incidents for the day once
        self.generate_daily_incidents(date)
        
        for route_id in self.routes.keys():
            route = self.routes[route_id]
            available_buses = [bus_id for bus_id, info in self.buses.items() 
                             if info['primary_route'] == route_id]
            
            # Generate trips throughout the day
            for trip_num in range(num_trips_per_route):
                # Vary start times throughout operating hours
                start_hour = random.randint(self.operating_hours[0], self.operating_hours[1] - 3)
                start_minute = random.randint(0, 59)
                
                start_time = datetime.combine(date, datetime.min.time().replace(
                    hour=start_hour, minute=start_minute
                ))
                
                # Select a bus
                bus_id = random.choice(available_buses)
                # Append schedule metadata
                self._schedule_rows.append({
                    'service_date': date.strftime('%Y-%m-%d'),
                    'route_id': route_id,
                    'trip_number': trip_num + 1,
                    'bus_id': bus_id,
                    'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'planned_headway_mins': self.routes[route_id].base_frequency_mins,
                    'num_stops': len(self.routes[route_id].stops),
                })
                
                # Simulate trip
                trip_data = self.simulate_bus_trip(bus_id, route_id, start_time)
                all_data.extend(trip_data)
        
        return all_data
    
    def generate_dataset(self, start_date_str='2024-09-01', days=7, output_file='kolkata_bus_eta_data.csv'):
        """Generate complete dataset"""
        print(f"Generating realistic bus ETA dataset for {days} days...")
        
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        all_data = []
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            print(f"Generating data for {current_date}...")
            
            # Adjust trips based on day of week
            if current_date.weekday() < 5:  # Weekday
                trips_per_route = 8
            else:  # Weekend
                trips_per_route = 5
            
            daily_data = self.generate_daily_operations(current_date, trips_per_route)
            all_data.extend(daily_data)
            
            print(f"  Generated {len(daily_data)} records for {current_date}")
        
        # Convert to DataFrame and save
        df = pd.DataFrame(all_data)
        
        # Sort by timestamp
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp_dt').drop('timestamp_dt', axis=1)
        
        # Save to CSV
        df.to_csv(output_file, index=False)

        # Export metadata CSVs alongside output_file
        base_dir = os.path.dirname(output_file) or '.'
        # Stops metadata
        stops_rows = [{
            'stop_id': sid,
            'name': s.name,
            'lat': s.lat,
            'lon': s.lon,
            'is_major_hub': int(s.is_major_hub)
        } for sid, s in self.stops.items()]
        pd.DataFrame(stops_rows).to_csv(os.path.join(base_dir, 'stops.csv'), index=False)
        # Routes metadata
        routes_rows = [{
            'route_id': rid,
            'name': r.name,
            'stops_sequence': json.dumps(r.stops),
            'base_frequency_mins': r.base_frequency_mins
        } for rid, r in self.routes.items()]
        pd.DataFrame(routes_rows).to_csv(os.path.join(base_dir, 'routes.csv'), index=False)
        # Segments metadata
        segments = []
        for rid, r in self.routes.items():
            for i in range(len(r.stops) - 1):
                s1 = self.stops[r.stops[i]]
                s2 = self.stops[r.stops[i+1]]
                segments.append({
                    'route_id': rid,
                    'from_stop_id': s1.stop_id,
                    'to_stop_id': s2.stop_id,
                    'segment_distance_km': round(self.calculate_distance(s1.lat, s1.lon, s2.lat, s2.lon), 3)
                })
        pd.DataFrame(segments).to_csv(os.path.join(base_dir, 'segments.csv'), index=False)
        # Buses metadata
        buses_rows = [{
            'bus_id': bid,
            'primary_route': info['primary_route'],
            'age_years': info['age_years'],
            'driver_type': info['driver_type'],
            'condition': info['condition']
        } for bid, info in self.buses.items()]
        pd.DataFrame(buses_rows).to_csv(os.path.join(base_dir, 'buses.csv'), index=False)
        # Schedule metadata
        if self._schedule_rows:
            pd.DataFrame(self._schedule_rows).to_csv(os.path.join(base_dir, 'schedule.csv'), index=False)
        
        print(f"\nDataset generated successfully!")
        print(f"Total records: {len(df)}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Unique buses: {df['bus_id'].nunique()}")
        print(f"Unique routes: {df['route_id'].nunique()}")
        print(f"File saved as: {output_file}")
        
        # Print sample statistics
        print(f"\nSample Statistics:")
        print(f"Average speed: {df['average_speed_last_segment'].mean():.2f} km/h")
        print(f"Weather distribution:")
        print(df['weather_condition'].value_counts().head())
        
        return df

# Generate the dataset
if __name__ == "__main__":
    generator = RealisticBusETAGenerator()
    
    # Generate 7 days of data (will create ~50,000-100,000 records)
    # Each bus updates every 15-30 seconds during active trips
    dataset = generator.generate_dataset(
        start_date_str='2024-09-01',
        days=30,
        output_file='jalandhar_bus_eta_realistic.csv'
    )
    
    print("\nFirst 5 records:")
    print(dataset.head().to_string())
    
    print(f"\nDataset shape: {dataset.shape}")
    print(f"Columns: {list(dataset.columns)}")
    
    