import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import math

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_bus_eta_dataset(num_rows=200, start_date="2024-01-01"):
    """
    Generate realistic bus ETA dataset with all specified features
    """
    
    # Define base parameters
    routes = ["ROUTE1", "ROUTE2", "ROUTE3", "ROUTE4", "ROUTE5"]
    buses = [f"BUS{str(i).zfill(3)}" for i in range(101, 121)]  # BUS101-BUS120
    
    # Define stops per route (realistic city layout)
    route_stops = {
        "ROUTE1": [f"STOP{i}" for i in range(1, 11)],      # 10 stops
        "ROUTE2": [f"STOP{i}" for i in range(11, 21)],     # 10 stops  
        "ROUTE3": [f"STOP{i}" for i in range(21, 31)],     # 10 stops
        "ROUTE4": [f"STOP{i}" for i in range(31, 41)],     # 10 stops
        "ROUTE5": [f"STOP{i}" for i in range(41, 51)]      # 10 stops
    }
    
    # Kolkata coordinates as base (can be adjusted for any city)
    base_lat, base_lon = 22.5726, 88.3639
    
    # Generate stop coordinates (distributed around the city)
    stop_coordinates = {}
    for route, stops in route_stops.items():
        route_offset = hash(route) % 100 / 10000  # Small route-specific offset
        for i, stop in enumerate(stops):
            # Create realistic stop distribution
            lat_offset = (i * 0.005) + random.uniform(-0.003, 0.003)
            lon_offset = route_offset + random.uniform(-0.003, 0.003)
            stop_coordinates[stop] = (base_lat + lat_offset, base_lon + lon_offset)
    
    # Define holidays and events (Indian holidays as example)
    events_2024 = [
        "2024-01-26",  # Republic Day
        "2024-03-08",  # Holi
        "2024-04-14",  # Bengali New Year
        "2024-08-15",  # Independence Day
        "2024-10-02",  # Gandhi Jayanti
        "2024-10-24",  # Diwali
        "2024-12-25"   # Christmas
    ]
    
    dataset = []
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    
    for _ in range(num_rows):
        # Random date within reasonable range (adjust for scaling)
        random_days = random.randint(0, 60)  # 2 months for sample, scale to 365 for full year
        trip_date = current_date + timedelta(days=random_days)
        
        # Trip basic info
        route_id = random.choice(routes)
        stops = route_stops[route_id]
        stop_id = random.choice(stops)
        bus_id = random.choice(buses)
        trip_id = f"TRIP_{trip_date.strftime('%Y%m%d')}_{route_id}_{bus_id}_{random.randint(1,20)}"
        
        # Date/time features
        day_of_week = trip_date.weekday()  # 0=Monday, 6=Sunday
        season = get_season(trip_date.month)
        
        # Time of day (affects traffic and occupancy)
        hour = random.choices(
            range(24), 
            weights=[2,1,1,1,2,4,8,12,15,10,8,8,8,8,10,15,18,12,8,6,4,3,2,2]  # Peak hours weighted
        )[0]
        minute = random.randint(0, 59)
        scheduled_time = trip_date.replace(hour=hour, minute=minute, second=0)
        
        # Traffic level based on time and day
        traffic_level = get_traffic_level(hour, day_of_week)
        
        # Weather (season-dependent)
        weather = get_weather(season, trip_date)
        
        # Event flag
        event_flag = 1 if trip_date.strftime("%Y-%m-%d") in events_2024 else 0
        
        # Base scheduled arrival (15-45 minutes between stops on average)
        base_interval = random.randint(15, 45)
        scheduled_arrival = scheduled_time + timedelta(minutes=base_interval)
        
        # Calculate delay based on multiple factors
        base_delay = calculate_delay(traffic_level, weather, event_flag, day_of_week, hour, season)
        delay_minutes = max(0, base_delay)  # No negative delays
        
        # Actual arrival
        actual_arrival = scheduled_arrival + timedelta(minutes=delay_minutes)
        
        # Location data
        lat, lon = stop_coordinates[stop_id]
        
        # Add some GPS noise
        lat += random.uniform(-0.0001, 0.0001)
        lon += random.uniform(-0.0001, 0.0001)
        
        # Speed calculation (affected by traffic and weather)
        base_speed = 25  # km/h average city speed
        speed_factor = get_speed_factor(traffic_level, weather)
        speed = max(5, base_speed * speed_factor + random.uniform(-3, 3))
        
        # Distance to stop (varies by route position)
        distance_to_stop = random.uniform(0.1, 2.5)  # km
        
        # Occupancy (affected by time, day, events)
        occupancy = calculate_occupancy(hour, day_of_week, event_flag, season)
        
        # Create record
        record = {
            'trip_id': trip_id,
            'bus_id': bus_id,
            'route_id': route_id,
            'date': trip_date.strftime('%Y-%m-%d'),
            'day_of_week': day_of_week,
            'season': season,
            'stop_id': stop_id,
            'scheduled_arrival': scheduled_arrival.strftime('%Y-%m-%d %H:%M:%S'),
            'actual_arrival': actual_arrival.strftime('%Y-%m-%d %H:%M:%S'),
            'delay_minutes': round(delay_minutes, 2),
            'lat': round(lat, 6),
            'lon': round(lon, 6),
            'speed': round(speed, 2),
            'distance_to_stop': round(distance_to_stop, 2),
            'traffic_level': traffic_level,
            'weather': weather,
            'event_flag': event_flag,
            'occupancy': occupancy
        }
        
        dataset.append(record)
    
    return pd.DataFrame(dataset)

def get_season(month):
    """Determine season based on month (Indian seasons)"""
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "summer"  
    elif month in [6, 7, 8, 9]:
        return "monsoon"
    else:
        return "autumn"

def get_traffic_level(hour, day_of_week):
    """Calculate traffic level based on time and day"""
    # Weekend traffic is generally lower
    if day_of_week >= 5:  # Saturday, Sunday
        if 10 <= hour <= 22:
            return random.choices(["low", "medium", "high"], weights=[30, 50, 20])[0]
        else:
            return "low"
    else:  # Weekdays
        if hour in [7, 8, 9, 17, 18, 19]:  # Peak hours
            return random.choices(["medium", "high"], weights=[30, 70])[0]
        elif hour in [10, 11, 12, 13, 14, 15, 16, 20]:
            return random.choices(["low", "medium", "high"], weights=[20, 60, 20])[0]
        else:
            return "low"

def get_weather(season, date):
    """Generate weather based on season"""
    if season == "monsoon":
        return random.choices(
            ["clear", "rain", "fog", "cloudy"], 
            weights=[20, 50, 15, 15]
        )[0]
    elif season == "winter":
        return random.choices(
            ["clear", "fog", "cloudy"], 
            weights=[60, 25, 15]
        )[0]
    elif season == "summer":
        return random.choices(
            ["clear", "cloudy"], 
            weights=[80, 20]
        )[0]
    else:
        return random.choices(
            ["clear", "cloudy"], 
            weights=[70, 30]
        )[0]

def calculate_delay(traffic_level, weather, event_flag, day_of_week, hour, season):
    """Calculate realistic delay based on multiple factors"""
    base_delay = 0
    
    # Traffic impact
    traffic_delays = {"low": 0, "medium": 2, "high": 5}
    base_delay += traffic_delays[traffic_level]
    
    # Weather impact
    weather_delays = {"clear": 0, "cloudy": 0.5, "rain": 4, "fog": 3}
    base_delay += weather_delays[weather]
    
    # Event impact
    if event_flag:
        base_delay += random.uniform(3, 8)
    
    # Peak hour impact
    if hour in [8, 9, 18, 19]:
        base_delay += random.uniform(1, 4)
    
    # Weekend vs weekday
    if day_of_week >= 5:
        base_delay *= 0.8  # Generally less delay on weekends
    
    # Seasonal impact
    seasonal_multipliers = {"summer": 1.1, "monsoon": 1.3, "winter": 1.0, "autumn": 1.0}
    base_delay *= seasonal_multipliers[season]
    
    # Add random variation
    base_delay += random.uniform(-1, 3)
    
    return base_delay

def get_speed_factor(traffic_level, weather):
    """Calculate speed reduction factor"""
    traffic_factors = {"low": 1.0, "medium": 0.8, "high": 0.6}
    weather_factors = {"clear": 1.0, "cloudy": 0.95, "rain": 0.7, "fog": 0.65}
    
    return traffic_factors[traffic_level] * weather_factors[weather]

def calculate_occupancy(hour, day_of_week, event_flag, season):
    """Calculate bus occupancy percentage"""
    base_occupancy = 40  # Base 40% occupancy
    
    # Peak hours
    if hour in [7, 8, 9, 17, 18, 19]:
        base_occupancy += random.uniform(20, 40)
    elif hour in [10, 11, 12, 13, 14, 15, 16]:
        base_occupancy += random.uniform(5, 20)
    
    # Weekend adjustment
    if day_of_week >= 5:
        base_occupancy *= random.uniform(0.6, 1.2)  # More variable on weekends
    
    # Event impact
    if event_flag:
        base_occupancy += random.uniform(10, 30)
    
    # Seasonal adjustment
    seasonal_multipliers = {"summer": 0.9, "monsoon": 1.1, "winter": 1.0, "autumn": 1.0}
    base_occupancy *= seasonal_multipliers[season]
    
    # Add randomness and cap at 100%
    occupancy = base_occupancy + random.uniform(-10, 15)
    return max(10, min(100, round(occupancy, 1)))

# Generate the dataset
print("Generating Bus ETA Dataset...")
df = generate_bus_eta_dataset(num_rows=200)

# Display basic stats
print(f"\nDataset generated with {len(df)} rows")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Routes: {sorted(df['route_id'].unique())}")
print(f"Average delay: {df['delay_minutes'].mean():.2f} minutes")
print(f"Weather distribution:\n{df['weather'].value_counts()}")
print(f"Traffic level distribution:\n{df['traffic_level'].value_counts()}")

# Save to CSV
df.to_csv("dummy_eta_dataset.csv", index=False)
print("\nDataset saved as 'dummy_eta_dataset.csv'")

# Display first few rows
print("\nFirst 5 rows:")
print(df.head())