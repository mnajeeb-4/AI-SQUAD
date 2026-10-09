"""
Weather Tool
------------
Implements the Hybrid Weather System: Local JSON cache + Open-Meteo API fallback.
"""

import json
import httpx
from pathlib import Path

# Path to the weather JSON file (located in the backend root folder)
WEATHER_DATA_PATH = Path(__file__).parent.parent.parent.parent / "weather_data.json"


def get_weather_from_cache(city: str) -> str | None:
    """
    Level 1: Check the local JSON file for the city's weather.
    Returns None if the city is not found in the cache.
    """
    if not WEATHER_DATA_PATH.exists():
        return None

    with open(WEATHER_DATA_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    city_data = data.get(city.title())  # Ensure title case (e.g., "Karachi")
    if city_data:
        return f"Current weather in {city}: {city_data['temp']}°C, {city_data['condition']}."
    
    return None


def get_weather_from_api(city: str) -> str:
    """
    Level 2: Fetch real-time weather from Open-Meteo API.
    First, get the latitude and longitude for the city, then get the weather.
    """
    # Step 1: Geocoding (City name to Lat/Lon)
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
    geo_response = httpx.get(geo_url)
    
    if geo_response.status_code != 200 or not geo_response.json().get("results"):
        return f"Sorry, I could not find the location for '{city}'."
    
    location = geo_response.json()["results"][0]
    lat, lon = location["latitude"], location["longitude"]

    # Step 2: Fetch Weather
    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    weather_response = httpx.get(weather_url)

    if weather_response.status_code != 200:
        return f"Sorry, I could not fetch the weather for '{city}' right now."

    weather_data = weather_response.json()["current_weather"]
    temp = weather_data["temperature"]
    wind_speed = weather_data["windspeed"]

    return f"Current weather in {city}: {temp}°C with a wind speed of {wind_speed} km/h."