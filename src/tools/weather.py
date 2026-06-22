import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog", 51: "light drizzle", 53: "moderate drizzle",
    55: "dense drizzle", 61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow", 80: "rain showers",
    81: "moderate rain showers", 82: "violent rain showers", 95: "thunderstorm",
}

GREEK_CITY_ALIASES = {
    "αθήνα": "Athens",
    "θεσσαλονίκη": "Thessaloniki",
    "πάτρα": "Patras",
    "ηράκλειο": "Heraklion",
    "λάρισα": "Larissa",
    "βόλος": "Volos",
    "ιωάννινα": "Ioannina",
    "καβάλα": "Kavala",
    "χανιά": "Chania",
    "ρόδος": "Rhodes",
    "κρήτη": "Crete",
    "κέρκυρα": "Corfu",
    "αττική": "Attica",
    "πελοπόννησος": "Peloponnese",
    "θεσσαλία": "Thessaly",
    "κεντρική μακεδονία": "Central Macedonia",
}


def _geocode(city: str):
    resp = requests.get(GEOCODE_URL, params={"name": city, "count": 1}, timeout=10)
    resp.raise_for_status()
    results = resp.json().get("results")
    if not results:
        return None
    r = results[0]
    return r["latitude"], r["longitude"], r.get("name", city), r.get("country", "")


def get_weather(city: str) -> str:
    city = GREEK_CITY_ALIASES.get(city.strip().lower(), city)
    geo = _geocode(city)
    if geo is None:
        return f"I couldn't find a location called '{city}'. Could you check the spelling or be more specific?"

    lat, lon, resolved_name, country = geo
    resp = requests.get(
        FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,weather_code,wind_speed_10m,cloud_cover",
            "daily": "temperature_2m_max,temperature_2m_min,weather_code,sunshine_duration",
            "timezone": "auto",
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    current = data.get("current", {})
    temp = current.get("temperature_2m")
    code = current.get("weather_code")
    wind = current.get("wind_speed_10m")
    cloud_cover = current.get("cloud_cover")
    condition = WEATHER_CODES.get(code, "unknown conditions")

    daily = data.get("daily", {})
    today_max = daily.get("temperature_2m_max", [None])[0]
    today_min = daily.get("temperature_2m_min", [None])[0]
    sunshine_seconds = daily.get("sunshine_duration", [None])[0]
    sunshine_hours = round(sunshine_seconds / 3600, 1) if sunshine_seconds is not None else None

    location_label = f"{resolved_name}, {country}" if country else resolved_name
    return (
        f"Weather in {location_label}: currently {temp}°C, {condition}, "
        f"wind {wind} km/h, cloud cover {cloud_cover}%. "
        f"Today's range: {today_min}°C to {today_max}°C, "
        f"sunshine duration today: {sunshine_hours} hours "
        f"(relevant for estimating solar production)."
    )