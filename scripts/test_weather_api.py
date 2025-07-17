#!/usr/bin/env python3
"""
Weather API Diagnostic Tool
Tests OpenWeatherMap API integration and provides troubleshooting information
"""

import asyncio
import aiohttp
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_weather_api():
    """Test OpenWeatherMap API with comprehensive diagnostics"""
    
    print("=== Weather API Diagnostic Tool ===\n")
    
    # Check API keys
    weather_api_key = os.getenv('WEATHER_API_KEY')
    openweather_key = os.getenv('OPENWEATHER_API_KEY')
    
    print(f"1. API Key Status:")
    if weather_api_key:
        print(f"   ✓ WeatherAPI key found")
        print(f"   ✓ Length: {len(weather_api_key)} characters")
        print(f"   ✓ Starts with: {weather_api_key[:8]}...")
        api_key = weather_api_key
        service_type = "weatherapi"
        base_url = "https://api.weatherapi.com/v1/current.json"
    elif openweather_key:
        print(f"   ✓ OpenWeatherMap key found")
        print(f"   ✓ Length: {len(openweather_key)} characters")
        print(f"   ✓ Starts with: {openweather_key[:8]}...")
        if len(openweather_key) != 32:
            print(f"   ⚠️  Warning: OpenWeatherMap key should be 32 characters, got {len(openweather_key)}")
        api_key = openweather_key
        service_type = "openweathermap"
        base_url = "https://api.openweathermap.org/data/2.5/weather"
    else:
        print(f"   ❌ No API key found in environment")
        print(f"   💡 Add WEATHER_API_KEY=your_key_here or OPENWEATHER_API_KEY=your_key_here to .env file")
        return
    
    print(f"\n2. Testing API Connection:")
    
    # Test cities
    test_cities = [
        ('New York,NY,US', 'New York'),
        ('Chicago,IL,US', 'Chicago'),
        ('Los Angeles,CA,US', 'Los Angeles')
    ]
    
    for city_query, city_name in test_cities:
        print(f"\n   Testing {city_name}...")
        
        try:
            async with aiohttp.ClientSession() as session:
                if service_type == "weatherapi":
                    params = {
                        'key': api_key,
                        'q': city_query,
                        'aqi': 'no'
                    }
                else:
                    params = {
                        'q': city_query,
                        'appid': api_key,
                        'units': 'imperial'
                    }
                
                async with session.get(base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if service_type == "weatherapi":
                            current = data['current']
                            print(f"   ✓ Success: {current['temp_f']:.1f}°F, {current['condition']['text']}")
                        else:
                            print(f"   ✓ Success: {data['main']['temp']:.1f}°F, {data['weather'][0]['description']}")
                    else:
                        error_text = await response.text()
                        print(f"   ❌ Error {response.status}: {error_text}")
                        
                        if response.status == 401:
                            print(f"   💡 API key is invalid or not activated")
                            if service_type == "weatherapi":
                                print(f"   💡 Get a new key at: https://www.weatherapi.com/")
                            else:
                                print(f"   💡 New keys take 10-60 minutes to activate")
                                print(f"   💡 Get a new key at: https://openweathermap.org/api")
                        elif response.status == 429:
                            print(f"   💡 Rate limit exceeded")
                        elif response.status == 404:
                            print(f"   💡 City not found")
                        
        except Exception as e:
            print(f"   ❌ Connection error: {e}")
    
    print(f"\n3. Environment File Check:")
    env_files = ['.env', '.env.local', '.env.production']
    for env_file in env_files:
        if os.path.exists(env_file):
            print(f"   ✓ Found: {env_file}")
            with open(env_file, 'r') as f:
                content = f.read()
                if 'WEATHER_API_KEY' in content:
                    print(f"   ✓ Contains WEATHER_API_KEY")
                elif 'OPENWEATHER_API_KEY' in content:
                    print(f"   ✓ Contains OPENWEATHER_API_KEY")
                else:
                    print(f"   ❌ No weather API key found")
        else:
            print(f"   - Not found: {env_file}")
    
    print(f"\n4. Recommendations:")
    print(f"   • WeatherAPI: Get a new key from https://www.weatherapi.com/ (instant activation)")
    print(f"   • OpenWeatherMap: Get a new key from https://openweathermap.org/api (10-60 min activation)")
    print(f"   • If still not working: Check your .env file location and syntax")
    print(f"   • System will fall back to database weather data if API fails")

if __name__ == "__main__":
    asyncio.run(test_weather_api()) 