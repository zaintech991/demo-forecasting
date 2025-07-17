"""
Live Weather Service for getting current weather conditions from OpenWeatherMap API
"""

import asyncio
import aiohttp
import os
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()

class LiveWeatherService:
    def __init__(self):
        # Try both WeatherAPI and OpenWeatherMap
        self.api_key = os.getenv('WEATHER_API_KEY') or os.getenv('OPENWEATHER_API_KEY')
        
        # Use WeatherAPI if we have WEATHER_API_KEY, otherwise OpenWeatherMap
        if os.getenv('WEATHER_API_KEY'):
            self.base_url = "https://api.weatherapi.com/v1/current.json"
            self.service_type = "weatherapi"
        else:
            self.base_url = "https://api.openweathermap.org/data/2.5/weather"
            self.service_type = "openweathermap"
        
        # Mapping of city_id to actual US city names
        self.city_mapping = {
            '1': 'New York,NY,US',
            '2': 'Chicago,IL,US', 
            '3': 'Houston,TX,US',
            '8': 'Dallas,TX,US',
            '10': 'Austin,TX,US',
            '11': 'San Antonio,TX,US',
            '12': 'Fort Worth,TX,US',
            '13': 'Columbus,OH,US',
            '14': 'Charlotte,NC,US',
            '15': 'Indianapolis,IN,US',
            '16': 'Seattle,WA,US',
            '17': 'Denver,CO,US',
            '4': 'Phoenix,AZ,US',
            '5': 'Philadelphia,PA,US',
            '6': 'San Diego,CA,US',
            '7': 'San Jose,CA,US',
            '9': 'Jacksonville,FL,US',
            '18': 'Boston,MA,US'
        }
    
    async def get_current_weather(self, city_id: str) -> Optional[Dict]:
        """
        Get current weather for a city using OpenWeatherMap API
        """
        if not self.api_key:
            return None
            
        city_name = self.city_mapping.get(city_id)
        if not city_name:
            return None
            
        try:
            async with aiohttp.ClientSession() as session:
                if self.service_type == "weatherapi":
                    # WeatherAPI.com format
                    params = {
                        'key': self.api_key,
                        'q': city_name,
                        'aqi': 'no'
                    }
                else:
                    # OpenWeatherMap format
                    params = {
                        'q': city_name,
                        'appid': self.api_key,
                        'units': 'imperial'  # Fahrenheit
                    }
                
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if self.service_type == "weatherapi":
                            # WeatherAPI.com response format
                            current = data['current']
                            return {
                                'temperature': current['temp_f'],
                                'humidity': current['humidity'],
                                'precipitation': current.get('precip_mm', 0.0),
                                'wind_level': current['wind_mph'],
                                'weather_category': self._categorize_weather(
                                    current['temp_f'],
                                    current.get('precip_mm', 0.0),
                                    current['humidity']
                                ),
                                'description': current['condition']['text'],
                                'city_name': city_name.split(',')[0]
                            }
                        else:
                            # OpenWeatherMap response format
                            return {
                                'temperature': data['main']['temp'],
                                'humidity': data['main']['humidity'],
                                'precipitation': data.get('rain', {}).get('1h', 0.0),
                                'wind_level': data['wind']['speed'],
                                'weather_category': self._categorize_weather(
                                    data['main']['temp'],
                                    data.get('rain', {}).get('1h', 0.0),
                                    data['main']['humidity']
                                ),
                                'description': data['weather'][0]['description'],
                                'city_name': city_name.split(',')[0]
                            }
                    else:
                        error_text = await response.text()
                        if response.status == 401:
                            print(f"{self.service_type.title()} API key is invalid or not activated. Status: {response.status}")
                        else:
                            print(f"{self.service_type.title()} API error for city {city_id}: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            print(f"Error fetching weather for city {city_id}: {e}")
            return None
    
    def _categorize_weather(self, temperature: float, precipitation: float, humidity: float) -> str:
        """
        Categorize weather conditions based on temperature, precipitation, and humidity
        """
        if precipitation > 0.2:  # More than 0.2mm/hour
            return "rainy"
        elif precipitation < 0.05:
            return "dry"
        elif temperature < 50.0:  # Below 50°F
            return "cold"
        elif temperature > 80.0:  # Above 80°F
            return "warm"
        else:
            return "mild"
    
    async def get_weather_for_cities(self, city_ids: list) -> Dict[str, Dict]:
        """
        Get weather for multiple cities concurrently
        """
        tasks = [self.get_current_weather(city_id) for city_id in city_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        weather_data = {}
        for city_id, result in zip(city_ids, results):
            if isinstance(result, dict) and result:
                weather_data[city_id] = result
            else:
                # Fallback to None if API fails
                weather_data[city_id] = None
                
        return weather_data

# Global instance
live_weather_service = LiveWeatherService() 