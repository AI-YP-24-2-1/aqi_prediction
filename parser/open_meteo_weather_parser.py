import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry

from datetime import datetime, timedelta
import time
import json
import os.path


class WeatherParser:
    def __init__(self, startdate, enddate):
        self.startdate = startdate
        self.enddate = enddate
    

    def open_json(self, filename: str) -> list[dict]:
        try:
            with open(filename) as file:
                data = json.load(file)
        except Exception as e:
            data = []
            print(f'JSON file: {filename} is empty\n{e}')
        
        return data
    

    def get_city_weather(self, lat: float, lon: float):
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)

        # Make sure all required weather variables are listed here
        # The order of variables in hourly or daily is important to assign them correctly below
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": self.startdate,
            "end_date": self.enddate,
            "hourly": ["temperature_2m", "relative_humidity_2m", "rain", "snowfall", "snow_depth", "surface_pressure", "cloud_cover", "wind_speed_10m", "wind_direction_10m"],
            "timezone": "Europe/Moscow"
        }
        response = openmeteo.weather_api(url, params=params)[0]

        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
        hourly_rain = hourly.Variables(2).ValuesAsNumpy()
        hourly_snowfall = hourly.Variables(3).ValuesAsNumpy()
        hourly_snow_depth = hourly.Variables(4).ValuesAsNumpy()
        hourly_surface_pressure = hourly.Variables(5).ValuesAsNumpy()
        hourly_cloud_cover = hourly.Variables(6).ValuesAsNumpy()
        hourly_wind_speed_10m = hourly.Variables(7).ValuesAsNumpy()
        hourly_wind_direction_10m = hourly.Variables(8).ValuesAsNumpy()

        hourly_data = {"date": pd.date_range(
            start = pd.to_datetime(hourly.Time(), unit = "s", utc = True) + pd.Timedelta(hours=3),
            end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True) + pd.Timedelta(hours=3),
            freq = pd.Timedelta(seconds = hourly.Interval()),
            inclusive = "left"
        )}
        hourly_data["temperature_2m"] = hourly_temperature_2m
        hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
        hourly_data["rain"] = hourly_rain
        hourly_data["snowfall"] = hourly_snowfall
        hourly_data["snow_depth"] = hourly_snow_depth
        hourly_data["surface_pressure"] = hourly_surface_pressure
        hourly_data["cloud_cover"] = hourly_cloud_cover
        hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
        hourly_data["wind_direction_10m"] = hourly_wind_direction_10m

        return hourly_data
    
    
    def save_to_csv(self, hourly_data, filename):
        hourly_dataframe = pd.DataFrame(data = hourly_data)
        hourly_dataframe.to_csv(filename, index=False)

    
    def get_weather(self):
        data = self.open_json('cfo.list.json')

        for i, city in enumerate(data):
            lat, lon, city_name, city_id = round(city['coord']['lat'], 2), round(city['coord']['lon'], 2), city['name'], city['id']

            if not os.path.isdir('weather_by_city'):
                os.makedirs('weather_by_city')

            if os.path.isfile(f'weather_by_city/{self.startdate}_{self.enddate}_{city_id}.csv'):
                 continue
            
            j = 1
            while j < 3:
                try:
                    city_weather = self.get_city_weather(lat, lon)
                    self.save_to_csv(city_weather, f'weather_by_city/{self.startdate}_{self.enddate}_{city_id}.csv')
                    break
                except Exception as e:
                    city_weather = pd.DataFrame(data = [])
                    delay = (e.args[0])['reason'].split()[0] 

                    if delay == 'Minutely':
                        timeout = 60 - datetime.now().second + 1
                        print(f'{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}: Exceeded {delay}, waiting {timeout} seconds')
                        time.sleep(timeout)
                    elif delay == 'Hourly':
                        new_dt = (datetime.now() + timedelta(hours=1))
                        timeout = (datetime(new_dt.year, new_dt.month, new_dt.day, new_dt.hour, 0, 0) - datetime.now()).total_seconds() + 1
                        m = int(timeout//60)
                        s = int(timeout - m * 60)
                        print(f'{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}: Exceeded {delay}, waiting {m} minutes {s} seconds')
                        time.sleep(timeout)
                    elif delay == 'Daily':
                        new_dt = (datetime.now() + timedelta(days=1))
                        timeout = (datetime(new_dt.year, new_dt.month, new_dt.day, 0, 0, 0) - datetime.now()).total_seconds() + 1
                        h = int(timeout//3600)
                        m = int((timeout - (h * 3600))//60)
                        s = int(timeout - h * 3600 - m * 60)
                        print(f'{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}: Exceeded {delay}, waiting {h} hours {m} minutes {s} seconds')
                        time.sleep(timeout)
                    else:
                        print(e)
                    
                    j += 1

            print(f'{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}: Attempt: {j}/3 {i+1}/{len(data)} City: {city_name} City_id: {city['id']} rows: {len(city_weather['date']) if len(city_weather) != 0 else 0}')


if __name__ == '__main__':
    weather_parser = WeatherParser('2022-01-01', '2024-11-01')
    weather_parser.get_weather()
