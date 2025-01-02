import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry

from datetime import datetime, timedelta
import time
import json
import os.path
import pyautogui


class AirQualityParser:
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
    

    def get_city_air_quality(self, lat: float, lon: float):
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)

        # Make sure all required weather variables are listed here
        # The order of variables in hourly or daily is important to assign them correctly below
        url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": self.startdate,
            "end_date": self.enddate,
            "hourly": ["pm10", "pm2_5", "carbon_monoxide", "carbon_dioxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "alder_pollen", "birch_pollen", "grass_pollen", "mugwort_pollen", "olive_pollen", "ragweed_pollen", "european_aqi", "formaldehyde", "pm10_wildfires", "nitrogen_monoxide"],
            "timezone": "Europe/Moscow"
        }
        response = openmeteo.weather_api(url, params=params)[0]

        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()
        hourly_pm10 = hourly.Variables(0).ValuesAsNumpy()
        hourly_pm2_5 = hourly.Variables(1).ValuesAsNumpy()
        hourly_carbon_monoxide = hourly.Variables(2).ValuesAsNumpy()
        hourly_carbon_dioxide = hourly.Variables(3).ValuesAsNumpy()
        hourly_nitrogen_dioxide = hourly.Variables(4).ValuesAsNumpy()
        hourly_sulphur_dioxide = hourly.Variables(5).ValuesAsNumpy()
        hourly_ozone = hourly.Variables(6).ValuesAsNumpy()
        hourly_alder_pollen = hourly.Variables(7).ValuesAsNumpy()
        hourly_birch_pollen = hourly.Variables(8).ValuesAsNumpy()
        hourly_grass_pollen = hourly.Variables(9).ValuesAsNumpy()
        hourly_mugwort_pollen = hourly.Variables(10).ValuesAsNumpy()
        hourly_olive_pollen = hourly.Variables(11).ValuesAsNumpy()
        hourly_ragweed_pollen = hourly.Variables(12).ValuesAsNumpy()
        hourly_european_aqi = hourly.Variables(13).ValuesAsNumpy()
        hourly_formaldehyde = hourly.Variables(14).ValuesAsNumpy()
        hourly_pm10_wildfires = hourly.Variables(15).ValuesAsNumpy()
        hourly_nitrogen_monoxide = hourly.Variables(16).ValuesAsNumpy()

        hourly_data = {"date": pd.date_range(
            start = pd.to_datetime(hourly.Time(), unit = "s", utc = True) + pd.Timedelta(hours=3),
            end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True) + pd.Timedelta(hours=3),
            freq = pd.Timedelta(seconds = hourly.Interval()),
            inclusive = "left"
        )}
        hourly_data["pm10"] = hourly_pm10
        hourly_data["pm2_5"] = hourly_pm2_5
        hourly_data["carbon_monoxide"] = hourly_carbon_monoxide
        hourly_data["carbon_dioxide"] = hourly_carbon_dioxide
        hourly_data["nitrogen_dioxide"] = hourly_nitrogen_dioxide
        hourly_data["sulphur_dioxide"] = hourly_sulphur_dioxide
        hourly_data["ozone"] = hourly_ozone
        hourly_data["alder_pollen"] = hourly_alder_pollen
        hourly_data["birch_pollen"] = hourly_birch_pollen
        hourly_data["grass_pollen"] = hourly_grass_pollen
        hourly_data["mugwort_pollen"] = hourly_mugwort_pollen
        hourly_data["olive_pollen"] = hourly_olive_pollen
        hourly_data["ragweed_pollen"] = hourly_ragweed_pollen
        hourly_data["european_aqi"] = hourly_european_aqi
        hourly_data["formaldehyde"] = hourly_formaldehyde
        hourly_data["pm10_wildfires"] = hourly_pm10_wildfires
        hourly_data["nitrogen_monoxide"] = hourly_nitrogen_monoxide

        return hourly_data
    
    
    def save_to_csv(self, hourly_data, filename):
        hourly_dataframe = pd.DataFrame(data = hourly_data)
        hourly_dataframe.to_csv(filename, index=False)


    def vpn(self):
        pyautogui.moveTo(1502, 651)
        time.sleep(0.3)
        pyautogui.click()
        time.sleep(5.2)
        pyautogui.moveTo(1501, 403)
        time.sleep(0.4)
        pyautogui.click()
        time.sleep(16)

    
    def get_air_quality(self):
        data = self.open_json('cfo.list.json')

        for i, city in enumerate(data):
            lat, lon, city_name, city_id = round(city['coord']['lat'], 2), round(city['coord']['lon'], 2), city['name'], city['id']

            if not os.path.isdir('air_quality_by_city'):
                os.makedirs('air_quality_by_city')

            if os.path.isfile(f'air_quality_by_city/{self.startdate}_{self.enddate}_{city_id}.csv'):
                 continue
            
            j = 1
            while j < 3:
                try:
                    city_air_quality = self.get_city_air_quality(lat, lon)
                    self.save_to_csv(city_air_quality, f'air_quality_by_city/{self.startdate}_{self.enddate}_{city_id}.csv')
                    break
                except Exception as e:
                    city_air_quality = pd.DataFrame(data = [])
                    delay = (e.args[0])['reason'].split()[0] 

                    if delay == 'Minutely':
                        timeout = 60 - datetime.now().second + 4
                        print(f'{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}: Exceeded {delay}, waiting {timeout} seconds')
                        time.sleep(timeout)

                    elif delay == 'Hourly':
                        new_dt = (datetime.now() + timedelta(hours=1))
                        timeout = (datetime(new_dt.year, new_dt.month, new_dt.day, new_dt.hour, 0, 0) - datetime.now()).total_seconds() + 4
                        m = int(timeout//60)
                        s = int(timeout - m * 60)
                        print(f'{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}: Exceeded {delay}, waiting {m} minutes {s} seconds')
                        time.sleep(timeout)

                    elif delay == 'Daily':
                        new_dt = (datetime.now() + timedelta(days=1))
                        timeout = (datetime(new_dt.year, new_dt.month, new_dt.day, 0, 0, 0) - datetime.now()).total_seconds() + 4
                        h = int(timeout//3600)
                        m = int((timeout - (h * 3600))//60)
                        s = int(timeout - h * 3600 - m * 60)
                        print(f'{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}: Exceeded {delay}, waiting {h} hours {m} minutes {s} seconds')
                        time.sleep(timeout)

                    else:
                        print(e)

                    j += 1

            print(f'{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}: Attempt: {j}/3 {i+1}/{len(data)} City: {city_name} City_id: {city['id']} rows: {len(city_air_quality['date']) if len(city_air_quality) != 0 else 0}')


if __name__ == '__main__':
    air_quality_parser = AirQualityParser('2020-01-01', '2021-12-31')
    air_quality_parser.get_air_quality()
