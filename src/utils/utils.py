import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry
import requests

from datetime import datetime, timedelta
import pytz

def get_weather(lat, lon):
	cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
	retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
	openmeteo = openmeteo_requests.Client(session = retry_session)

	today = pd.Timestamp.now(tz='UTC').normalize()
	start_date = (today - timedelta(days=3)).strftime('%Y-%m-%d')
	end_date = (today - timedelta(days=1)).strftime('%Y-%m-%d')

	url = "https://archive-api.open-meteo.com/v1/archive"
	params = {
		"latitude": lat,
		"longitude": lon,
		"start_date": start_date,
		"end_date": end_date,
		"hourly": ["temperature_2m", "relative_humidity_2m", "rain", "snowfall", "snow_depth", "cloud_cover", "wind_speed_10m", "wind_direction_10m"],
		"timezone": "auto"
	}
	responses = openmeteo.weather_api(url, params=params)

	response = responses[0]

	hourly = response.Hourly()
	hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
	hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
	hourly_rain = hourly.Variables(2).ValuesAsNumpy()
	hourly_snowfall = hourly.Variables(3).ValuesAsNumpy()
	hourly_snow_depth = hourly.Variables(4).ValuesAsNumpy()
	hourly_cloud_cover = hourly.Variables(5).ValuesAsNumpy()
	hourly_wind_speed_10m = hourly.Variables(6).ValuesAsNumpy()
	hourly_wind_direction_10m = hourly.Variables(7).ValuesAsNumpy()

	hourly_data = {"date": pd.date_range(
		start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
		end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = hourly.Interval()),
		inclusive = "left"
	)}

	hourly_data["temperature_2m"] = hourly_temperature_2m
	hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
	hourly_data["rain"] = hourly_rain
	hourly_data["snowfall"] = hourly_snowfall
	hourly_data["snow_depth"] = hourly_snow_depth
	hourly_data["cloud_cover"] = hourly_cloud_cover
	hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
	hourly_data["wind_direction_10m"] = hourly_wind_direction_10m

	hourly_data['lat'] = lat
	hourly_data['lon'] = lon

	hourly_dataframe = pd.DataFrame(data = hourly_data)

	df = hourly_dataframe.dropna(subset=['temperature_2m', 'relative_humidity_2m', 'rain', 'snowfall', 'snow_depth', 'cloud_cover', 'wind_speed_10m', 'wind_direction_10m'], how='all')
	df = df.sort_values('date', ascending=False).head(24).reset_index(drop=True)
	
	return df

def get_aqi_data(lat, lon, min_date, max_date):
	cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
	retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
	openmeteo = openmeteo_requests.Client(session = retry_session)

	url = "https://air-quality-api.open-meteo.com/v1/air-quality"
	params = {
		"latitude": lat,
		"longitude": lon,
		"hourly": ["pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "nitrogen_monoxide"],
		"start_date": min_date,
		"end_date": max_date
	}
	responses = openmeteo.weather_api(url, params=params)

	response = responses[0]

	hourly = response.Hourly()
	hourly_pm2_5 = hourly.Variables(0).ValuesAsNumpy()
	hourly_carbon_monoxide = hourly.Variables(1).ValuesAsNumpy()
	hourly_nitrogen_dioxide = hourly.Variables(2).ValuesAsNumpy()
	hourly_sulphur_dioxide = hourly.Variables(3).ValuesAsNumpy()
	hourly_ozone = hourly.Variables(4).ValuesAsNumpy()
	hourly_nitrogen_monoxide = hourly.Variables(5).ValuesAsNumpy()

	hourly_data = {"date": pd.date_range(
		start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
		end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = hourly.Interval()),
		inclusive = "left"
	)}

	hourly_data["pm2_5"] = hourly_pm2_5
	hourly_data["carbon_monoxide"] = hourly_carbon_monoxide
	hourly_data["nitrogen_dioxide"] = hourly_nitrogen_dioxide
	hourly_data["sulphur_dioxide"] = hourly_sulphur_dioxide
	hourly_data["ozone"] = hourly_ozone
	hourly_data["nitrogen_monoxide"] = hourly_nitrogen_monoxide

	hourly_dataframe = pd.DataFrame(data = hourly_data)

	return hourly_dataframe

def get_city(lat, lon, api_key):
	url = f'http://api.openweathermap.org/geo/1.0/reverse?lat={lat}&lon={lon}&limit=1&appid={api_key}'
	data = requests.get(url).json()[0]
	city = data['name']
	region = data['state']

	return city, region

def get_population_income(region):
	df = pd.read_json('population_income.json')
	population = df[df['region'] == region]['population']
	income = df[df['region'] == region]['income']

	return population, income

def process_data(lat, lon):
	weather = get_weather(lat, lon)

	min_date = weather['date'].min().strftime('%Y-%m-%d')
	max_date = weather['date'].max().strftime('%Y-%m-%d')

	aqi = get_aqi_data(lat, lon, min_date, max_date)

	df = pd.merge(weather, aqi, on='date', how='inner')

	df['year'] = df['date'].dt.year
	df['month'] = df['date'].dt.month
	df['day'] = df['date'].dt.day
	
	df = df.drop('date', axis=1)

	api_key = ''

	city, region = get_city(lat, lon, api_key)

	df['city_name'] = city
	df['region'] = region

	population, income = get_population_income(region)

	df['population'] = int(population)
	df['income'] = float(income * 1.1)


	return df

df = process_data(lat=55.7522, lon=37.6156)
print(df)
