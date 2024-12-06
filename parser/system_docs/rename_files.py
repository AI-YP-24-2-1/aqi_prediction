import os
import json


def open_json(filename: str) -> list[dict]:
        try:
            with open(filename) as file:
                data = json.load(file)
        except Exception as e:
            data = []
            print(f'JSON file: {filename} is empty\n{e}')
        
        return data

data = open_json('city.list.json')

def find_city_id(city_name, lat, lon):
    for city in data:
        if city['name'].replace('e'+'̈', 'ë').replace(r'\u00eb', 'ё').strip() == city_name.replace('e'+'̈', 'ë').replace(r'\u00eb', 'ё').strip()  and round(city['coord']['lat'],2) == float(lat) and round(city['coord']['lon'],2) == float(lon):
            return city['id']


def rename(folder_name):
    for filename in os.listdir(folder_name):
        filename_data = filename[:-4].split('_')

        if len(filename_data) == 3 or filename == '.DS_Store':
            continue
        
        city_name, lat, lon = filename_data[2], filename_data[3], filename_data[4]
        city_id = find_city_id(city_name, lat, lon)

        if city_id:
            filename_new = f'{filename[:-4].split('_')[0]}_{filename[:-4].split('_')[1]}_{city_id}.csv'
            os.rename(f'{folder_name}/{filename}', f'{folder_name}/{filename_new}')
            print(f'{filename} -> {filename_new}')

        else:
            print(f'Failed to rename: {city_name} {lat} {lon} {city_id}')


rename('weather_by_city')
