import requests
import json
import os.path
from datetime import datetime
import time


class OpenWeatherParser:
    def __init__(self, token: str, timeout: float = 1):
        self.token = token
        self.timeout = timeout
    
    def open_json(self, filename: str) -> list[dict]:
        """
        Open json file and return emply list if json is empty
        """

        try:
            with open(filename) as file:
                data = json.load(file)
        except Exception as e:
            data = []
            print(f'JSON file: {filename} is empty\n{e}')

        return data


    def write_json(self, filename: str, data: list[dict]):
        """
        Write data to json file
        """
        
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)


    def get_weather(self, city_id: int, start: int, end: int) -> list:
        url = f'https://history.openweathermap.org/data/2.5/history/city?id={city_id}&start={start}&end={end}&units=metric&type=day&appid={self.token}'
        weather_list = requests.get(url).json()['list']
        return weather_list

    
    def load_weather_by_city(self, start: int, end: int):
        city_list = self.open_json('cfo.list.json')

        if not os.path.isdir('weather_by_city'):
             os.makedirs('weather_by_city')

        for city in city_list:
            city_weather = []
            startdate = start
            enddate = startdate + 82800
            filename = f'weather_by_city/{city['id']}_{city['name']}_{start}_{end}.json'
            
            if os.path.isfile(filename):
                 continue

            while True:
                if startdate == end:
                    break

                try:
                    weather = self.get_weather(city['id'], startdate, enddate)
                except Exception:
                     weather = []
                
                print(f'{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}: City: {city['name']} {datetime.fromtimestamp(startdate).strftime('%d.%m.%Y %H:%M:%S')}-{datetime.fromtimestamp(enddate).strftime('%d.%m.%Y %H:%M:%S')} rows: {len(weather)}')
                city_weather.extend(weather)
                time.sleep(self.timeout)

                startdate += 86400
                enddate += 86400
            self.write_json(f'weather_by_city/{city['id']}_{city['name']}_{start}_{end}.json', city_weather)
            break


if __name__ == '__main__':
    with open('token.json') as file:
                data = json.load(file)
    token = data[0]['openweathermap_TOKEN']
    
    open_weather_parser = OpenWeatherParser(token, timeout=0.1)
    open_weather_parser.load_weather_by_city(1704056400, 1730408400)
