from datetime import datetime
import time
import json
from dadata import Dadata


class DadataParser:
    def __init__(self, token: str, timeout: float = 1):
        self.token = token
        self.dadata = Dadata(self.token)
        self.timeout = timeout

    def __check_cfo(self, lat: float, lon: float) -> str:
        """
        Get federal dictrict by coordinates and return federal dictrict name
        Try-except is used when city is not found
        """

        try:
            data = self.dadata.geolocate(name="address", count=1,
                                         radius_meters=1000,
                                         lat=lat,
                                         lon=lon
                                         )
            cfo = data[0]['data']['federal_district']
            region = data[0]['data']['region_with_type']
        except Exception:
            cfo, region = '', ''

        time.sleep(self.timeout)
        return (cfo.strip(), region.strip())

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

    def get_cfo_cities(self, city_list_filename: str, cfo_list_filename: str,
                       city_not_found_filename: str):
        """
        Find cities in central federal district and save to json
        Try-except is applied in cases when number of api requests is exceeded
        """

        city_not_found_list = []
        city_list = self.open_json(city_list_filename)
        cfo_list = self.open_json(cfo_list_filename)
        ru_city_list_len = len([city for city in city_list
                                if city['country'] == 'RU'])

        n = 0
        for i, city in enumerate(city_list):
            if (city['country'] == 'RU'
                    and city not in cfo_list):
                n += 1

                try:
                    cfo, region = self.__check_cfo(
                        city['coord']['lat'],
                        city['coord']['lon']
                        )
                except Exception as e:
                    print(f'{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}:'
                          f'\n{e}')
                    break

                if cfo == 'Центральный':
                    city['region'] = region
                    del city['state']
                    cfo_list.append(city)
                elif not cfo:
                    city_not_found_list.append(city)
                    print(f'{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}: '
                          f'city not found {city_list[i]}')

                print(f'{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}: '
                      f'ru_cities: '
                      f'{n}/{ru_city_list_len} all_cities: '
                      f'{i}/{len(city_list)} '
                      f'added: {len(cfo_list)} City: {city['name']} '
                      f'CFO: {cfo}')

        self.write_json(cfo_list_filename, cfo_list)
        self.write_json(city_not_found_filename, city_not_found_list)


if __name__ == '__main__':
    with open('token.json') as file_token:
        dadata_token = json.load(file_token)[0]['dadata_TOKEN']

    dadata_parser = DadataParser(dadata_token, timeout=0.5)
    dadata_parser.get_cfo_cities('city.list.json', 'cfo.list.json',
                                 'city.not.found.json'
                                 )
