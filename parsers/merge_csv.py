import json
import os
from datetime import datetime
import pandas as pd


class MergeCsv:
    def __init__(self, weather_directory, air_directory, file_to_save):
        self.weather_directory = weather_directory
        self.air_directory = air_directory
        self.file_to_save = file_to_save
        self.i = 0

    def open_json(self, filename: str) -> list[dict]:
        '''
        Opening json
        '''

        try:
            with open(filename) as file:
                data = json.load(file)
        except Exception as e:
            data = []
            print(f'JSON file: {filename} is empty\n{e}')

        return data

    def get_city_info(self, city_id):
        '''
        Get info about city
        '''

        city_list = self.open_json('cfo.list.json')

        for city in city_list:
            if int(city['id']) == city_id:
                city_name = city['name'] if 'name' in city.keys() else ''
                region = city['region'] if 'region' in city.keys() else ''
                lat = city['coord']['lat'] if 'coord' in city.keys() else 1000
                lon = city['coord']['lon'] if 'coord' in city.keys() else 1000
                return (city_name, region, lat, lon)

    def find_empty_info(self, city_name, region, lat, lon):
        '''
        Finding empty fields
        '''

        fields = ''

        if city_name == '':
            fields += ' city_name'
        elif region == '':
            fields += ' region'
        elif lat == 1000:
            fields += ' lat'
        elif lon == 1000:
            fields += ' lon'

        return fields.strip()

    def add_fields(self, df, city_id):
        '''
        Adding new fields
        '''

        city_name, region, lat, lon = self.get_city_info(city_id)
        check_fields = self.find_empty_info(city_name, region, lat, lon)

        if check_fields:
            list_len = len(os.listdir(self.weather_directory)) + \
                len(os.listdir(self.air_directory))
            print(f'{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}: '
                  f'{self.i}/{list_len} Info "{check_fields}" '
                  f'not found for {city_id}')

        df['year'] = pd.DatetimeIndex(df['date']).year
        df['quarter'] = pd.DatetimeIndex(df['date']).quarter
        df['month'] = pd.DatetimeIndex(df['date']).month
        df['day'] = pd.DatetimeIndex(df['date']).day
        df['hour'] = pd.DatetimeIndex(df['date']).hour
        df['city_id'] = city_id
        df['city_name'] = city_name
        df['region'] = region
        df['lat'] = lat
        df['lon'] = lon

        return df

    def merge_csv_files(self):
        '''
        Merging csv files
        '''

        city_id_list = [city['id'] for city in self.open_json('cfo.list.json')]
        period_list = []

        for filename in (
            os.listdir(self.air_directory) + os.listdir(self.weather_directory)
                ):

            if filename[-4:] == '.csv':
                filename = '_'.join(filename.split('_')[:2])
                period_list.append(filename)

        period_list = sorted(list(set(period_list)))
        excel_data = pd.read_excel('excel_data_to_load.xlsx')

        for period in period_list:
            for city_id in city_id_list:
                self.i += 1

                try:
                    air_csv_file = f'{self.air_directory}/'
                    f'{period}_{city_id}.csv'
                    weather_csv_file = f'{self.weather_directory}/'
                    f'{period}_{city_id}.csv'
                    air_csv = pd.read_csv(air_csv_file)
                    weather_csv = pd.read_csv(weather_csv_file)
                except Exception:
                    print(f'{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}: '
                          f'{self.i}/{len(city_id_list)*2}: '
                          f'Error on {period}_{city_id}.csv')
                    continue

                air_csv = self.add_fields(air_csv, city_id)
                weather_csv = self.add_fields(weather_csv, city_id)

                fields_merge = ['date', 'year', 'quarter', 'month', 'day',
                                'hour', 'city_id', 'city_name', 'region',
                                'lat', 'lon']
                weather_air_csv = pd.merge(weather_csv, air_csv,
                                           on=fields_merge
                                           )
                weather_air_excel_csv = pd.merge(weather_air_csv,
                                                 excel_data,
                                                 on=['year', 'quarter',
                                                     'region'
                                                     ]
                                                 )

                print(f'{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}: '
                      f'{self.i}/{len(city_id_list)*len(period_list)}')

                self.save_df(weather_air_excel_csv, self.file_to_save)

    def save_df(self, df, filename):
        '''
        Saving df
        '''

        if not os.path.isfile(filename):
            df.to_csv(filename, mode='w', header=True, index=False)
        else:
            df.to_csv(filename, mode='a', header=False, index=False)


if __name__ == '__main__':
    merge_files = MergeCsv(weather_directory='weather_by_city',
                           air_directory='air_quality_by_city',
                           file_to_save='air_weather_data.csv'
                           )
    merge_files.merge_csv_files()
