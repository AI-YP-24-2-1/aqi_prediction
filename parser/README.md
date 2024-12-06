
Данные хранятся на Яндекс Диске: https://disk.yandex.com/d/tTooISHjjJf5UQ

Список городов с координатами были извлечены в файл city.list.json по ссылке: https://bulk.openweathermap.org/sample/

С помощью обратного геокодирования из координат были извлечены названия регионов и проверена принадлежность к Центральному федеральному округу. С помощью dadata api (cfo_cities.py) в файле cfo.list.json были сохранены только населенные пункты в Центральном федеральном округе

По координатам были получены архивные данные по погоде и пыльце с помощью https://open-meteo.com и сохранены в .csv (open_metro_air_quality_parser.py и open_meteo_weather_parser.py)

.csv файлы были объединены в один файл (merge_csv.py)
