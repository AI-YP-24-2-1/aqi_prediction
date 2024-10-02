## 1.Сбор данных
### 1.1.Определение индикаторов, которые могут оказывать влияние на качество воздуха
В рамках данного пункта каждый из участников формирует список индикаторов, которые могут оказывать влияние на качество воздуха и количество пыльцы в воздухе
На встрече в Zoom 12.10.2024 формируется согласованный всеми участниками список индикаторов

Сроки: 07.10.2024 - 13.10.2024
  
### 1.2.Поиск источников данных
На этом этапе каждый участник команды формирует список источников данных для дальнейшего обучения модели
* Исторические данные по ключевым индикаторам качества воздуха по различным странам (PM2.5, PM10, CO, NO2, SO2, O3). Данные могут быть предоставлены службами экологического мониторинга, экологическими агентствами и датчиками качества воздуха
* Сбор метеорологических данных с метеорологических станций и из открытых источников исследовательских институтов (скорость и направление ветра, влажность воздуха, температура)
* Сбор данных о типе пыльцы и ее количестве в воздухе. Источниками могут служить установленные датчики, а также архивы исторических данных
* Сбор данных о плотности движения и пробках
* Данные о землепользовании
* Социально-экономические данные о плотности населения, уровне доходов
* Статистика по заболеваниям, связанным с загрязнением воздуха
* Данные о лесных пожарах

На встрече в Zoom 19.10.2024 будет оцениваться количество источников данных, оценочная полнота и качество данных, а также возможность их получения в БД. По итогам встречи будут определены источники, из которых данные будут загружены в БД. Также будет принято решение для какого региона модель будет выдавать прогнозные значения (город, страна или все страны)
Сроки: 14.10.2024 - 20.10.2024

### 1.3.Получение данных и их загрузка в БД
На данном этапе происходит загрузка данных в PostgreSQL
По итогам встречи в Zoom 09.11.2024 будет проведена оценка собранных данных и возможность их использования для построения модели

Сроки: 21.10.2024 - 10.11.2024


2.Первичный анализ собранных данных, их обработка и создание новых переменных для дальнейшего использования и выбор параметров для модели
Сроки: 21.10.2024 - 03.11.2024
26.10.2024 и 02.11.2024 проходят встречи в Zoom

3.Построение модели регрессии на основе собранных данных
Сроки: 04.11.2024 - 17.11.2024
09.11.2024 и 16.11.2024 проходят встречи в Zoom

4.Валидация модели, повышение точности прогнозирования
Сроки: 18.11.2024 - 01.12.2024

5.Разработка Telegram-бота как инструмента получения прогнозных данных от модели
Сроки: 02.12.2024 - 22.12.2024
