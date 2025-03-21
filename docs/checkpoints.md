## 1. Сбор данных
### 1.1. Поиск источников данных с индикаторами, которые оказывают влияние на качество воздуха и уровень пыльцы
На этом этапе каждый участник команды формирует список источников данных для дальнейшего обучения модели. Данные загружаем с 2020 года по настоящее время

Махров Иван:
* Сбор метеорологических данных с метеорологических станций и из открытых источников исследовательских институтов (скорость и направление ветра, влажность воздуха, температура)
* Исторические данные по ключевым индикаторам качества воздуха (PM2.5, PM10, CO, NO2, SO2, O3). Данные могут быть предоставлены службами экологического мониторинга, экологическими агентствами и датчиками качества воздуха
* Сбор исторических данных о типе пыльцы и ее количестве в воздухе. Источниками могут служить установленные датчики, а также архивы исторических данных

Лука Марков:
* Социально-экономические данные (плотность населения, уровень доходов, уровень урбанизации, развитие промышленности, количество ТЭС итд)
* Сбор данных о плотности движения, пробках

Владислав Щукин:
* Сбор данных о площади лесов, а также видах деревьев, которые производят пыльцу
* Статистика по заболеваниям, связанным с загрязнением воздуха

Мансур Аглиев:
* Данные о землепользовании и сельскохозяйственной деятельности (использование пестицидов ухдшают качество воздуха)
* Данные об интенсивности строительства
* Данные о лесных пожарах

На встрече в Zoom 12.10.2024 будет оцениваться количество источников данных, оценочная полнота и качество данных, а также возможность их получения из источника. По итогам встречи будут определены источники, из которых данные будут загружены в БД, а также составлен список индикаторов, которые будут использоваться для прогнозирования качества воздуха и количества пыльцы в воздухе. Также будет принято решение для какого региона модель будет выдавать прогнозные значения (город, страна или все страны)

Сроки: 07.10.2024 - 13.10.2024

### 1.3. Получение данных и их загрузка в БД
На данном этапе происходит загрузка данных в PostgreSQL
По итогам встречи в Zoom 19.10.2024 будет проведена оценка собранных данных и возможность их использования для построения модели

Сроки: 14.10.2024 - 20.10.2024

## 2. Подготовка данных
### 2.1. Исследовательский анализ данных (EDA)
На данном этапе происходит анализ существующих данных, а именно: 
* Построение графиков основных измерений датасета
* Выявление корреляции между признаками
* Поиск особенностей в данных и неявных закономерностей
* Определение статистических характеристик данных

По итогам встречи в Zoom 26.10.2024 будет произведена оценка проведенного анализа данных, результаты которого помогут увеличить точность прогноза модели

Сроки: 21.10.2024 - 27.10.2024

### 2.2. Предобработка данных
На данном этапе необходимо провести первичную предобработку данных, а именно:
* Обработка пропусков в данных
* Удаление аномальных значений
* Удаление дубликатов
* Приведение данных к единому формату
* Анализ качества данных и корректировка "кривых" значений (лишние пробелы в категориальных переменных)
* Генерация новых признаков на основании уже существующих для повышения точности прогноза

По итогам встречи 16.11.2024 будет произведена оценка качества и полноты датасета, определены существующие недочеты в данных, которые необходимо учитывать

Сроки: 27.10.2024 - 17.11.2024

### 2.3. Подбор признаков и подготовка данных к обучению
На данном этапе необходимо:
* Сгенерировать новые признаки на основании уже существующих для повышения точности прогноза
* Провести нормализацию данных
* Преобразовать категориальные переменные в числовые

Сроки: 17.11.2024 - 01.12.2024

## 3. Построение ML модели
По результатам данного этапа должна быть реализована модель линейной регрессии. Необходимо:
* Разделить датасет на обучающую и тестовую выборки
* Построить модель линейной регрессии
* Для каждой итерации необходимо оценивать ошибки модели и контролировать ее качество

Сроки: 02.12.2024 - 15.12.2024

## 4. Развертывание модели
На данном этапе необходимо реализовать возможность взаимодействия с моделью

### 4.1. Разработка телеграм-бота
Необходимо разработать телеграм-бот, который будет осуществлять рассылку прогнозных значений по индексу качества воздуха и пыльце в выбранном регионе

Сроки: 16.12.2024 - 29.12.2024

### 4.2. Разработать web-сервис
Необходимо разработать web-сервис, через который будет возможно получать прогнозные значения данных через api

Сроки: 13.01.2025 - 26.01.2025

## 5. Построение DL модели
По результатам данного этапа необходимо реализовать одну из моделей глубокого обучения, которая будет давать наилучший результат

### 5.1. Построение модели основанной на полносвязных слоях
* Необходимо выбрать тип модели глубокого обучения
* Определить количество слоев и нейронов на каждом слое
* Выбрать функцию активации
* Установить значения для гиперпараметров
* Обучить модель и оценить ее качество

Сроки: 27.01.2025 - 23.02.2025

### 5.2. Построение GAN модели
* Выбрать архитектуру для генератора и дискриминатора
* Определить количество слоев и нейронов на каждом слое
* Выбрать функцию активации
* Обучить модель и оценить ее качество

Сроки: 24.02.2025 - 23.03.2025

## 6. Разработка веб-сайта для мониторинга индекса качества воздуха и пыльцы

Сроки: 24.03.2025 - 04.05.2025

## 7. Финальный мониторинг контроля реализованных моделей и инструментов передачи ппогнозных значений

Сроки: 05.05.2025 - 25.05.2025





