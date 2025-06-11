# Прогнозирование индекса качества воздуха (AQI)

[Checkpoints](https://github.com/AI-YP-24-2-1/aqi_prediction/blob/main/docs/checkpoints.md)

## Описание проекта
В рамках проекта будет реализована модель прогнозирования индекса качества воздуха на основе загрязнителей воздуха, погоды и социально-экономических данных<br>
Пользователь сможет взаимодействовать с моделью с помощью Streamlit сервиса<br>

Модель будет оценивать множество факторов, влияющих на качество воздуха. Среди них:
* Скорость и направление ветра
* Температура и влажность воздуха
* Атмосферное давление
* Количество осадков
* Основные загрязнители воздуха
* Доход и численность населения

Модель будет анализировать данные из разных источников:
* Базы данных с историческими значениями о качестве воздуха
* Данные метеорологических станций

## Цель проекта
Предоставить возможность мониторинга прогнозного значения качества воздуха по городам Центрального федерального округа на будущие периоды<br>

## Технологии
* Python 3.11+
* Git
* PostgreSQL
* Docker
* FastAPI
* Aiogram 3.x

## Структура проекта
![Alt text](https://github.com/AI-YP-24-2-1/aqi_prediction/blob/main/images/1.project_structure.png?raw=true)

## Деплой проекта
Для запуска Streamlit сервиса с Fastapi необходимо:
1. Клонировать репозиторий
```
git clone https://github.com/AI-YP-24-2-1/aqi_prediction.git
```
2. перейти и директорию "aqi_prediction"
```
cd aqi_prediction
```
3. Контейнеризовать приложение в Docker
```
docker-compose build
docker-compose up
```

Streamlit сервис будет доступен по url: http://localhost:8501

## Возможности Streamlit
1. Streamlit предоставляет возможность загрузить csv файл с данными для обучения модели
Для этого в меню слева загружаем csv файл (тестовый файл можно взять в директории fast_api_dataset/dataset.csv)<br>
![Alt text](https://github.com/AI-YP-24-2-1/aqi_prediction/blob/main/images/2.load_train_data.png?raw=true)

На этом этапе у нас есть 5 разделов:
* Загрузка и удаление моделей - мы можем посмотреть загруженные и незагруженные модели, можем загрузить выбранную модель или удалить все модели
![Alt text](https://github.com/AI-YP-24-2-1/aqi_prediction/blob/main/images/3.loading_deleting_models.png?raw=true)<br>

* Анализ загруженных данных для обучения - в этом разделе отображается информация о загруженном датасете (размер датасета, информация о показателях, корреляционная матрица и графики по показателям)
![Alt text](https://github.com/AI-YP-24-2-1/aqi_prediction/blob/main/images/4.train_data_analysis.png?raw=true)<br>

* Обучение SGD модели - в этом разделе обучаем SGD модель. Необходимо указать название модели, указать гиперпараметры и начать ее обучение. После обучения модель будет отображаться в списке загруженных моделей в разделе "Загрузка и удаление моделей"
![Alt text](https://github.com/AI-YP-24-2-1/aqi_prediction/blob/main/images/5.train_sgd_model.png?raw=true)<br>

* Обучение Flaml AutoML модели - в этом разделе обучаем Flaml AutoML модель. Необходимо указать название модели, указать время обучения и начать ее обучение. После обучения модель будет отображаться в списке загруженных моделей в разделе "Загрузка и удаление моделей"
![Alt text](https://github.com/AI-YP-24-2-1/aqi_prediction/blob/main/images/6.train_flaml_automl_model.png?raw=true)<br>

* Сравнение моделей - В этом разделе необходимо выбрать две загруженные модели. Для них будет отображены гиперпараметры, метрики качества и коэффициенты модели, а также кривые обучения
![Alt text](https://github.com/AI-YP-24-2-1/aqi_prediction/blob/main/images/7.compare_models.png?raw=true)<br>

2. Есть возможность загрузить csv файл с данными для прогнозирования AQI
Для этого в меню слева загружаем csv файл (тестовый файл можно взять в директории fast_api_dataset/dataset_predict.csv)<br>
![Alt text](https://github.com/AI-YP-24-2-1/aqi_prediction/blob/main/images/8.load_prediction_data.png?raw=true)<br>

На этом этапе у нас появляется 2 новых раздела:
* Анализ загруженных данных для прогноза - в этом разделе отображается информация о загруженном датасете (размер датасета, информация о показателях, корреляционная матрица и графики по показателям)
![Alt text](https://github.com/AI-YP-24-2-1/aqi_prediction/blob/main/images/9.forecast_data_analysis.png?raw=true)<br>

* Построение прогноза - мы можем выбрать модель из списка загруженных моделей и сделать прогноз. После построения прогноза можно будет загрузить csv файл с прогнозными значениями AQI
![Alt text](https://github.com/AI-YP-24-2-1/aqi_prediction/blob/main/images/10.model_forecast.png?raw=true)<br>

## Демонстрация работы Streamlit
![alt text](https://github.com/AI-YP-24-2-1/aqi_prediction/blob/main/images/Streamlit_video.gif?raw=true)

## Методы Fastapi
Взаимодействие с моделями происходит с помощью Fastapi<br>
Fastapi доступен по url: http://0.0.0.0:8000/

Методы Fastapi:<br>
* POST /load - загружает выбранную модель
* GET /list_models - отображает список загруженных моделей
* GET /list_models_not_loaded - отображает список незагруженных моделей
* DELETE /remove_all - удаляет все модели
* POST /fit_sgd - обучение sgd модели
* POST /fit_flaml - обучение flaml модели
* POST /predict - строит прогноз AQI
* GET /list_models_for_comparison - загружает обученные модели для которых есть кривые обучения
* POST /compare_models - отображает данные модели для сравнения

# Команда проекта
Куратор проекта -  [Марк Блуменау](http://telegram.me/markblumenau)

Участники проекта:
* [Иван Махров](https://telegram.me/MakhrovIvan)
* [Владислав Щукин](https://telegram.me/shchukin_ve)
* [Лука Марков](https://telegram.me/lulu_fw01)
* [Мансур Аглиев](https://telegram.me/mansagliev)
