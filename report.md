### Основные блоки проекта:
* Fastapi - api, который истользуется приложением Streamlit для обучения моделей и построения прогнозов
* Streamlit - приложение для взаимодействия пользователей с моделями

### Структура проекта:
* app.py - Streamlit приложение
* main.py - Fastapi сервис
* models/ - директория с обученными моделями
* logs/ - log файлы

### Возможности Streamlit
1. Streamlit предоставляет возможность загрузить csv файл с данными для обучения модели
Для этого в меню слева загружаем csv файл (тестовый файл можно взять в директории fast_api_dataset/dataset.csv)<br>
![Alt text](https://github.com/AI-YP-24-2-1/aqi_prediction/blob/main/images/load_train_data.png?raw=true)

На этом этапе у нас есть 3 раздела:
* Загрузка и удаление моделей - мы можем посмотреть загруженные и незагруженные модели, можем загрузить выбранную модель или удалить все модели
![Alt text](https://github.com/AI-YP-24-2-1/aqi_prediction/blob/main/images/loading_deleting_models.png?raw=true)
* Анализ загруженных данных для обучения - в этом разделе отображается информация о загруженном датасете (размер датасета, информация о показателях, корреляционная матрица и графики по показателям)
![Alt text](https://github.com/AI-YP-24-2-1/aqi_prediction/blob/main/images/train_data_analysis.png?raw=true)
* Обучение моделей - в этом разделе необходимо указать название модели и начать ее обучение. После обучения модель будет отображаться в списке загруженных моделей в разделе "Загрузка и удаление моделей"
![Alt text](https://github.com/AI-YP-24-2-1/aqi_prediction/blob/main/images/train_model.png?raw=true)

2. Есть возможность загрузить csv файл с данными для прогнозирования AQI
Для этого в меню слева загружаем csv файл (тестовый файл можно взять в директории fast_api_dataset/dataset_predict.csv)<br>
![Alt text](https://github.com/AI-YP-24-2-1/aqi_prediction/blob/main/images/load_prediction_data.png?raw=true)

На этом этапе у нас появляется 2 новых раздела:
* Анализ загруженных данных для прогноза - в этом разделе отображается информация о загруженном датасете (размер датасета, информация о показателях, корреляционная матрица и графики по показателям)
![Alt text](https://github.com/AI-YP-24-2-1/aqi_prediction/blob/main/images/forecast_data_analysis.png?raw=true)
* Построение прогноза - мы можем выбрать модель из списка загруженных моделей и сделать прогноз. После построения прогноза можно будет загрузить csv файл с прогнозными значениями AQI
![Alt text](https://github.com/AI-YP-24-2-1/aqi_prediction/blob/main/images/train_model.png?raw=true)

## Методы Fastapi
Взаимодействие с моделями происходит с помощью Fastapi<br>
Fastapi доступен по url: http://0.0.0.0:8000/

Методы Fastapi:<br>
* POST /fit - обучение модели 
* GET /load_main - загружает базовую модель
* POST /load - загружает выбранную модель
* POST /predict - строит прогноз AQI
* GET /list_models - отображает список загруженных моделей
* GET /list_models_not_loaded - отображает список незагруженных моделей
* DELETE /remove_all - удаляет все модели

### Деплой проекта
Для запуска Streamlit сервиса с Fastapi необходимо:
1. Клонировать репозиторий, перейти и директорию "aqi_prediction"
2. Контейнеризовать приложение в Docker
```
docker-compose build
docker-compose up
```

Streamlit сервис будет доступен по url: http://localhost:8501
