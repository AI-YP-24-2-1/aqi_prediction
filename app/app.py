import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import httpx
import numpy as np
import plotly.graph_objects as go


st.set_page_config(layout="wide")

def plots(df: pd.DataFrame, column: str, region: str) -> None:
    '''
    Creating plots by year and region
    '''

    df = df if region == 'Все регионы' else df.loc[df['region'] == region]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    sns.lineplot(data=df, x='year', y=column, hue='region',
                 marker='o', ax=axes[0]
                 )
    axes[0].set_title(f'{column} by Region')
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel(column)
    axes[0].legend(title='Region', loc='upper left', bbox_to_anchor=(1, 1))

    avg_aqi_by_region = df.groupby('region')[column].mean().reset_index()
    sns.barplot(data=avg_aqi_by_region, x='region', y=column,
                palette='viridis', ax=axes[1]
                )
    axes[1].set_title(f'Average {column} by Region')
    axes[1].set_xlabel('Region')
    axes[1].set_ylabel(f'Average {column}')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(plt)

def boxplot(df: pd.DataFrame, column: str, region: str) -> None:
    '''
    Creating boxplots
    '''

    df = df if region == 'Все регионы' else df.loc[df['region'] == region]

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    right = Q3 + 1.5 * IQR
    left = Q1 - 1.5 * IQR

    filtered_df = df[(df[column] <= right) & (df[column] >= left)]

    fig, axes = plt.subplots(figsize=(18, 6))
    sns.boxplot(data=filtered_df, x='region', y=column, palette='coolwarm')
    axes.set_title(f'Box Plot of {column} by {region}')
    axes.set_xlabel('Region')
    axes.set_ylabel(column)
    axes.grid(True)
    axes.tick_params(axis='x', rotation=45)

    st.pyplot(plt)


st.title("Индекс качества воздуха (AQI)")
st.sidebar.header("Настройки")

file_train_text = "Загрузите csv файл с данными для обучения модели"
file_predict_text = "Загрузите csv файл с данными для прогноза"
file_train = st.sidebar.file_uploader(file_train_text, type="csv")
file_predict = st.sidebar.file_uploader(file_predict_text, type="csv",
                                        key='predict'
                                        )

with st.expander('Загрузка и удаление моделей'):
    response = httpx.get("http://0.0.0.0:8000/list_models_not_loaded",
                         timeout=1000000
                         )
    model_list_not_loaded = response.json()['models']
    model_name_load = st.selectbox('Выберите модель для загрузки:',
                                   model_list_not_loaded, key='load'
                                   )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button('Список загруженных моделей'):
            response = httpx.get("http://0.0.0.0:8000/list_models",
                                 timeout=1000000
                                 )
            model_list = response.json()['models']

            st.write('Список загруженных моделей:')
            for i, model in enumerate(model_list):
                st.write(f'{i+1}. {model}')

    with col2:
        if st.button('Список незагруженных моделей'):
            response = httpx.get("http://0.0.0.0:8000/list_models_not_loaded",
                                 timeout=1000000
                                 )
            model_list_not_loaded = response.json()['models']

            st.write('Список незагруженных моделей:')
            for i, model in enumerate(model_list_not_loaded):
                st.write(f'{i+1}. {model}')

    with col3:
        if st.button('Загрузить указанную модель'):
            response = httpx.post("http://0.0.0.0:8000/load",
                                  data={'model_name': model_name_load},
                                  timeout=1000000
                                  )
            if response.status_code == 200:
                st.write(response.json()['message'])
            else:
                st.write(response.json()['detail'])

    with col4:
        if st.button('Загрузить базовую модель'):
            response = httpx.get("http://0.0.0.0:8000/load_main",
                                 timeout=1000000
                                 )
            if response.status_code == 200:
                st.write(response.json()['message'])
            else:
                st.write(response.json()['detail'])

    with col5:
        if st.button('Удалить все модели'):
            response = httpx.delete("http://0.0.0.0:8000/remove_all",
                                    timeout=1000000
                                    )
            if response.status_code == 200:
                st.write(response.json()['message'])
            else:
                st.write(response.json()['detail'])

if file_train is not None:
    df_train = pd.read_csv(file_train)
    csv_file_train = df_train.to_csv(index=False)

    with st.expander('Анализ загруженных данных для обучения'):
        st.write(
            f'Размер датасета: {df_train.shape[0]} строк и '
            f'{df_train.shape[1]} признаков'
            )

        st.write('Описательная статистика:')
        st.write(df_train.describe().T)

        st.write('Корреляционная матрица:')
        plt.figure(figsize=(20, 8))
        sns.heatmap(df_train.select_dtypes(['float', 'int']).corr(),
                    cmap='coolwarm',
                    annot=True, fmt=".2f", linewidths=.5,
                    xticklabels=df_train.select_dtypes(
                        ['float', 'int']).columns,
                    yticklabels=df_train.select_dtypes(
                        ['float', 'int']).columns
                    )
        st.pyplot(plt)

        indicator = df_train.select_dtypes(['float', 'int']).columns
        region = list(df_train['region'].unique()) + ['Все регионы']

        col1, col2 = st.columns(2)
        with col1:
            indicator_train = st.selectbox("Выберите показатель:",
                                           indicator, key='indicator_train'
                                           )

        with col2:
            region_train = st.selectbox('Выберите регион:', region,
                                        key='region_train'
                                        )

        plots(df_train, indicator_train, region_train)
        boxplot(df_train, indicator_train, region_train)

    with st.expander('Обучение моделей'):
        model_name = st.text_input('Введите название модели:', key='fit')

        st.write('Укажите гиперпараметры модели. Если поле будет пустым,'
                 'то значение будет выбрано по умолчанию')
        alpha, l1_ratio, max_iter, tol, eta0 = st.columns(5)

        with alpha:
            alpha = st.text_input('Введите alpha:', key='alpha')
        with l1_ratio:
            l1_ratio = st.text_input('Введите l1_ratio:', key='l1_ratio')
        with max_iter:
            max_iter = st.text_input('Введите max_iter:', key='max_iter')
        with tol:
            tol = st.text_input('Введите tol:', key='tol')
        with eta0:
            eta0 = st.text_input('Введите eta0:', key='eta0')

        alpha = alpha if alpha != '' else -1
        l1_ratio = l1_ratio if l1_ratio != '' else -1
        max_iter = max_iter if max_iter != '' else -1
        tol = tol if tol != '' else -1
        eta0 = eta0 if eta0 != '' else -1

        button1 = st.button('Начать обучение')

        data = {'model_name': model_name,
                'alpha': alpha,
                'l1_ratio': l1_ratio,
                'max_iter': max_iter,
                'tol': tol,
                'eta0': eta0
                }

        if button1:
            response = httpx.post("http://0.0.0.0:8000/fit",
                                  files={'file': csv_file_train},
                                  data=data,
                                  timeout=1000000
                                  )

            if response.status_code == 201:
                st.write(response.json()['message'])
                data = response.json()['data']

                params_col, metrics_col, coef_col = st.columns(3)

                with params_col:
                    st.write('Гиперпараметры модели:')
                    params = pd.DataFrame({'alpha': [data['alpha']],
                                    'l1_ratio': [data['l1_ratio']],
                                    'max_iter': [data['max_iter']],
                                    'tol': [data['tol']],
                                    'eta0': [data['eta0']]}
                                    ).T
                    params = params.rename(columns={0: 'value'})
                    st.write(params)

                with metrics_col:
                    st.write('Качество модели:')
                    metrics = pd.DataFrame({'r2': [data['r2']],
                                            'MSE': [data['MSE']],
                                            'RMSE': [data['RMSE']]}
                                            ).T
                    metrics = metrics.rename(columns={0: 'value'})
                    st.write(metrics)

                with coef_col:
                    st.write('Коэффициенты модели:')
                    coef = pd.DataFrame([data['coef']]).T
                    coef = coef.rename(columns={0: 'value'})
                    st.write(coef)
                
                st.write('Кривая обучения:')
                loss_list = data['loss_list']
                plt.figure(figsize=(18, 6))
                plt.plot(np.arange(len(loss_list)), loss_list)
                plt.xlabel("Time in epochs")
                plt.ylabel("Loss")
                st.pyplot(plt)

            else:
                st.write(response.json()['detail'])
    
    with st.expander('Сравнение моделей'):
        response = httpx.get("http://0.0.0.0:8000/list_models_for_comparison",
                            timeout=1000000
                            )
        model_list = response.json()['models']

        model_1, model_2 = st.columns(2)
        models_data_1 = None
        models_data_2 = None

        with model_1:
            model_name_1 = st.selectbox('Выберите модель 1:',
                                        model_list, key='model_1'
                                        )

        with model_2:
            model_name_2 = st.selectbox('Выберите модель 2:',
                                        model_list, key='model_2'
                                        )
        if model_name_1:
            response_1 = httpx.post("http://0.0.0.0:8000/compare_models",
                                        data={'model_name': model_name_1},
                                        timeout=1000000
                                        )
            models_data_1 = response_1.json()

        if model_name_2:
            response_2 = httpx.post("http://0.0.0.0:8000/compare_models",
                                        data={'model_name': model_name_2},
                                        timeout=1000000
                                        )
            models_data_2 = response_2.json()

        if models_data_1 and models_data_2:
            if model_name_1 == model_name_2:
                st.write('Модели должны быть разными')
            else:
                st.write('Гиперпараметры моделей:')
                params = pd.DataFrame({'alpha': [models_data_1['models_data']['alpha'],
                                                models_data_2['models_data']['alpha']
                                                ],
                                    'l1_ratio': [models_data_1['models_data']['l1_ratio'],
                                                models_data_2['models_data']['l1_ratio']
                                                ],
                                    'max_iter': [models_data_1['models_data']['max_iter'],
                                                models_data_2['models_data']['max_iter']
                                                ],
                                    'tol': [models_data_1['models_data']['tol'],
                                            models_data_2['models_data']['tol']
                                            ],
                                    'eta0': [models_data_1['models_data']['eta0'],
                                            models_data_2['models_data']['eta0']
                                            ]
                                            }
                                    ).T
                params = params.rename(columns={0: model_name_1, 1: model_name_2})
                st.write(params)

                st.write('Качество моделей:')
                metrics = pd.DataFrame({'r2': [models_data_1['models_data']['r2'],
                                            models_data_2['models_data']['r2']
                                            ],
                                        'MSE': [models_data_1['models_data']['MSE'],
                                                models_data_2['models_data']['MSE']
                                                ],
                                        'RMSE': [models_data_1['models_data']['RMSE'],
                                                models_data_2['models_data']['RMSE']
                                                ]
                                                }
                                        ).T
                metrics = metrics.rename(columns={0: model_name_1, 1: model_name_2})
                st.write(metrics)

                st.write('Коэффициенты моделей:')
                coef = pd.DataFrame([models_data_1['models_data']['coef'],
                                    models_data_2['models_data']['coef']
                                    ]
                                    ).T
                coef = coef.rename(columns={0: model_name_1, 1: model_name_2})
                st.write(coef)

                st.write('Кривая обучения:')
                loss_list_1 = models_data_1['models_data']['loss_list']
                loss_list_2 = models_data_2['models_data']['loss_list']
                fig = go.Figure()

                fig.add_trace(go.Scatter(x=np.arange(len(loss_list_1)), y=loss_list_1,
                                        mode='lines+markers', name=model_name_1,
                                        line=dict(color='blue')))

                fig.add_trace(go.Scatter(x=np.arange(len(loss_list_2)), y=loss_list_2,
                                        mode='lines+markers', name=model_name_2,
                                        line=dict(color='red')))


                fig.update_layout(title="Model Loss Over Epochs",
                                xaxis_title="Epochs",
                                yaxis_title="Loss",
                                xaxis=dict(showgrid=True),
                                yaxis=dict(showgrid=True))  

                st.plotly_chart(fig)

if file_predict is not None:
    df_predict = pd.read_csv(file_predict)
    csv_file_predict = df_predict.to_csv(index=False)

    with st.expander('Анализ загруженных данных для прогноза'):
        st.write(
            f'Размер датасета: {df_predict.shape[0]} строк и '
            f'{df_predict.shape[1]} признаков'
            )

        st.write('Описательная статистика:')
        st.write(df_predict.describe().T)

        st.write('Корреляционная матрица:')
        plt.figure(figsize=(20, 8))
        sns.heatmap(df_predict.select_dtypes(['float', 'int']).corr(),
                    cmap='coolwarm',
                    annot=True, fmt=".2f", linewidths=.5,
                    xticklabels=df_predict.select_dtypes(
                        ['float', 'int']).columns,
                    yticklabels=df_predict.select_dtypes(
                        ['float', 'int']).columns
                    )
        st.pyplot(plt)

        indicator = df_predict.select_dtypes(['float', 'int']).columns
        region = list(df_predict['region'].unique()) + ['Все регионы']

        col1, col2 = st.columns(2)
        with col1:
            indicator_predict = st.selectbox("Выберите показатель:", indicator,
                                             key='indicator_predict'
                                             )

        with col2:
            region_predict = st.selectbox('Выберите регион:', region,
                                          key='region_predict'
                                          )

        plots(df_predict, indicator_predict, region_predict)
        boxplot(df_predict, indicator_predict, region_predict)

    with st.expander('Построение прогноза'):
        response = httpx.get("http://0.0.0.0:8000/list_models",
                             timeout=1000000
                             )
        model_list = response.json()['models']
        select_model = st.selectbox('Выберите модель:', model_list)

        button6 = st.button('Прогноз на выбранной модели')

        if button6:
            response = httpx.post("http://0.0.0.0:8000/predict",
                                  files={'file': csv_file_predict},
                                  data={'model_name': select_model},
                                  timeout=1000000
                                  )
            if response.status_code == 200:
                st.write('Прогноз построен')
                st.download_button(label='Скачать прогноз',
                                   data=response.content,
                                   file_name=f'european_aqi_forecast_'
                                   f'{select_model}.csv',
                                   mime='text/csv'
                                   )
            else:
                st.write(response.json())
