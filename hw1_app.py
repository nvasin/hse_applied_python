import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import io
import datetime

def dA(df: pd.DataFrame, thr: float = 2.0) -> pd.DataFrame:
    stt = df.groupby(["city", "season"])["temperature"].agg(["mean", "std"]).reset_index()
    stt.rename(columns={"mean": "s_mean", "std": "s_std"}, inplace=True)
    df_ = pd.merge(df, stt, on=["city", "season"], how="left")
    df_["anomaly"] = np.abs(df_["temperature"] - df_["s_mean"]) > thr * df_["s_std"]
    return df_

def getT(city: str, api_key: str, lat: float, lon: float) -> float:
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    resp = requests.get(url, params=params)
    if resp.status_code == 200:
        data = resp.json()
        return data["main"]["temp"]
    else:
        st.error(f"Ошибка {resp.status_code}: {resp.text}")
        raise ValueError(resp.json())

def isNorm(temp_now: float, mean_season: float, std_season: float, thr: float = 2.0) -> bool:
    if np.isnan(mean_season) or np.isnan(std_season):
        return True
    return abs(temp_now - mean_season) <= thr * std_season

def get_coords(city: str, api_key: str):
    geocode_url = f'http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={api_key}'
    geocode_response = requests.get(geocode_url)
    if geocode_response.status_code == 200:
        geocode_data = geocode_response.json()
        if geocode_data:
            return geocode_data[0]['lat'], geocode_data[0]['lon']
        else:
            st.error('Не удалось получить данные геокодирования. Проверьте название города.')
            return None, None
    else:
        st.error(f'Ошибка при геокодировании: {geocode_response.status_code}')
        raise ValueError(geocode_response.json())

def main():
    st.title("Анализ температур и аномалий")

    st.subheader("Шаг 1: Загрузите исторические данные (CSV)")
    file = st.file_uploader("Выберите файл с данными", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)
        st.write("Пример загруженных данных:")
        st.write(df.head())

        if not {"city", "season", "temperature", "timestamp"}.issubset(df.columns):
            st.error("Ошибка! В файле не хватает столбцов city, season, temperature или timestamp")
            return

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])

        city_list = sorted(df["city"].unique())
        sel_city = st.selectbox("Выберите город", city_list)

        df_city = df[df["city"] == sel_city].copy()
        if df_city.empty:
            st.warning("Нет данных по выбранному городу.")
            return

        st.subheader("Введите API ключ OpenWeatherMap (необязательно)")
        key = st.text_input("API Key")

        st.subheader(f"Описательные статистики для {sel_city}")
        st.write(df_city["temperature"].describe())

        df_det = dA(df.copy())
        df_city_anom = df_det[df_det["city"] == sel_city].copy()

        st.subheader("Временной ряд температур с аномалиями")

        if "timestamp" in df_city_anom.columns:
            df_city_anom.sort_values("timestamp", inplace=True)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(
                df_city_anom["timestamp"], 
                df_city_anom["temperature"], 
                label="Температура", 
                marker="o", 
                linestyle="-"
            )

            anoms = df_city_anom[df_city_anom["anomaly"] == True]

            if not anoms.empty:
                st.write(f"Обнаружено аномалий: {len(anoms)}")

            ax.scatter(
                anoms["timestamp"], 
                anoms["temperature"], 
                color="red", 
                label="Аномалия",
                zorder=5
            )

            ax.set_title(f"{sel_city}: временной ряд")
            ax.set_xlabel("Дата")
            ax.set_ylabel("Температура (°C)")
            ax.legend()
            plt.xticks(rotation=30)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Нет столбца 'timestamp' — не можем отрисовать временной ряд.")

        st.subheader("Сезонные профили (среднее и стандартное отклонение)")
        df_city["month"] = df_city["timestamp"].dt.month
        monthly_stats = df_city.groupby("month")["temperature"].agg(["mean", "std"])
        st.write(monthly_stats)

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(
            monthly_stats.index, 
            monthly_stats["mean"], 
            yerr=monthly_stats["std"], 
            capsize=5, 
            color='skyblue'
        )
        ax2.set_xlabel("Месяц")
        ax2.set_ylabel("Температура (°C)")
        ax2.set_title(f"{sel_city}: средняя температура по месяцам (±1σ)")
        plt.tight_layout()
        st.pyplot(fig2)

        if key:
            try:
                lat_city, lon_city = get_coords(sel_city, key)
                if lat_city is not None and lon_city is not None:
                    cur_temp = getT(sel_city, key, lat_city, lon_city)
                    if cur_temp is not None:
                        st.write(f"Текущая температура по API: {cur_temp:.1f} °C")
                        current_month = datetime.datetime.now().month
                        if current_month in monthly_stats.index:
                            monthly_mean = monthly_stats.loc[current_month, "mean"]
                            monthly_std = monthly_stats.loc[current_month, "std"]

                            if abs(cur_temp - monthly_mean) <= monthly_std:
                                st.success("Текущая температура в пределах нормы для текущего месяца.")
                            else:
                                st.warning("Текущая температура аномальна для текущего месяца!")
                        else:
                            st.info("Нет данных для текущего месяца для сравнения.")
            except ValueError as e:
                st.error(f"Ошибка при запросе API: {e}")
        else:
            st.info("API ключ не введён. Текущая температура не будет получена.")
    else:
        st.info("Пожалуйста, загрузите CSV-файл, чтобы продолжить.")

if __name__ == "__main__":
    main()