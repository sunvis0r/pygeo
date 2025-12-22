"""
Модуль для загрузки всех типов данных
"""
import os
from typing import Dict

import lasio
import numpy as np
import pandas as pd


def load_welltrajectories(filepath: str) -> Dict[str, np.ndarray]:
    """
    Загружает траектории скважин из файла

    Аргументы:
        filepath: путь к файлу с траекториями

    Возвращает:
        Словарь: {название_скважины: массив [X, Y, Z, MD]}
    """
    trajectories = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        current_well = None
        well_data = []

        for line in lines:
            line = line.strip()

            # Ищем начало новой скважины
            if 'welltrack' in line.lower() and "'" in line:
                # Сохраняем предыдущую скважину
                if current_well and well_data:
                    trajectories[current_well] = np.array(well_data)

                # Начинаем новую скважину
                current_well = line.split("'")[1]
                well_data = []

            # Пропускаем пустые строки и строки с комментариями
            elif not line or ';' in line:
                continue

            # Читаем данные траектории
            elif current_well:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        x, y, z, md = map(float, parts[:4])
                        well_data.append([x, y, z, md])
                    except ValueError:
                        continue

        # Сохраняем последнюю скважину
        if current_well and well_data:
            trajectories[current_well] = np.array(well_data)

    return trajectories


def load_h_data(filepath: str) -> pd.DataFrame:
    """
    Загружает данные мощности пласта (H)
    """
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        comment="#",
        names=["X", "Y", "Z", "Well", "H"]
    )
    # Преобразуем числовые колонки
    for col in ['X', 'Y', 'Z', 'H']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def load_eff_h_data(filepath: str) -> pd.DataFrame:
    """
    Загружает данные эффективной мощности (EFF_H)
    """
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        comment="#",
        names=["X", "Y", "Z", "Well", "EFF_H"]
    )
    # Преобразуем числовые колонки
    for col in ['X', 'Y', 'Z', 'EFF_H']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def load_las_file(filepath: str) -> Dict:
    """
    Загружает LAS-файл с данными ГИС

    Аргументы:
        filepath: путь к LAS-файлу

    Возвращает:
        Словарь с данными: {'depth': [], 'curve': [], 'well_name': str}
    """
    try:
        las = lasio.read(filepath)
        well_name = las.well.WELL.value if hasattr(las.well, 'WELL') else os.path.basename(filepath).replace('.las',
                                                                                                             '').replace(
            '.txt', '')

        data = {
            'well_name': well_name,
            'depth': las['DEPT'] if 'DEPT' in las.curves else las.index,
            'curve': las['КриваяГИС1'] if 'КриваяГИС1' in las.curves else las.curves[1].data,
            'null_value': -999.25
        }
        return data
    except Exception as e:
        print(f"Ошибка при загрузке {filepath}: {e}")
        return None


def load_all_las_files(folder_path: str) -> Dict[str, Dict]:
    """
    Загружает все LAS-файлы из папки

    Аргументы:
        folder_path: путь к папке с LAS-файлами

    Возвращает:
        Словарь: {название_скважины: данные_лас}
    """
    las_data = {}

    if not os.path.exists(folder_path):
        print(f"Папка {folder_path} не существует!")
        return las_data

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.las', '.txt')):
            filepath = os.path.join(folder_path, filename)
            data = load_las_file(filepath)
            if data:
                las_data[data['well_name']] = data

    return las_data


def combine_all_data(h_path: str, eff_h_path: str) -> pd.DataFrame:
    """
    Объединяет данные H и EFF_H в один DataFrame

    Возвращает:
        DataFrame с колонками: X, Y, Z, Well, H, EFF_H, Доля_коллектора
    """
    df_h = load_h_data(h_path)
    df_eff = load_eff_h_data(eff_h_path)

    # Объединяем по названию скважины (left join, чтобы сохранить все скважины из H)
    # Координаты должны быть одинаковыми для одной скважины
    df = pd.merge(df_h, df_eff, on="Well", how='left', suffixes=('_H', '_EFF'))

    # Оставляем координаты из файла H, переименовываем колонки
    df = df.rename(columns={'X_H': 'X', 'Y_H': 'Y', 'Z_H': 'Z'})

    # Удаляем дублирующие колонки координат из EFF_H (если они есть)
    columns_to_drop = ['X_EFF', 'Y_EFF', 'Z_EFF']
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    if existing_columns:
        df = df.drop(columns=existing_columns)

    # Удаляем строки с пустыми именами скважин
    df = df.dropna(subset=['Well'])

    # Рассчитываем долю коллектора (защита от деления на ноль и NaN)
    def calculate_collector_ratio(row):
        h = row["H"]
        eff_h = row["EFF_H"]
        if pd.isna(h) or pd.isna(eff_h) or h <= 0:
            return None  # или 0, в зависимости от требований
        return eff_h / h

    df["Доля_коллектора"] = df.apply(calculate_collector_ratio, axis=1)

    # Сортируем по названию скважины
    df = df.sort_values("Well")

    return df
