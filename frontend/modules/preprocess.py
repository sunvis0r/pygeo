"""
Модуль для предобработки данных
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def clean_las_data(las_data: Dict) -> Dict:
    """
    Очищает данные LAS от пропущенных значений

    Аргументы:
        las_data: словарь с LAS-данными

    Возвращает:
        Очищенный словарь
    """
    if not las_data:
        return las_data

    # Маски для валидных данных
    valid_mask = las_data['curve'] != las_data['null_value']

    return {
        'well_name': las_data['well_name'],
        'depth': las_data['depth'][valid_mask],
        'curve': las_data['curve'][valid_mask],
        'null_value': las_data['null_value']
    }


def interpolate_trajectory(trajectory: np.ndarray, step: float = 1.0) -> np.ndarray:
    """
    Интерполирует траекторию скважины с заданным шагом

    Аргументы:
        trajectory: массив [X, Y, Z, MD]
        step: шаг интерполяции по MD

    Возвращает:
        Интерполированную траекторию
    """
    if len(trajectory) < 2:
        return trajectory

    md = trajectory[:, 3]
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]

    # Создаем новые точки по MD
    new_md = np.arange(md[0], md[-1], step)

    # Интерполируем координаты
    new_x = np.interp(new_md, md, x)
    new_y = np.interp(new_md, md, y)
    new_z = np.interp(new_md, md, z)

    return np.column_stack([new_x, new_y, new_z, new_md])


def create_grid_from_points(df: pd.DataFrame, grid_size: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Создает регулярную сетку для интерполяции

    Аргументы:
        df: DataFrame с точками
        grid_size: размер сетки

    Возвращает:
        X_grid, Y_grid, Z_grid
    """
    # Преобразуем колонки в числа, если они строки
    df = df.copy()
    for col in ['X', 'Y', 'Z']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Удаляем строки с NaN
    df = df.dropna(subset=['X', 'Y'])

    x_min, x_max = df['X'].min(), df['X'].max()
    y_min, y_max = df['Y'].min(), df['Y'].max()

    # Добавляем небольшой отступ
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1

    x_range = np.linspace(x_min - x_pad, x_max + x_pad, grid_size)
    y_range = np.linspace(y_min - y_pad, y_max + y_pad, grid_size)

    X_grid, Y_grid = np.meshgrid(x_range, y_range)

    return X_grid, Y_grid


def prepare_ml_data(df: pd.DataFrame, las_dict: Dict) -> Dict:
    """
    Подготавливает данные для ML-модели

    Аргументы:
        df: объединенный DataFrame с H/EFF_H
        las_dict: словарь с LAS-данными

    Возвращает:
        Словарь с подготовленными данными
    """
    ml_data = {
        'coordinates': df[['X', 'Y', 'Z']].values,
        'labels': df['Доля_коллектора'].values,
        'well_names': df['Well'].values,
        'las_data': {}
    }

    # Добавляем LAS-данные для каждой скважины
    for well_name in df['Well'].unique():
        if well_name in las_dict:
            clean_data = clean_las_data(las_dict[well_name])
            ml_data['las_data'][well_name] = {
                'depth': clean_data['depth'].tolist(),
                'curve': clean_data['curve'].tolist()
            }

    return ml_data


def filter_by_depth(las_data: Dict, min_depth: float = None, max_depth: float = None) -> Dict:
    """
    Фильтрует LAS-данные по глубине

    Аргументы:
        las_data: словарь с LAS-данными
        min_depth: минимальная глубина
        max_depth: максимальная глубина

    Возвращает:
        Отфильтрованные данные
    """
    if not las_data or 'depth' not in las_data:
        return las_data

    depth = las_data['depth']
    curve = las_data['curve']

    # Создаем маску для фильтрации
    mask = np.ones_like(depth, dtype=bool)

    if min_depth is not None:
        mask = mask & (depth >= min_depth)

    if max_depth is not None:
        mask = mask & (depth <= max_depth)

    return {
        'well_name': las_data['well_name'],
        'depth': depth[mask],
        'curve': curve[mask],
        'null_value': las_data['null_value']
    }
