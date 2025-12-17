"""
Модуль для ML предсказаний - заглушка для будущего API
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class MLPredictor:
    """
    Заглушка для ML модели предсказания коллекторских свойств
    В будущем будет заменена на реальный API вызов
    """

    def __init__(self):
        """Инициализация ML предиктора"""
        # Заглушка - в будущем здесь будет загрузка модели
        self.model_loaded = True
        print("🤖 ML Predictor initialized (mock mode)")

    def predict_collector_zones(self, well_name: str, x: float, y: float,
                               depth_range: Tuple[float, float],
                               num_points: int = 50) -> Dict[str, np.ndarray]:
        """
        Предсказывает коллекторские зоны для новой скважины

        Args:
            well_name: Название скважины
            x, y: Координаты скважины
            depth_range: Диапазон глубин (min_depth, max_depth)
            num_points: Количество точек предсказания

        Returns:
            Словарь с массивами глубин и предсказанных значений
        """

        min_depth, max_depth = depth_range

        # Создаем массив глубин
        depths = np.linspace(min_depth, max_depth, num_points)

        # Генерируем предсказания (заглушка)
        # В реальности здесь будет вызов ML модели
        np.random.seed(hash(well_name) % 2**32)  # Детерминированный сид для консистентности

        # Имитируем геологическую структуру:
        # - Случайные коллекторские зоны
        # - Переходы между коллектором и неколлектором
        # ВАЖНО: Возвращаем только 0 или 1 (бинарная классификация)
        predictions = []

        current_zone = 0  # 0 = неколлектор, 1 = коллектор
        zone_length = np.random.randint(5, 15)  # длина зоны в точках

        for i, depth in enumerate(depths):
            if i % zone_length == 0:
                # Меняем тип зоны
                current_zone = 1 - current_zone
                zone_length = np.random.randint(5, 15)

            # Возвращаем строго 0 или 1 (без шума)
            # 0 = неколлектор, 1 = коллектор
            predictions.append(current_zone)

        predictions = np.array(predictions, dtype=int)

        return {
            'depth': depths,
            'prediction': predictions,
            'well_name': well_name,
            'x': x,
            'y': y
        }

    def predict_multiple_wells(self, wells_data: List[Dict],
                              depth_range: Tuple[float, float] = (-200, 0),
                              num_points: int = 50) -> Dict[str, Dict]:
        """
        Предсказывает коллекторские зоны для нескольких скважин

        Args:
            wells_data: Список словарей с данными скважин
                       [{'name': str, 'x': float, 'y': float}, ...]
            depth_range: Диапазон глубин для предсказания
            num_points: Количество точек на скважину

        Returns:
            Словарь предсказаний по названиям скважин
        """

        predictions = {}

        for well_data in wells_data:
            well_name = well_data['name']
            x = well_data['x']
            y = well_data['y']

            pred = self.predict_collector_zones(
                well_name=well_name,
                x=x, y=y,
                depth_range=depth_range,
                num_points=num_points
            )

            predictions[well_name] = pred

        return predictions

    def get_prediction_stats(self, predictions: Dict[str, Dict]) -> Dict:
        """
        Вычисляет статистику по предсказаниям

        Args:
            predictions: Словарь предсказаний от predict_multiple_wells

        Returns:
            Статистика предсказаний
        """

        if not predictions:
            return {}

        all_predictions = []
        collector_ratios = []

        for well_name, pred_data in predictions.items():
            preds = pred_data['prediction']
            all_predictions.extend(preds)

            # Вычисляем долю коллектора (порог 0.5)
            collector_ratio = np.mean(preds > 0.5)
            collector_ratios.append(collector_ratio)

        return {
            'total_points': len(all_predictions),
            'mean_prediction': np.mean(all_predictions),
            'std_prediction': np.std(all_predictions),
            'collector_ratio_mean': np.mean(collector_ratios),
            'collector_ratio_std': np.std(collector_ratios),
            'num_wells': len(predictions)
        }


# Глобальный экземпляр предиктора
ml_predictor = MLPredictor()

