"""
Модуль для визуализации данных
"""
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_2d_map(df: pd.DataFrame, show_well_names: bool = True) -> go.Figure:
    """
    Создает 2D карту скважин
    """
    fig = go.Figure()

    # Точки скважин
    fig.add_trace(go.Scatter(
        x=df["X"],
        y=df["Y"],
        mode="markers" + ("+text" if show_well_names else ""),
        text=df["Well"] if show_well_names else None,
        textposition="top center",
        marker=dict(
            size=15,
            color=df["Доля_коллектора"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(
                title="Доля коллектора"
            ),
            line=dict(width=2, color="black")
        ),
        hoverinfo="text",
        hovertext=[f"{row['Well']}<br>Доля: {row['Доля_коллектора']:.2%}"
                   for _, row in df.iterrows()],
        name="Скважины"
    ))

    # Настройки макета
    fig.update_layout(
        title="Карта скважин (вид сверху)",
        xaxis_title="Координата X",
        yaxis_title="Координата Y",
        hovermode="closest",
        template="plotly_white",
        height=600
    )

    # Настраиваем оси для равного масштаба
    fig.update_xaxes(
        scaleanchor="y",
        scaleratio=1,
        constrain="domain"
    )

    return fig


def create_3d_trajectories(trajectories: Dict[str, np.ndarray]) -> go.Figure:
    """
    Создает 3D визуализацию траекторий скважин

    Аргументы:
        trajectories: словарь с траекториями

    Возвращает:
        3D Figure для Plotly
    """
    fig = go.Figure()

    colors = px.colors.qualitative.Plotly

    for i, (well_name, trajectory) in enumerate(trajectories.items()):
        if len(trajectory) < 2:
            continue

        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter3d(
            x=trajectory[:, 0],
            y=trajectory[:, 1],
            z=trajectory[:, 2],
            mode="lines",
            name=well_name,
            line=dict(
                width=4,
                color=color
            ),
            hoverinfo="name+z",
            hovertemplate=f"{well_name}<br>Z: %{{z:.1f}}<extra></extra>"
        ))

        # Добавляем маркеры для начала и конца
        fig.add_trace(go.Scatter3d(
            x=[trajectory[0, 0], trajectory[-1, 0]],
            y=[trajectory[0, 1], trajectory[-1, 1]],
            z=[trajectory[0, 2], trajectory[-1, 2]],
            mode="markers",
            marker=dict(
                size=5,
                color=color
            ),
            showlegend=False,
            hoverinfo="skip"
        ))

    fig.update_layout(
        title="3D траектории скважин",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Глубина Z",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.7)
        ),
        height=700,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig


def create_las_cross_section(las_data: Dict, well_name: str = None) -> go.Figure:
    """
    Создает разрез скважины по данным LAS

    Аргументы:
        las_data: словарь с LAS-данными
        well_name: название скважины

    Возвращает:
        Figure с разрезом
    """
    if not las_data or 'depth' not in las_data:
        fig = go.Figure()
        fig.update_layout(
            title="Нет данных для отображения",
            xaxis_title="Значение",
            yaxis_title="Глубина"
        )
        return fig

    depth = las_data['depth']
    curve = las_data['curve']
    well_name = well_name or las_data.get('well_name', 'Неизвестная скважина')

    # Определяем цвет по значению кривой
    colors = []
    for val in curve:
        if val == 1:
            colors.append('yellow')  # Эффективный коллектор
        elif val == 0:
            colors.append('gray')  # Неэффективный коллектор/неколлектор
        else:
            colors.append('lightblue')  # Другие значения

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=curve,
        y=depth,
        mode='markers',
        marker=dict(
            size=8,
            color=colors,
            line=dict(width=1, color='black')
        ),
        name='ГИС',
        hovertemplate='Глубина: %{y:.1f}<br>Значение: %{x}<extra></extra>'
    ))

    fig.update_layout(
        title=f"Разрез скважины: {well_name}",
        xaxis_title="Значение ГИС (1=коллектор, 0=неколлектор)",
        yaxis_title="Глубина",
        yaxis=dict(autorange="reversed"),
        hovermode="y unified",
        height=600
    )

    # Добавляем горизонтальные линии для глубины
    for d in np.linspace(depth.min(), depth.max(), 10):
        fig.add_hline(
            y=d,
            line=dict(color="lightgray", width=1, dash="dot"),
            opacity=0.5
        )

    return fig


def create_prediction_heatmap(X_grid: np.ndarray, Y_grid: np.ndarray, Z_pred: np.ndarray) -> go.Figure:
    """
    Создает тепловую карту предсказаний

    Аргументы:
        X_grid, Y_grid: координаты сетки
        Z_pred: предсказанные значения

    Возвращает:
        Figure с тепловой картой
    """
    fig = go.Figure(data=go.Heatmap(
        z=Z_pred,
        x=X_grid[0, :],
        y=Y_grid[:, 0],
        colorscale="RdBu_r",
        colorbar=dict(title="Вероятность коллектора"),
        hovertemplate="X: %{x:.1f}<br>Y: %{y:.1f}<br>Вероятность: %{z:.2f}<extra></extra>"
    ))

    fig.update_layout(
        title="Карта предсказания коллектора",
        xaxis_title="Координата X",
        yaxis_title="Координата Y",
        height=500
    )

    return fig


def create_well_comparison(df: pd.DataFrame) -> go.Figure:
    """
    Создает сравнительную диаграмму скважин

    Аргументы:
        df: DataFrame с данными скважин

    Возвращает:
        Figure для сравнения
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["Well"],
        y=df["H"],
        name="Мощность пласта (H)",
        marker_color="lightblue",
        hovertemplate="Скважина: %{x}<br>Мощность: %{y:.1f}<extra></extra>"
    ))

    fig.add_trace(go.Bar(
        x=df["Well"],
        y=df["EFF_H"],
        name="Эффективная мощность (EFF_H)",
        marker_color="orange",
        hovertemplate="Скважина: %{x}<br>Эффективная мощность: %{y:.1f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=df["Well"],
        y=df["Доля_коллектора"] * 100,
        name="Доля коллектора (%)",
        yaxis="y2",
        mode="lines+markers",
        line=dict(color="red", width=3),
        marker=dict(size=10),
        hovertemplate="Скважина: %{x}<br>Доля: %{y:.1f}%<extra></extra>"
    ))

    fig.update_layout(
        title="Сравнение характеристик скважин",
        xaxis_title="Скважина",
        yaxis_title="Мощность (м)",
        yaxis2=dict(
            title="Доля коллектора (%)",
            overlaying="y",
            side="right"
        ),
        barmode="group",
        hovermode="x unified",
        height=500
    )

    return fig
