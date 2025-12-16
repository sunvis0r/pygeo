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


def create_3d_reservoir_layers(well_data: pd.DataFrame, trajectories: Dict[str, np.ndarray] = None,
                                las_data: Dict[str, Dict] = None, show_trajectories: bool = True,
                                show_vertical_layers: bool = True, show_well_logs: bool = True) -> go.Figure:
    """
    Создает 3D визуализацию траекторий скважин с окраской по типу коллектора
    Траектории окрашиваются: зеленый = коллектор, серый = неколлектор
    
    Аргументы:
        well_data: DataFrame с данными скважин (X, Y, Z, H, EFF_H)
        trajectories: словарь с траекториями скважин (обязательно)
        las_data: словарь с LAS-данными (обязательно для окраски)
        show_trajectories: не используется (оставлено для совместимости)
        show_vertical_layers: не используется (оставлено для совместимости)
        show_well_logs: показывать окраску коллекторов на траекториях
    
    Возвращает:
        3D Figure с траекториями, окрашенными по типу коллектора
    """
    fig = go.Figure()
    
    if not trajectories:
        return fig
    
    layers_added = 0
    wells_processed = 0
    
    # Обрабатываем каждую скважину
    for well_name, trajectory in trajectories.items():
        if len(trajectory) < 2:
            continue
        
        wells_processed += 1
        traj_x = trajectory[:, 0]
        traj_y = trajectory[:, 1]
        traj_z = trajectory[:, 2]
        traj_md = trajectory[:, 3]
        
        # Если нужно показать слои коллекторов И есть LAS данные
        if show_well_logs and las_data and well_name in las_data:
            las = las_data[well_name]
            depth = las['depth']  # MD
            curve = las['curve']
            null_value = las.get('null_value', -999.25)
            
            # Фильтруем валидные данные
            valid_mask = (curve != null_value) & (~np.isnan(curve))
            
            if np.any(valid_mask):
                depth_valid = depth[valid_mask]
                curve_valid = curve[valid_mask]
                
                # Проверяем диапазоны MD
                las_md_min, las_md_max = depth_valid.min(), depth_valid.max()
                traj_md_min, traj_md_max = traj_md.min(), traj_md.max()
                
                # Если диапазоны пересекаются
                if not (las_md_max < traj_md_min or las_md_min > traj_md_max):
                    # Интерполируем координаты по MD
                    x_coords = np.interp(depth_valid, traj_md, traj_x)
                    y_coords = np.interp(depth_valid, traj_md, traj_y)
                    z_coords = np.interp(depth_valid, traj_md, traj_z)
                    
                    # Рисуем сегменты траектории с окраской по типу коллектора
                    i = 0
                    while i < len(curve_valid):
                        current_value = curve_valid[i]
                        start_idx = i
                        
                        # Находим конец текущего сегмента
                        while i < len(curve_valid) and curve_valid[i] == current_value:
                            i += 1
                        end_idx = i - 1
                        
                        # Определяем цвет и ширину
                        if current_value == 1:  # Коллектор
                            color = 'green'
                            width = 8
                            name = 'Коллектор'
                        elif current_value == 0:  # Неколлектор
                            color = 'gray'
                            width = 6
                            name = 'Неколлектор'
                        else:
                            continue
                        
                        # Рисуем сегмент траектории
                        segment_x = x_coords[start_idx:end_idx+1]
                        segment_y = y_coords[start_idx:end_idx+1]
                        segment_z = z_coords[start_idx:end_idx+1]
                        
                        fig.add_trace(go.Scatter3d(
                            x=segment_x,
                            y=segment_y,
                            z=segment_z,
                            mode='lines',
                            line=dict(color=color, width=width),
                            name=well_name if layers_added == 0 else None,
                            showlegend=(layers_added == 0),
                            legendgroup=well_name,
                            hovertemplate=f"{well_name}<br>{name}<br>Z: %{{z:.1f}}<extra></extra>"
                        ))
                        layers_added += 1
                    
                    # Добавляем маркеры начала и конца траектории
                    fig.add_trace(go.Scatter3d(
                        x=[traj_x[0], traj_x[-1]],
                        y=[traj_y[0], traj_y[-1]],
                        z=[traj_z[0], traj_z[-1]],
                        mode="markers",
                        marker=dict(
                            size=6,
                            color=['blue', 'red'],
                            symbol=['circle', 'diamond']
                        ),
                        showlegend=False,
                        hoverinfo="skip"
                    ))
                    continue
        
        # Если нет LAS данных или не нужно показывать слои - рисуем обычную траекторию
        fig.add_trace(go.Scatter3d(
            x=traj_x,
            y=traj_y,
            z=traj_z,
            mode="lines",
            name=well_name,
            line=dict(
                width=4,
                color='lightblue'
            ),
            hoverinfo="name+z",
            hovertemplate=f"{well_name}<br>Z: %{{z:.1f}}<br>MD: %{{customdata:.1f}}<extra></extra>",
            customdata=traj_md
        ))
        
        # Маркеры начала и конца
        fig.add_trace(go.Scatter3d(
            x=[traj_x[0], traj_x[-1]],
            y=[traj_y[0], traj_y[-1]],
            z=[traj_z[0], traj_z[-1]],
            mode="markers",
            marker=dict(
                size=6,
                color=['blue', 'red'],
                symbol=['circle', 'diamond']
            ),
            showlegend=False,
            hoverinfo="skip"
        ))
    
    # Добавляем легенду для типов коллекторов
    if layers_added > 0:
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='lines',
            line=dict(color='green', width=8),
            name='Коллектор (1)',
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='lines',
            line=dict(color='gray', width=6),
            name='Неколлектор (0)',
            showlegend=True
        ))
    
    # Формируем заголовок
    title_text = "3D визуализация пластов-коллекторов"
    if layers_added > 0:
        title_text += f" ({wells_processed} скважин, {layers_added} сегментов)"
    else:
        title_text += f" ({wells_processed} скважин)"
    
    # Настройки макета
    fig.update_layout(
        title={
            'text': title_text,
            'x': 0.5,
            'xanchor': 'center'
        },
        scene=dict(
            xaxis_title="X (м)",
            yaxis_title="Y (м)",
            zaxis_title="Глубина Z (м)",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.7),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                center=dict(x=0, y=0, z=-0.1)
            ),
            xaxis=dict(backgroundcolor="rgb(240, 240, 240)", gridcolor="white"),
            yaxis=dict(backgroundcolor="rgb(240, 240, 240)", gridcolor="white"),
            zaxis=dict(backgroundcolor="rgb(240, 240, 240)", gridcolor="white")
        ),
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig


def create_2d_well_projection(well_data: pd.DataFrame, las_data: Dict[str, Dict],
                               selected_well: str, trajectories: Dict[str, np.ndarray] = None) -> go.Figure:
    """
    Создает 2D проекцию скважины с отображением слоев коллекторов
    
    Аргументы:
        well_data: DataFrame с данными скважин
        las_data: словарь с LAS-данными
        selected_well: название выбранной скважины
        trajectories: словарь с траекториями скважин (опционально)
    
    Возвращает:
        2D Figure с проекцией скважины и слоями
    """
    fig = go.Figure()
    
    # Получаем данные выбранной скважины
    well_row = well_data[well_data["Well"] == selected_well]
    if well_row.empty:
        fig.add_annotation(
            text=f"Скважина {selected_well} не найдена",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    well_row = well_row.iloc[0]
    z_top = well_row["Z"]
    z_bottom = well_row["Z"] - well_row["H"]
    h_total = well_row["H"]
    eff_h = well_row["EFF_H"]
    
    # Проверяем наличие LAS данных
    if selected_well not in las_data:
        fig.add_annotation(
            text=f"LAS данные для {selected_well} не найдены",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    las = las_data[selected_well]
    depth = las['depth']  # MD - measured depth (длина трубы)
    curve = las['curve']
    null_value = las.get('null_value', -999.25)
    
    # Фильтруем валидные данные
    valid_mask = (curve != null_value) & (~np.isnan(curve))
    if not np.any(valid_mask):
        fig.add_annotation(
            text=f"Нет валидных данных для {selected_well}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    depth_valid = depth[valid_mask]  # MD values
    curve_valid = curve[valid_mask]
    
    # Нормализуем глубину к диапазону скважины
    depth_min, depth_max = depth_valid.min(), depth_valid.max()
    if depth_max - depth_min < 0.1:
        fig.add_annotation(
            text=f"Недостаточно данных для {selected_well}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Преобразуем MD в Z координаты
    if trajectories and selected_well in trajectories:
        # Используем траекторию для точного преобразования MD -> Z
        trajectory = trajectories[selected_well]
        traj_md = trajectory[:, 3]  # MD из траектории
        traj_z = trajectory[:, 2]   # Z из траектории
        
        # КРИТИЧНО: Проверяем соответствие диапазонов MD
        las_md_min, las_md_max = depth_valid.min(), depth_valid.max()
        traj_md_min, traj_md_max = traj_md.min(), traj_md.max()
        
        # Проверка на выход за пределы траектории
        if las_md_min < traj_md_min - 1.0 or las_md_max > traj_md_max + 1.0:
            # Попытка использовать относительные MD
            md_offset = las_md_min - traj_md_min
            if abs(md_offset) > h_total * 2:  # Слишком большое смещение
                # Используем fallback
                z_coords = z_top - (depth_valid - depth_min) * (h_total / (depth_max - depth_min))
            else:
                # Интерполируем с учетом смещения
                z_coords = np.interp(depth_valid, traj_md, traj_z)
        else:
            # Интерполируем Z координаты по MD из траектории
            z_coords = np.interp(depth_valid, traj_md, traj_z)
        
        # Проверка: если скважина вертикальная (малая вариация Z)
        if np.std(z_coords) < 0.1:
            # Вертикальная скважина - используем линейное приближение
            z_coords = z_top - (depth_valid - las_md_min) * (h_total / (las_md_max - las_md_min))
    else:
        # Fallback: линейное приближение (работает только для вертикальных скважин)
        # Предполагаем, что MD в LAS соответствует глубине от кровли пласта
        z_coords = z_top - (depth_valid - depth_min) * (h_total / (depth_max - depth_min))
    
    # 1. Рисуем ствол скважины (вертикальная линия)
    fig.add_trace(go.Scatter(
        x=[0, 0],
        y=[z_top, z_bottom],
        mode='lines',
        line=dict(color='black', width=4),
        name='Ствол скважины',
        showlegend=True
    ))
    
    # 2. Рисуем слои коллекторов и неколлекторов
    # Группируем последовательные значения
    i = 0
    while i < len(curve_valid):
        current_value = curve_valid[i]
        start_idx = i
        
        # Находим конец текущего слоя
        while i < len(curve_valid) and curve_valid[i] == current_value:
            i += 1
        end_idx = i - 1
        
        if current_value == 1:  # Коллектор
            color = 'green'
            name = 'Коллектор'
            width = 40
        elif current_value == 0:  # Неколлектор
            color = 'gray'
            name = 'Неколлектор'
            width = 40
        else:
            continue
        
        # Рисуем прямоугольник слоя
        z_start = z_coords[start_idx]
        z_end = z_coords[end_idx]
        
        fig.add_trace(go.Scatter(
            x=[-width/2, width/2, width/2, -width/2, -width/2],
            y=[z_start, z_start, z_end, z_end, z_start],
            fill='toself',
            fillcolor=color,
            line=dict(color=color, width=1),
            mode='lines',
            name=name,
            showlegend=False,
            opacity=0.7,
            hovertemplate=f'{name}<br>Глубина: %{{y:.1f}}<extra></extra>'
        ))
    
    # 3. Добавляем каротажную кривую (красная линия)
    # Масштабируем кривую для отображения
    curve_normalized = (curve_valid - curve_valid.min()) / (curve_valid.max() - curve_valid.min() + 0.001)
    curve_scaled = curve_normalized * 30 - 15  # Центрируем относительно ствола
    
    fig.add_trace(go.Scatter(
        x=curve_scaled,
        y=z_coords,
        mode='lines',
        line=dict(color='red', width=2),
        name='Каротажная кривая',
        showlegend=True,
        hovertemplate='Значение: %{x:.2f}<br>Глубина: %{y:.1f}<extra></extra>'
    ))
    
    # 4. Добавляем маркеры кровли и подошвы
    fig.add_trace(go.Scatter(
        x=[0, 0],
        y=[z_top, z_bottom],
        mode='markers+text',
        marker=dict(size=10, color=['blue', 'red'], symbol=['triangle-up', 'triangle-down']),
        text=['Кровля', 'Подошва'],
        textposition=['top center', 'bottom center'],
        name='Границы пласта',
        showlegend=True
    ))
    
    # 5. Добавляем легенду для коллектора и неколлектора
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=15, color='green', symbol='square'),
        name='Коллектор (1)',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=15, color='gray', symbol='square'),
        name='Неколлектор (0)',
        showlegend=True
    ))
    
    # Настройки макета
    fig.update_layout(
        title=f"2D проекция скважины {selected_well}",
        xaxis_title="Отклонение (м)",
        yaxis_title="Глубина Z (м)",
        height=700,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        hovermode='closest',
        template='plotly_white'
    )
    
    # Добавляем аннотации с информацией
    fig.add_annotation(
        text=f"Мощность пласта: {h_total:.2f} м<br>" +
             f"Эффективная мощность: {eff_h:.2f} м<br>" +
             f"Доля коллектора: {(eff_h/h_total*100):.1f}%",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1
    )
    
    return fig
