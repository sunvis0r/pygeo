"""
Модуль для визуализации данных
"""
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .ml_predictor import ml_predictor


def create_2d_map(df: pd.DataFrame, trajectories: Dict[str, np.ndarray] = None,
                  show_well_names: bool = True, show_trajectories: bool = True) -> go.Figure:
    """
    Создает 2D карту ВСЕХ скважин с траекториями (вид сверху - проекция XY)
    
    Аргументы:
        df: DataFrame с данными скважин (для окраски по доле коллектора)
        trajectories: словарь с траекториями скважин (опционально)
        show_well_names: показывать названия скважин
        show_trajectories: показывать траектории скважин
    """
    fig = go.Figure()
    
    # 1. Сначала рисуем траектории скважин (если есть)
    if show_trajectories and trajectories:
        colors = px.colors.qualitative.Plotly
        
        for i, (well_name, trajectory) in enumerate(trajectories.items()):
            if len(trajectory) < 2:
                continue
            
            traj_x = trajectory[:, 0]
            traj_y = trajectory[:, 1]
            
            color = colors[i % len(colors)]
            
            # Рисуем траекторию (тонкая линия)
            fig.add_trace(go.Scatter(
                x=traj_x,
                y=traj_y,
                mode='lines',
                line=dict(color=color, width=2),
                name=well_name,
                showlegend=False,
                hoverinfo='skip',
                opacity=0.6
            ))
            
            # Добавляем стрелку направления (от начала к концу)
            # Берем последние 2 точки для направления
            if len(traj_x) >= 2:
                # Вектор направления
                dx = traj_x[-1] - traj_x[-2]
                dy = traj_y[-1] - traj_y[-2]
                
                # Нормализуем и масштабируем для стрелки
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    dx = dx / length * 50  # Длина стрелки
                    dy = dy / length * 50
                    
                    # Рисуем стрелку в конце траектории
                    fig.add_annotation(
                        x=traj_x[-1],
                        y=traj_y[-1],
                        ax=traj_x[-1] - dx,
                        ay=traj_y[-1] - dy,
                        xref='x',
                        yref='y',
                        axref='x',
                        ayref='y',
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor=color,
                        opacity=0.7
                    )

    # 2. Затем рисуем точки ВСЕХ скважин (поверх траекторий)
    # Объединяем данные из df (well_data) и trajectories
    all_wells_x = []
    all_wells_y = []
    all_wells_names = []
    all_wells_colors = []
    all_wells_hover = []
    
    # Сначала добавляем скважины из df (с данными о коллекторе)
    for _, row in df.iterrows():
        all_wells_x.append(row["X"])
        all_wells_y.append(row["Y"])
        all_wells_names.append(row["Well"])
        all_wells_colors.append(row["Доля_коллектора"])
        all_wells_hover.append(
            f"{row['Well']}<br>X: {row['X']:.1f}<br>Y: {row['Y']:.1f}<br>"
            f"H: {row['H']:.2f} м<br>EFF_H: {row['EFF_H']:.2f} м<br>"
            f"Доля: {row['Доля_коллектора']:.2%}"
        )
    
    # Затем добавляем скважины из траекторий, которых нет в df
    if trajectories:
        df_wells = set(df["Well"].values)
        for well_name, trajectory in trajectories.items():
            if len(trajectory) == 0:
                continue
            
            # Если скважины нет в df - добавляем из траектории
            if well_name not in df_wells:
                x_start = trajectory[0, 0]
                y_start = trajectory[0, 1]
                
                all_wells_x.append(x_start)
                all_wells_y.append(y_start)
                all_wells_names.append(well_name)
                all_wells_colors.append(0.5)  # Средний цвет для скважин без данных
                all_wells_hover.append(
                    f"{well_name}<br>X: {x_start:.1f}<br>Y: {y_start:.1f}<br>"
                    f"Нет данных о мощности и коллекторе"
                )
    
    # Рисуем все точки скважин
    fig.add_trace(go.Scatter(
        x=all_wells_x,
        y=all_wells_y,
        mode="markers" + ("+text" if show_well_names else ""),
        text=all_wells_names if show_well_names else None,
        textposition="top center",
        marker=dict(
            size=15,
            color=all_wells_colors,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(
                title="Доля коллектора"
            ),
            line=dict(width=2, color="black"),
            cmin=0,
            cmax=1
        ),
        hoverinfo="text",
        hovertext=all_wells_hover,
        name="Скважины",
        showlegend=False
    ))

    # Настройки макета
    title_text = "Карта скважин (вид сверху)"
    if show_trajectories and trajectories:
        title_text += f" - {len(trajectories)} скважин с траекториями"
    
    fig.update_layout(
        title=title_text,
        xaxis_title="Координата X (м)",
        yaxis_title="Координата Y (м)",
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

        # ML скважины отображаются розовым цветом
        if well_name.startswith("ML_"):
            color = 'hotpink'
        else:
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
        # ML скважины: розовый (начало) и фиолетовый (конец)
        if well_name.startswith("ML_"):
            marker_colors = ['hotpink', 'purple']
        else:
            marker_colors = [color, color]
        
        fig.add_trace(go.Scatter3d(
            x=[trajectory[0, 0], trajectory[-1, 0]],
            y=[trajectory[0, 1], trajectory[-1, 1]],
            z=[trajectory[0, 2], trajectory[-1, 2]],
            mode="markers",
            marker=dict(
                size=5,
                color=marker_colors
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


def create_3d_reservoir_layers(well_data: pd.DataFrame = None, trajectories: Dict[str, np.ndarray] = None,
                                las_data: Dict[str, Dict] = None, show_trajectories: bool = True,
                                show_vertical_layers: bool = True, show_well_logs: bool = True) -> go.Figure:
    """
    Создает 3D визуализацию траекторий скважин с наложением слоев коллекторов
    
    Двухслойная визуализация:
    1. Базовый слой: все траектории скважин (бледно-синие линии)
    2. Верхний слой: слои коллекторов (зеленый/серый) поверх траекторий
    
    Аргументы:
        well_data: DataFrame с данными скважин (НЕ ИСПОЛЬЗУЕТСЯ, оставлено для совместимости)
        trajectories: словарь с траекториями скважин (обязательно)
        las_data: словарь с LAS-данными (для отображения слоев коллекторов)
        show_trajectories: не используется (оставлено для совместимости)
        show_vertical_layers: не используется (оставлено для совместимости)
        show_well_logs: показывать слои коллекторов поверх траекторий
    
    Возвращает:
        3D Figure с двухслойной визуализацией
    """
    fig = go.Figure()
    
    if not trajectories:
        return fig
    
    layers_added = 0
    wells_processed = 0
    wells_with_layers = 0
    
    # ПЕРВЫЙ ПРОХОД: Рисуем ВСЕ базовые траектории (бледно-синие или розовые для ML)
    for well_name, trajectory in trajectories.items():
        if len(trajectory) < 2:
            continue
        
        wells_processed += 1
        traj_x = trajectory[:, 0]
        traj_y = trajectory[:, 1]
        traj_z = trajectory[:, 2]
        traj_md = trajectory[:, 3]
        
        # ML скважины отображаются розовым цветом
        if well_name.startswith("ML_"):
            base_color = 'hotpink'
        else:
            base_color = 'lightblue'
        
        # Рисуем базовую траекторию
        fig.add_trace(go.Scatter3d(
            x=traj_x,
            y=traj_y,
            z=traj_z,
            mode="lines",
            name=well_name,
            line=dict(
                width=3,
                color=base_color
            ),
            hoverinfo="name+z",
            hovertemplate=f"{well_name}<br>Z: %{{z:.1f}}<br>MD: %{{customdata:.1f}}<extra></extra>",
            customdata=traj_md,
            showlegend=True
        ))
        
        # Маркеры начала и конца
        # ML скважины: розовый круг (начало) и фиолетовый ромб (конец)
        if well_name.startswith("ML_"):
            marker_colors = ['hotpink', 'purple']
            marker_symbols = ['circle', 'diamond']
        else:
            marker_colors = ['blue', 'red']
            marker_symbols = ['circle', 'diamond']
        
        fig.add_trace(go.Scatter3d(
            x=[traj_x[0], traj_x[-1]],
            y=[traj_y[0], traj_y[-1]],
            z=[traj_z[0], traj_z[-1]],
            mode="markers",
            marker=dict(
                size=6,
                color=marker_colors,
                symbol=marker_symbols
            ),
            showlegend=False,
            hoverinfo="skip"
        ))
    
    # ВТОРОЙ ПРОХОД: Добавляем слои коллекторов ПОВЕРХ траекторий
    if show_well_logs and las_data:
        for well_name, trajectory in trajectories.items():
            if len(trajectory) < 2:
                continue
            
            # Проверяем наличие LAS данных
            if well_name not in las_data:
                continue
            
            traj_x = trajectory[:, 0]
            traj_y = trajectory[:, 1]
            traj_z = trajectory[:, 2]
            traj_md = trajectory[:, 3]
            
            las = las_data[well_name]
            depth = las['depth']  # MD
            curve = las['curve']
            null_value = las.get('null_value', -999.25)
            
            # Фильтруем валидные данные
            valid_mask = (curve != null_value) & (~np.isnan(curve))
            if not np.any(valid_mask):
                continue
            
            depth_valid = depth[valid_mask]
            curve_valid = curve[valid_mask]
            
            # Проверяем диапазоны MD
            las_md_min, las_md_max = depth_valid.min(), depth_valid.max()
            traj_md_min, traj_md_max = traj_md.min(), traj_md.max()
            
            # Если диапазоны не пересекаются - пропускаем
            if las_md_max < traj_md_min or las_md_min > traj_md_max:
                continue
            
            # Интерполируем координаты по MD
            x_coords = np.interp(depth_valid, traj_md, traj_x)
            y_coords = np.interp(depth_valid, traj_md, traj_y)
            z_coords = np.interp(depth_valid, traj_md, traj_z)
            
            wells_with_layers += 1
            
            # Рисуем сегменты слоев коллекторов ПОВЕРХ базовой траектории
            i = 0
            while i < len(curve_valid):
                current_value = curve_valid[i]
                start_idx = i
                
                # Находим конец текущего сегмента
                while i < len(curve_valid) and curve_valid[i] == current_value:
                    i += 1
                end_idx = i - 1
                
                # Определяем цвет и ширину (толще базовой траектории)
                if current_value == 1:  # Коллектор
                    color = 'green'
                    width = 8  # Толще базовой линии
                    name = 'Коллектор'
                elif current_value == 0:  # Неколлектор
                    color = 'gray'
                    width = 6  # Толще базовой линии
                    name = 'Неколлектор'
                else:
                    continue
                
                # Рисуем сегмент слоя ПОВЕРХ траектории
                segment_x = x_coords[start_idx:end_idx+1]
                segment_y = y_coords[start_idx:end_idx+1]
                segment_z = z_coords[start_idx:end_idx+1]
                
                fig.add_trace(go.Scatter3d(
                    x=segment_x,
                    y=segment_y,
                    z=segment_z,
                    mode='lines',
                    line=dict(color=color, width=width),
                    showlegend=False,  # Не показываем в легенде каждый сегмент
                    hovertemplate=f"{well_name}<br>{name}<br>Z: %{{z:.1f}}<extra></extra>"
                ))
                layers_added += 1
    
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
    
    # Добавляем легенду для базовых траекторий
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='lines',
        line=dict(color='lightblue', width=3),
        name='Траектория скважины',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='lines',
        line=dict(color='hotpink', width=3),
        name='ML скважина',
        showlegend=True
    ))
    
    # Формируем заголовок
    title_text = "3D визуализация пластов-коллекторов"
    if layers_added > 0:
        title_text += f" ({wells_processed} скважин: {wells_with_layers} с данными коллекторов, {layers_added} сегментов)"
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


def create_2d_trajectory_projections(well_name: str, trajectories: Dict[str, np.ndarray],
                                      las_data: Dict[str, Dict] = None) -> Dict[str, go.Figure]:
    """
    Создает три 2D проекции траектории скважины (XY, XZ, YZ) с окраской по типу коллектора
    
    Аргументы:
        well_name: название скважины
        trajectories: словарь с траекториями скважин
        las_data: словарь с LAS-данными (для окраски коллекторов)
    
    Возвращает:
        Словарь с тремя Figure: {'XY': fig_xy, 'XZ': fig_xz, 'YZ': fig_yz}
    """
    if well_name not in trajectories:
        return {}
    
    trajectory = trajectories[well_name]
    traj_x = trajectory[:, 0]
    traj_y = trajectory[:, 1]
    traj_z = trajectory[:, 2]
    traj_md = trajectory[:, 3]
    
    # Подготавливаем данные о коллекторах, если есть
    has_collector_data = False
    x_coords, y_coords, z_coords, curve_valid = None, None, None, None
    
    if las_data and well_name in las_data:
        las = las_data[well_name]
        depth = las['depth']
        curve = las['curve']
        null_value = las.get('null_value', -999.25)
        
        valid_mask = (curve != null_value) & (~np.isnan(curve))
        if np.any(valid_mask):
            depth_valid = depth[valid_mask]
            curve_valid = curve[valid_mask]
            
            # Интерполируем координаты
            x_coords = np.interp(depth_valid, traj_md, traj_x)
            y_coords = np.interp(depth_valid, traj_md, traj_y)
            z_coords = np.interp(depth_valid, traj_md, traj_z)
            has_collector_data = True
    
    # Функция для создания сегментов с окраской
    def add_colored_segments(fig, x_data, y_data, x_label, y_label):
        """Добавляет базовую траекторию и цветные сегменты коллекторов"""
        # Базовая траектория (бледно-синяя)
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines',
            line=dict(color='lightblue', width=2),
            name='Траектория',
            hovertemplate=f'{x_label}: %{{x:.1f}}<br>{y_label}: %{{y:.1f}}<extra></extra>'
        ))
        
        # Маркеры начала и конца
        fig.add_trace(go.Scatter(
            x=[x_data[0], x_data[-1]],
            y=[y_data[0], y_data[-1]],
            mode='markers',
            marker=dict(size=10, color=['blue', 'red'], symbol=['circle', 'diamond']),
            name='Начало/Конец',
            showlegend=True,
            hoverinfo='skip'
        ))
        
        # Добавляем цветные сегменты коллекторов, если есть данные
        if has_collector_data:
            # Определяем какие координаты использовать
            if x_label == 'X' and y_label == 'Y':
                seg_x, seg_y = x_coords, y_coords
            elif x_label == 'X' and y_label == 'Z':
                seg_x, seg_y = x_coords, z_coords
            else:  # Y, Z
                seg_x, seg_y = y_coords, z_coords
            
            i = 0
            while i < len(curve_valid):
                current_value = curve_valid[i]
                start_idx = i
                
                while i < len(curve_valid) and curve_valid[i] == current_value:
                    i += 1
                end_idx = i - 1
                
                if current_value == 1:
                    color, width, name = 'green', 4, 'Коллектор'
                elif current_value == 0:
                    color, width, name = 'gray', 3, 'Неколлектор'
                else:
                    continue
                
                fig.add_trace(go.Scatter(
                    x=seg_x[start_idx:end_idx+1],
                    y=seg_y[start_idx:end_idx+1],
                    mode='lines',
                    line=dict(color=color, width=width),
                    showlegend=False,
                    hovertemplate=f'{name}<br>{x_label}: %{{x:.1f}}<br>{y_label}: %{{y:.1f}}<extra></extra>'
                ))
    
    # Создаем три проекции
    figures = {}
    
    # 1. Проекция XY (вид сверху)
    fig_xy = go.Figure()
    add_colored_segments(fig_xy, traj_x, traj_y, 'X', 'Y')
    fig_xy.update_layout(
        title=f'Проекция XY (вид сверху) - {well_name}',
        xaxis_title='X (м)',
        yaxis_title='Y (м)',
        height=500,
        hovermode='closest',
        template='plotly_white',
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
    )
    fig_xy.update_xaxes(scaleanchor="y", scaleratio=1)
    figures['XY'] = fig_xy
    
    # 2. Проекция XZ (вид сбоку)
    fig_xz = go.Figure()
    add_colored_segments(fig_xz, traj_x, traj_z, 'X', 'Z')
    fig_xz.update_layout(
        title=f'Проекция XZ (вид сбоку) - {well_name}',
        xaxis_title='X (м)',
        yaxis_title='Глубина Z (м)',
        height=500,
        hovermode='closest',
        template='plotly_white',
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
    )
    figures['XZ'] = fig_xz
    
    # 3. Проекция YZ (вид сбоку)
    fig_yz = go.Figure()
    add_colored_segments(fig_yz, traj_y, traj_z, 'Y', 'Z')
    fig_yz.update_layout(
        title=f'Проекция YZ (вид сбоку) - {well_name}',
        xaxis_title='Y (м)',
        yaxis_title='Глубина Z (м)',
        height=500,
        hovermode='closest',
        template='plotly_white',
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
    )
    figures['YZ'] = fig_yz
    
    # Добавляем легенду для коллекторов во все графики
    if has_collector_data:
        for fig in figures.values():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color='green', width=4),
                name='Коллектор (1)',
                showlegend=True
            ))
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color='gray', width=3),
                name='Неколлектор (0)',
                showlegend=True
            ))
    
    return figures


def create_ml_predictions_map(existing_wells: pd.DataFrame,
                            predicted_wells: Dict[str, Dict],
                            show_existing: bool = True) -> go.Figure:
    """
    Создает карту с существующими скважинами и ML предсказаниями

    Args:
        existing_wells: DataFrame с существующими скважинами
        predicted_wells: Словарь предсказаний от ML модели
        show_existing: Показывать ли существующие скважины

    Returns:
        Plotly Figure с картой
    """

    fig = go.Figure()

    # 1. Добавляем существующие скважины
    if show_existing and not existing_wells.empty:
        # Фильтруем скважины с координатами
        wells_with_coords = existing_wells.dropna(subset=['X', 'Y'])

        if not wells_with_coords.empty:
            # Создаем цветовую шкалу по доле коллектора
            collector_ratios = wells_with_coords['Доля_коллектора'].fillna(0)

            fig.add_trace(go.Scatter(
                x=wells_with_coords['X'],
                y=wells_with_coords['Y'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=collector_ratios,
                    colorscale='Viridis',
                    colorbar=dict(
                        title="Доля коллектора"
                    ),
                    showscale=True,
                    symbol='circle',
                    line=dict(width=1, color='black')
                ),
                name='Существующие скважины',
                text=wells_with_coords['Well'],
                hovertemplate=
                '<b>%{text}</b><br>' +
                'X: %{x:.1f}<br>' +
                'Y: %{y:.1f}<br>' +
                'Доля коллектора: %{marker.color:.3f}<br>' +
                '<extra></extra>'
            ))

    # 2. Добавляем предсказанные скважины
    if predicted_wells:
        pred_x = []
        pred_y = []
        pred_names = []
        pred_ratios = []

        for well_name, pred_data in predicted_wells.items():
            pred_x.append(pred_data['x'])
            pred_y.append(pred_data['y'])
            pred_names.append(well_name)

            # Вычисляем предсказанную долю коллектора
            predictions = np.array(pred_data['prediction'])
            collector_ratio = np.mean(predictions > 0.5)  # Порог 0.5
            pred_ratios.append(collector_ratio)

        if pred_x:
            fig.add_trace(go.Scatter(
                x=pred_x,
                y=pred_y,
                mode='markers',
                marker=dict(
                    size=12,
                    color=pred_ratios,
                    colorscale='RdBu',
                    showscale=True,
                    symbol='diamond',
                    line=dict(width=2, color='red')
                ),
                name='ML предсказания',
                text=pred_names,
                hovertemplate=
                '<b>%{text}</b> (ПРЕДСКАЗАНИЕ)<br>' +
                'X: %{x:.1f}<br>' +
                'Y: %{y:.1f}<br>' +
                'Предсказанная доля: %{marker.color:.3f}<br>' +
                '<extra></extra>'
            ))

    # 3. Настраиваем layout
    fig.update_layout(
        title="Карта скважин с ML предсказаниями",
        xaxis=dict(
            title="Координата X (м)",
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            title="Координата Y (м)",
            scaleanchor="x",
            scaleratio=1
        ),
        width=800,
        height=600,
        template='plotly_white'
    )

    return fig


def create_ml_prediction_details(prediction_data: Dict) -> go.Figure:
    """
    Создает детальный график предсказаний для одной скважины

    Args:
        prediction_data: Данные предсказания от ML модели

    Returns:
        Plotly Figure с графиком предсказаний
    """

    fig = go.Figure()

    depths = prediction_data['depth']
    predictions = prediction_data['prediction']
    well_name = prediction_data['well_name']

    # Основная кривая предсказаний
    fig.add_trace(go.Scatter(
        x=predictions,
        y=depths,
        mode='lines',
        line=dict(color='red', width=3),
        name='ML предсказание',
        hovertemplate=
        'Значение: %{x:.3f}<br>' +
        'Глубина: %{y:.1f} м<br>' +
        '<extra></extra>'
    ))

    # Добавляем порог 0.5 для определения коллектора
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="gray",
        annotation_text="Порог коллектора",
        annotation_position="bottom right"
    )

    # Раскрашиваем зоны коллектора/неколлектора
    collector_mask = predictions > 0.5

    # Зоны коллектора (зеленый)
    if np.any(collector_mask):
        fig.add_trace(go.Scatter(
            x=predictions[collector_mask],
            y=depths[collector_mask],
            mode='markers',
            marker=dict(
                color='green',
                size=6,
                symbol='circle'
            ),
            name='Коллектор',
            showlegend=False,
            hoverinfo='skip'
        ))

    # Зоны неколлектора (серый)
    if np.any(~collector_mask):
        fig.add_trace(go.Scatter(
            x=predictions[~collector_mask],
            y=depths[~collector_mask],
            mode='markers',
            marker=dict(
                color='gray',
                size=4,
                symbol='circle'
            ),
            name='Неколлектор',
            showlegend=False,
            hoverinfo='skip'
        ))

    # Настраиваем layout
    fig.update_layout(
        title=f"ML предсказания для скважины {well_name}",
        xaxis=dict(
            title="Вероятность коллектора",
            range=[-0.1, 1.1]
        ),
        yaxis=dict(
            title="Глубина (м)",
            autorange="reversed"  # Глубина увеличивается вниз
        ),
        width=600,
        height=800,
        template='plotly_white',
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )

    return fig


def create_ml_comparison_chart(existing_wells: pd.DataFrame,
                             predicted_wells: Dict[str, Dict]) -> go.Figure:
    """
    Создает сравнительную диаграмму существующих и предсказанных скважин

    Args:
        existing_wells: DataFrame с существующими скважинами
        predicted_wells: Словарь предсказаний

    Returns:
        Plotly Figure со сравнением
    """

    fig = go.Figure()

    # Собираем данные для сравнения
    existing_ratios = []
    existing_names = []

    if not existing_wells.empty:
        for _, row in existing_wells.iterrows():
            if pd.notna(row.get('Доля_коллектора')):
                existing_ratios.append(row['Доля_коллектора'])
                existing_names.append(row['Well'])

    predicted_ratios = []
    predicted_names = []

    for well_name, pred_data in predicted_wells.items():
        predictions = pred_data['prediction']
        ratio = np.mean(predictions > 0.5)
        predicted_ratios.append(ratio)
        predicted_names.append(f"{well_name} (ML)")

    # Создаем сравнительную диаграмму
    if existing_ratios:
        fig.add_trace(go.Bar(
            x=existing_names,
            y=existing_ratios,
            name='Реальные данные',
            marker_color='blue',
            opacity=0.7
        ))

    if predicted_ratios:
        fig.add_trace(go.Bar(
            x=predicted_names,
            y=predicted_ratios,
            name='ML предсказания',
            marker_color='red',
            opacity=0.7
        ))

    fig.update_layout(
        title="Сравнение доли коллектора: Реальные vs ML предсказания",
        xaxis=dict(
            title="Скважины",
            tickangle=45
        ),
        yaxis=dict(
            title="Доля коллектора",
            range=[0, 1]
        ),
        width=800,
        height=500,
        template='plotly_white',
        barmode='group'
    )

    return fig
