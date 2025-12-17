"""
Главное приложение для визуализации данных скважин
"""
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.modules.data_loader import load_all_las_files, combine_all_data, load_welltrajectories
from frontend.modules.preprocess import create_grid_from_points, filter_by_depth
from frontend.modules.visualizer import create_2d_map, create_prediction_heatmap, create_3d_trajectories, \
    create_las_cross_section, create_well_comparison, create_3d_reservoir_layers, create_2d_well_projection, \
    create_2d_trajectory_projections, create_ml_predictions_map, create_ml_prediction_details, \
    create_ml_comparison_chart

# Импорт ML предиктора
from frontend.modules.ml_predictor import ml_predictor

# Импорт DatabaseManager
try:
    from backend.database import DatabaseManager
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("⚠️ DatabaseManager не доступен. Работа только с файлами.")

# Настройки страницы
st.set_page_config(
    page_title="Визуализация данных скважин",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок
st.title("🛢️ Визуализация данных геологоразведки")
st.markdown("### Анализ свойств пласта в межскважинном пространстве")

# Инициализация состояния сессии
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'trajectories' not in st.session_state:
    st.session_state.trajectories = None
if 'well_data' not in st.session_state:
    st.session_state.well_data = None
if 'las_data' not in st.session_state:
    st.session_state.las_data = None
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = os.getenv('DATA_SOURCE', 'database')
if 'auto_load_attempted' not in st.session_state:
    st.session_state.auto_load_attempted = False

# Инициализация DatabaseManager при старте
if DB_AVAILABLE and st.session_state.db_manager is None:
    try:
        st.session_state.db_manager = DatabaseManager()
        print("✅ DatabaseManager инициализирован")
    except Exception as e:
        print(f"❌ Ошибка инициализации DatabaseManager: {e}")
        st.session_state.db_manager = None

# Автоматическая загрузка данных при старте
if (not st.session_state.data_loaded and
    not st.session_state.auto_load_attempted and
    DB_AVAILABLE and
    st.session_state.db_manager):
    
    st.session_state.auto_load_attempted = True
    
    with st.spinner("🔄 Проверка базы данных..."):
        try:
            # Проверяем есть ли данные в БД
            well_data_from_db = st.session_state.db_manager.get_all_wells()
            
            if len(well_data_from_db) > 0:
                # База НЕ пустая - загружаем из БД
                print(f"✅ Найдено {len(well_data_from_db)} скважин в БД. Загружаем из базы данных...")
                st.session_state.well_data = well_data_from_db
                st.session_state.trajectories = st.session_state.db_manager.get_all_trajectories()
                st.session_state.las_data = st.session_state.db_manager.get_all_las_data()
                st.session_state.data_loaded = True
                st.success(f"✅ Загружено из БД: {len(well_data_from_db)} скважин")
            else:
                # База пустая - загружаем из файлов и сохраняем в БД
                print("⚠️ База данных пустая. Загружаем данные из файлов...")
                
                data_folder = "src_data"
                traj_path = f"{data_folder}/INKL/траектории"
                h_path = f"{data_folder}/dot_dtv/H"
                eff_h_path = f"{data_folder}/dot_dtv/EFF_H"
                las_folder = f"{data_folder}/"
                
                # Загружаем из файлов
                st.session_state.trajectories = load_welltrajectories(traj_path)
                st.session_state.well_data = combine_all_data(h_path, eff_h_path)
                st.session_state.las_data = load_all_las_files(las_folder)
                
                # Сохраняем в БД
                print("💾 Сохранение данных в базу данных...")
                success = st.session_state.db_manager.load_data_from_files_to_db(
                    st.session_state.well_data,
                    st.session_state.trajectories,
                    st.session_state.las_data
                )
                
                if success:
                    st.session_state.data_loaded = True
                    st.success(f"✅ Загружено из файлов и сохранено в БД: {len(st.session_state.well_data)} скважин")
                else:
                    st.warning("⚠️ Данные загружены из файлов, но не удалось сохранить в БД")
                    st.session_state.data_loaded = True
                    
        except Exception as e:
            print(f"❌ Ошибка автоматической загрузки: {e}")
            import traceback
            traceback.print_exc()

# Сайдбар
with st.sidebar:
    st.header("⚙️ Настройки")

    # Переключение режимов
    view_mode = st.radio(
        "Режим просмотра:",
        ["Карта", "3D траектории", "3D пласты коллекторов", "2D проекция скважины", "2D проекции XY/XZ/YZ", "Разрезы", "Анализ", "🤖 ML предсказания", "➕ Добавить скважину"],
        index=0
    )

    st.divider()

    # Загрузка данных
    st.header("📁 Загрузка данных")

    data_folder = "src_data"
    las_folder = f"{data_folder}/"

    # Показываем источник данных
    if DB_AVAILABLE and st.session_state.db_manager:
        if st.session_state.data_loaded:
            # Определяем откуда были загружены данные
            well_count_db = len(st.session_state.db_manager.get_all_wells())
            if well_count_db > 0:
                st.info(f"🗄️ Данные из базы данных ({well_count_db} скважин)")
            else:
                st.info("📁 Данные из файлов")
    
    if st.button("🔄 Перезагрузить данные", type="secondary"):
        with st.spinner("Перезагрузка данных из файлов..."):
            try:
                # Загружаем траектории из файлов
                traj_path = f"{data_folder}/INKL/траектории"
                st.session_state.trajectories = load_welltrajectories(traj_path)

                # Загружаем данные по скважинам из файлов
                h_path = f"{data_folder}/dot_dtv/H"
                eff_h_path = f"{data_folder}/dot_dtv/EFF_H"
                st.session_state.well_data = combine_all_data(h_path, eff_h_path)

                # Загружаем LAS-файлы
                st.session_state.las_data = load_all_las_files(las_folder)
                
                # Если доступна БД - сохраняем данные (перезаписываем)
                if DB_AVAILABLE and st.session_state.db_manager:
                    with st.spinner("💾 Обновление базы данных..."):
                        success = st.session_state.db_manager.load_data_from_files_to_db(
                            st.session_state.well_data,
                            st.session_state.trajectories,
                            st.session_state.las_data
                        )
                        if success:
                            st.success("✅ Данные перезагружены из файлов и обновлены в БД!")
                        else:
                            st.warning("⚠️ Данные перезагружены из файлов, но не удалось обновить БД")
                else:
                    st.success("✅ Данные перезагружены из файлов!")

                st.session_state.data_loaded = True
            except Exception as e:
                st.error(f"Ошибка при перезагрузке данных: {e}")
                import traceback
                st.error(traceback.format_exc())

    if st.session_state.data_loaded:
        st.success(f"✅ Загружено: {len(st.session_state.trajectories)} скважин")

        # Фильтры
        st.divider()
        st.header("🔍 Фильтры")

        if st.session_state.well_data is not None:
            min_h, max_h = st.session_state.well_data["H"].min(), st.session_state.well_data["H"].max()
            h_filter = st.slider(
                "Фильтр по мощности пласта:",
                float(min_h), float(max_h),
                (float(min_h), float(max_h))
            )

# Основной контент
if not st.session_state.data_loaded:
    st.info("👈 Нажмите кнопку 'Загрузить все данные' в сайдбаре, чтобы начать")

    # Показываем структуру данных
    with st.expander("📊 О данных"):
        st.markdown("""
        **Типы данных:**
        1. **Траектории скважин** - координаты X, Y, Z и измеренная глубина
        2. **Данные H** - мощность пласта
        3. **Данные EFF_H** - эффективная мощность (только коллекторы)
        4. **LAS-файлы** - данные геофизических исследований скважин

        **Обозначения:**
        - **1** = эффективный коллектор (хорошая порода)
        - **0** = неэффективный коллектор или неколлектор
        - **-999.25** = данные отсутствуют
        """)

    # Пример данных
    if os.path.exists("data"):
        st.markdown("### Пример структуры данных")

        col1, col2 = st.columns(2)

        with col1:
            if os.path.exists("data/H.txt"):
                st.code("""# Пример H.txt
X Y Z Well H
6681.46 74209.62 1086.29 WELL_067 8.62
7508.89 75459.73 1089.02 WELL_037 7.91""")

        with col2:
            if os.path.exists("data/траектории.txt"):
                st.code("""# Пример траектории
welltrack 'WELL_034'
7131.91 75939.70 71.62 0.00
7131.88 75939.60 61.62 10.00""")
else:
    # Режим КАРТА
    if view_mode == "Карта":
        st.header("🗺️ Карта скважин")

        col1, col2 = st.columns([3, 1])

        with col1:
            # 2D карта с траекториями
            fig_map = create_2d_map(
                st.session_state.well_data,
                st.session_state.trajectories,
                show_well_names=True,
                show_trajectories=True
            )
            st.plotly_chart(fig_map, use_container_width=True)

        with col2:
            st.metric(
                "Всего скважин",
                len(st.session_state.well_data)
            )

            avg_collector = st.session_state.well_data["Доля_коллектора"].mean() * 100
            st.metric(
                "Средняя доля коллектора",
                f"{avg_collector:.1f}%"
            )

            # Таблица данных
            with st.expander("📋 Данные скважин"):
                st.dataframe(
                    st.session_state.well_data[
                        ["Well", "X", "Y", "H", "EFF_H", "Доля_коллектора"]
                    ].round(3),
                    height=300
                )

        # Прогнозы (заглушка для ML)
        st.subheader("🎯 Предсказания (ML-модуль)")

        col1, col2 = st.columns(2)

        with col1:
            # Пример предсказания
            st.info("Здесь будет результат работы ML-модели от 1 курса")

            # Создаем случайную сетку для демонстрации
            X_grid, Y_grid = create_grid_from_points(st.session_state.well_data, 50)
            Z_pred = np.random.rand(*X_grid.shape)  # Заглушка

            fig_pred = create_prediction_heatmap(X_grid, Y_grid, Z_pred)
            st.plotly_chart(fig_pred, width='stretch')

        with col2:
            # Настройки предсказания
            st.markdown("#### Настройки предсказания")
            prediction_type = st.selectbox(
                "Тип предсказания:",
                ["Доля коллектора", "Наличие коллектора", "Мощность пласта"]
            )

            interpolation_method = st.selectbox(
                "Метод интерполяции:",
                ["Kriging", "IDW", "RBF"]
            )

            if st.button("Запустить предсказание", type="secondary"):
                st.success(f"Запущено предсказание: {prediction_type}")
                # Здесь будет вызов ML-модели

    # Режим 3D ТРАЕКТОРИИ
    elif view_mode == "3D траектории":
        st.header("🔄 3D траектории скважин")

        col1, col2 = st.columns([4, 1])

        with col1:
            fig_3d = create_3d_trajectories(st.session_state.trajectories)
            st.plotly_chart(fig_3d, width='stretch', height=700)

        with col2:
            st.markdown("#### Выбор скважин")

            well_list = list(st.session_state.trajectories.keys())
            selected_wells = st.multiselect(
                "Показать траектории:",
                well_list,
                default=well_list[:min(5, len(well_list))]
            )

            st.markdown("#### Информация")
            for well in selected_wells[:3]:  # Показываем только первые 3
                traj = st.session_state.trajectories[well]
                if len(traj) > 0:
                    st.text(f"{well}:")
                    st.text(f"Длина: {traj[-1, 3]:.1f} м")
                    st.text(f"Глубина: {traj[-1, 2]:.1f} м")
                    st.divider()

    # Режим 3D ПЛАСТЫ КОЛЛЕКТОРОВ
    elif view_mode == "3D пласты коллекторов":
        st.header("🏔️ 3D визуализация пластов-коллекторов")
        
        st.info("💡 Желтая поверхность = коллектор, серая = неколлектор. Зеленые линии на стволах = коллектор, серые = неколлектор")
        
        col1, col2 = st.columns([4, 1])
        
        with col2:
            st.markdown("#### Настройки отображения")
            
            # Показать маркеры коллекторов
            show_logs = st.checkbox("Маркеры коллекторов", value=True)
            
            # Показать траектории
            show_trajectories = st.checkbox("Показать траектории", value=True)
            
            # Показать вертикальные линии скважин
            show_vertical = st.checkbox("Вертикальные линии", value=True)
            
            st.markdown("#### Статистика")
            st.metric("Всего скважин", len(st.session_state.well_data))
            
            avg_h = st.session_state.well_data["H"].mean()
            st.metric("Средняя мощность H", f"{avg_h:.2f} м")
            
            avg_eff_h = st.session_state.well_data["EFF_H"].mean()
            st.metric("Средняя эфф. мощность", f"{avg_eff_h:.2f} м")
            
            avg_collector = st.session_state.well_data["Доля_коллектора"].mean() * 100
            st.metric("Средняя доля коллектора", f"{avg_collector:.1f}%")
            
            if st.session_state.las_data:
                st.metric("LAS файлов загружено", len(st.session_state.las_data))
            
            st.markdown("#### Легенда")
            st.markdown("""
            **Поверхности:**
            - 🟡 **Желтая** - кровля пласта (коллектор)
            - 🟠 **Оранжевая** - высокая доля коллектора
            - ⚪ **Серая** - подошва пласта
            
            **Маркеры на стволах:**
            - 🟢 **Зеленая линия** - коллектор (1)
            - ⚫ **Серая линия** - неколлектор (0)
            - ⬛ **Черная линия** - ствол скважины
            """)
        
        with col1:
            # Создаем 3D визуализацию пластов с каротажными диаграммами
            # ВАЖНО: траектории нужны всегда для корректного маппинга MD -> Z
            fig_3d_layers = create_3d_reservoir_layers(
                st.session_state.well_data,
                st.session_state.trajectories,  # Всегда передаем траектории
                st.session_state.las_data,
                show_trajectories=show_trajectories,
                show_vertical_layers=show_vertical,
                show_well_logs=show_logs
            )
            st.plotly_chart(fig_3d_layers, use_container_width=True)
    
    # Режим 2D ПРОЕКЦИЯ СКВАЖИНЫ
    elif view_mode == "2D проекция скважины":
        st.header("📊 2D проекция скважины с слоями")
        
        st.info("💡 Выберите скважину для просмотра её 2D проекции с отображением слоев коллекторов и неколлекторов")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("#### Выбор скважины")
            
            # Список скважин
            well_list = sorted(st.session_state.well_data["Well"].tolist())
            selected_well = st.selectbox(
                "Скважина:",
                well_list,
                index=0
            )
            
            # Информация о выбранной скважине
            if selected_well:
                well_info = st.session_state.well_data[
                    st.session_state.well_data["Well"] == selected_well
                ].iloc[0]
                
                st.markdown("#### Информация")
                st.metric("Координата X", f"{well_info['X']:.2f} м")
                st.metric("Координата Y", f"{well_info['Y']:.2f} м")
                st.metric("Кровля Z", f"{well_info['Z']:.2f} м")
                st.metric("Мощность H", f"{well_info['H']:.2f} м")
                st.metric("Эфф. мощность", f"{well_info['EFF_H']:.2f} м")
                st.metric("Доля коллектора", f"{well_info['Доля_коллектора']*100:.1f}%")
                
                st.markdown("#### Легенда")
                st.markdown("""
                - 🟢 **Зеленый** - коллектор (1)
                - ⚪ **Серый** - неколлектор (0)
                - 🔴 **Красная линия** - каротажная кривая
                - ⬛ **Черная линия** - ствол скважины
                - 🔵 **Синий треугольник** - кровля пласта
                - 🔴 **Красный треугольник** - подошва пласта
                """)
        
        with col1:
            if selected_well:
                # Создаем 2D проекцию с траекториями для точного преобразования MD -> Z
                fig_2d_proj = create_2d_well_projection(
                    st.session_state.well_data,
                    st.session_state.las_data,
                    selected_well,
                    st.session_state.trajectories
                )
                st.plotly_chart(fig_2d_proj, use_container_width=True)
            else:
                st.warning("Выберите скважину для отображения")
    
    # Режим 2D ПРОЕКЦИИ XY/XZ/YZ
    elif view_mode == "2D проекции XY/XZ/YZ":
        st.header("📐 2D проекции траектории скважины")
        
        st.info("💡 Три проекции траектории: XY (вид сверху), XZ и YZ (виды сбоку) с окраской коллекторов")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("#### Выбор скважины")
            
            # Список скважин
            well_list = sorted(list(st.session_state.trajectories.keys()))
            selected_well = st.selectbox(
                "Скважина:",
                well_list,
                index=0,
                key="projections_well_select"
            )
            
            # Информация о выбранной скважине
            if selected_well and selected_well in st.session_state.well_data["Well"].values:
                well_info = st.session_state.well_data[
                    st.session_state.well_data["Well"] == selected_well
                ].iloc[0]
                
                st.markdown("#### Информация")
                st.metric("Координата X", f"{well_info['X']:.2f} м")
                st.metric("Координата Y", f"{well_info['Y']:.2f} м")
                st.metric("Кровля Z", f"{well_info['Z']:.2f} м")
                st.metric("Мощность H", f"{well_info['H']:.2f} м")
                st.metric("Эфф. мощность", f"{well_info['EFF_H']:.2f} м")
                st.metric("Доля коллектора", f"{well_info['Доля_коллектора']*100:.1f}%")
            
            # Информация о траектории
            if selected_well in st.session_state.trajectories:
                traj = st.session_state.trajectories[selected_well]
                st.markdown("#### Траектория")
                st.metric("Длина MD", f"{traj[-1, 3]:.1f} м")
                st.metric("Глубина Z", f"{traj[-1, 2]:.1f} м")
                st.metric("Точек", len(traj))
            
            st.markdown("#### Легенда")
            st.markdown("""
            **Цвета:**
            - 🔵 **Бледно-синий** - траектория скважины
            - 🟢 **Зеленый** - коллектор (1)
            - ⚪ **Серый** - неколлектор (0)
            
            **Маркеры:**
            - 🔵 **Синий круг** - начало
            - 🔴 **Красный ромб** - конец
            
            **Проекции:**
            - **XY** - вид сверху (горизонтальная плоскость)
            - **XZ** - вид сбоку (вертикальная плоскость)
            - **YZ** - вид сбоку (вертикальная плоскость)
            """)
        
        with col1:
            if selected_well:
                # Создаем три проекции
                projections = create_2d_trajectory_projections(
                    selected_well,
                    st.session_state.trajectories,
                    st.session_state.las_data
                )
                
                if projections:
                    # Отображаем все три проекции одновременно
                    st.markdown("### 📍 Проекция XY (вид сверху)")
                    st.plotly_chart(projections['XY'], use_container_width=True)
                    st.caption("Вид сверху: показывает горизонтальное отклонение скважины")
                    
                    st.divider()
                    
                    st.markdown("### 📏 Проекция XZ (вид сбоку)")
                    st.plotly_chart(projections['XZ'], use_container_width=True)
                    st.caption("Вид сбоку (X-Z): показывает отклонение по оси X и глубину")
                    
                    st.divider()
                    
                    st.markdown("### 📐 Проекция YZ (вид сбоку)")
                    st.plotly_chart(projections['YZ'], use_container_width=True)
                    st.caption("Вид сбоку (Y-Z): показывает отклонение по оси Y и глубину")
                else:
                    st.warning(f"Не удалось создать проекции для {selected_well}")
            else:
                st.warning("Выберите скважину для отображения")
    
    # Режим РАЗРЕЗЫ
    elif view_mode == "Разрезы":
        st.header("📐 Геофизические разрезы")

        if st.session_state.las_data:
            col1, col2 = st.columns([1, 3])

            with col1:
                # Выбор скважины
                las_wells = list(st.session_state.las_data.keys())
                selected_well = st.selectbox(
                    "Выберите скважину:",
                    las_wells
                )

                # Настройки глубины
                st.markdown("#### Настройки глубины")

                if selected_well in st.session_state.las_data:
                    depth_data = st.session_state.las_data[selected_well]['depth']
                    min_depth = float(depth_data.min())
                    max_depth = float(depth_data.max())

                    depth_range = st.slider(
                        "Диапазон глубины:",
                        min_depth, max_depth,
                        (min_depth, max_depth)
                    )

            with col2:
                # Отображение разреза
                if selected_well in st.session_state.las_data:
                    las_data = st.session_state.las_data[selected_well]

                    # Фильтруем по глубине
                    filtered_data = filter_by_depth(
                        las_data,
                        depth_range[0],
                        depth_range[1]
                    )

                    fig_cross = create_las_cross_section(filtered_data, selected_well)
                    st.plotly_chart(fig_cross, width='stretch')

                    # Статистика по разрезу
                    if len(filtered_data['curve']) > 0:
                        collector_count = np.sum(filtered_data['curve'] == 1)
                        non_collector_count = np.sum(filtered_data['curve'] == 0)
                        total = len(filtered_data['curve'])

                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        with col_stat1:
                            st.metric("Коллектор", f"{collector_count}")
                        with col_stat2:
                            st.metric("Не коллектор", f"{non_collector_count}")
                        with col_stat3:
                            if total > 0:
                                percent = (collector_count / total) * 100
                                st.metric("Доля", f"{percent:.1f}%")
        else:
            st.warning("LAS-файлы не загружены или не найдены")

    # Режим АНАЛИЗ
    elif view_mode == "Анализ":
        st.header("📊 Анализ данных")

        tab1, tab2, tab3 = st.tabs(["Сравнение", "Статистика", "Экспорт"])

        with tab1:
            # Сравнительная диаграмма
            fig_compare = create_well_comparison(st.session_state.well_data)
            st.plotly_chart(fig_compare, width='stretch')

        with tab2:
            # Статистика
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### Описательная статистика")
                stats_df = st.session_state.well_data[
                    ["H", "EFF_H", "Доля_коллектора"]
                ].describe()
                st.dataframe(stats_df)

            with col2:
                st.markdown("##### Корреляция")
                corr_matrix = st.session_state.well_data[
                    ["H", "EFF_H", "Доля_коллектора"]
                ].corr()
                st.dataframe(corr_matrix)

                # Гистограмма
                fig_hist = px.histogram(
                    st.session_state.well_data,
                    x="Доля_коллектора",
                    nbins=20,
                    title="Распределение доли коллектора"
                )
                st.plotly_chart(fig_hist, width='stretch')

        with tab3:
            # Экспорт данных
            st.markdown("##### Экспорт данных")

            export_format = st.radio(
                "Формат экспорта:",
                ["CSV", "Excel", "JSON"]
            )

            if st.button("Экспортировать данные", type="primary"):
                if export_format == "CSV":
                    csv = st.session_state.well_data.to_csv(index=False)
                    st.download_button(
                        label="Скачать CSV",
                        data=csv,
                        file_name="well_data.csv",
                        mime="text/csv"
                    )
                elif export_format == "Excel":
                    excel_buffer = st.session_state.well_data.to_excel(index=False)
                    st.download_button(
                        label="Скачать Excel",
                        data=excel_buffer,
                        file_name="well_data.xlsx",
                        mime="application/vnd.ms-excel"
                    )

    # Режим ML ПРЕДСКАЗАНИЙ
    elif view_mode == "🤖 ML предсказания":
        st.header("🤖 Демонстрация ML предсказаний")

        st.info("💡 Этот режим демонстрирует работу ML модели. Для реального использования добавьте скважину через режим '➕ Добавить скважину' с включенными ML предсказаниями")

        # Настройки предсказаний
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### Настройки предсказаний")

            num_predictions = st.slider(
                "Количество предсказываемых скважин",
                min_value=1,
                max_value=10,
                value=3,
                help="Сколько новых скважин сгенерировать для демонстрации"
            )

            depth_min = st.number_input(
                "Минимальная глубина (м)",
                value=-200.0,
                step=10.0,
                format="%.1f"
            )

            depth_max = st.number_input(
                "Максимальная глубина (м)",
                value=0.0,
                step=10.0,
                format="%.1f"
            )

            num_points = st.slider(
                "Точек предсказания на скважину",
                min_value=20,
                max_value=100,
                value=50
            )

        with col2:
            st.markdown("### Статистика модели")
            if st.button("🔄 Запустить предсказания", type="primary"):
                st.session_state.ml_predictions = None  # Сброс предыдущих

                # Генерируем тестовые скважины для предсказания
                test_wells = []
                existing_coords = []

                # Собираем координаты существующих скважин
                if not st.session_state.well_data.empty:
                    for _, row in st.session_state.well_data.iterrows():
                        if pd.notna(row.get('X')) and pd.notna(row.get('Y')):
                            existing_coords.append((row['X'], row['Y']))

                # Генерируем новые координаты рядом с существующими
                for i in range(num_predictions):
                    if existing_coords:
                        # Выбираем случайную существующую скважину как центр
                        center_x, center_y = existing_coords[np.random.randint(len(existing_coords))]

                        # Добавляем случайное смещение
                        offset_x = np.random.normal(0, 500)  # ±500м
                        offset_y = np.random.normal(0, 500)

                        new_x = center_x + offset_x
                        new_y = center_y + offset_y
                    else:
                        # Если нет существующих, генерируем случайные
                        new_x = np.random.uniform(6000, 15000)
                        new_y = np.random.uniform(70000, 78000)

                    test_wells.append({
                        'name': f'WELL_ML_{i+1:02d}',
                        'x': new_x,
                        'y': new_y
                    })

                # Запускаем предсказания
                with st.spinner("🤖 ML модель делает предсказания..."):
                    predictions = ml_predictor.predict_multiple_wells(
                        wells_data=test_wells,
                        depth_range=(depth_min, depth_max),
                        num_points=num_points
                    )

                st.session_state.ml_predictions = predictions
                st.success(f"✅ Сгенерировано {len(predictions)} предсказаний!")

        # Отображение результатов
        if hasattr(st.session_state, 'ml_predictions') and st.session_state.ml_predictions:

            predictions = st.session_state.ml_predictions

            # Статистика предсказаний
            stats = ml_predictor.get_prediction_stats(predictions)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Предсказанных скважин", stats.get('num_wells', 0))
            with col2:
                st.metric("Средняя доля коллектора", f"{stats.get('collector_ratio_mean', 0):.3f}")
            with col3:
                st.metric("Всего точек", stats.get('total_points', 0))
            with col4:
                st.metric("Стандартное отклонение", f"{stats.get('collector_ratio_std', 0):.3f}")

            # Вкладки с визуализациями
            tab1, tab2, tab3 = st.tabs(["🗺️ Карта", "📊 Детали", "📈 Сравнение"])

            with tab1:
                st.markdown("### Карта с предсказаниями")
                fig_map = create_ml_predictions_map(
                    existing_wells=st.session_state.well_data,
                    predicted_wells=predictions,
                    show_existing=True
                )
                st.plotly_chart(fig_map, use_container_width=True)

            with tab2:
                st.markdown("### Детальные предсказания")

                # Выбор скважины для детального просмотра
                well_names = list(predictions.keys())
                selected_well = st.selectbox(
                    "Выберите скважину для детального анализа:",
                    well_names,
                    key="ml_detail_well"
                )

                if selected_well and selected_well in predictions:
                    fig_detail = create_ml_prediction_details(predictions[selected_well])
                    st.plotly_chart(fig_detail, use_container_width=True)

                    # Показываем числовые данные
                    pred_data = predictions[selected_well]
                    collector_ratio = np.mean(pred_data['prediction'] > 0.5)

                    st.markdown(f"**Предсказанная доля коллектора: {collector_ratio:.3f}**")
                    st.markdown(f"**Координаты: X={pred_data['x']:.1f}, Y={pred_data['y']:.1f}**")

            with tab3:
                st.markdown("### Сравнение с реальными данными")
                fig_comparison = create_ml_comparison_chart(
                    existing_wells=st.session_state.well_data,
                    predicted_wells=predictions
                )
                st.plotly_chart(fig_comparison, use_container_width=True)

                st.info("🔍 Синие столбцы - реальные данные существующих скважин, Красные - ML предсказания для новых скважин")
        else:
            st.info("👆 Нажмите 'Запустить предсказания' для демонстрации работы ML модели")

    # Режим ДОБАВИТЬ СКВАЖИНУ
    elif view_mode == "➕ Добавить скважину":
        st.header("➕ Добавить новую скважину с ML предсказаниями")

        st.info("💡 Создайте вертикальную скважину и автоматически получите ML предсказания коллекторских зон по всей глубине")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Параметры скважины")
            
            # Форма ввода данных
            with st.form("add_well_form"):
                well_name = st.text_input(
                    "Название скважины",
                    value="ML_WELL_NEW",
                    help="Например: ML_WELL_100, ML_WELL_TEST (префикс ML_ для скважин с ML предсказаниями)"
                )
                
                col_x, col_y = st.columns(2)
                with col_x:
                    x_coord = st.number_input(
                        "Координата X (м)",
                        value=7000.0,
                        step=10.0,
                        format="%.2f"
                    )
                with col_y:
                    y_coord = st.number_input(
                        "Координата Y (м)",
                        value=74000.0,
                        step=10.0,
                        format="%.2f"
                    )
                
                col_z1, col_z2 = st.columns(2)
                with col_z1:
                    z1_coord = st.number_input(
                        "Z1 - Начало (м)",
                        value=0.0,
                        step=1.0,
                        format="%.2f",
                        help="Глубина начала скважины (обычно 0 или положительное значение)"
                    )
                with col_z2:
                    z2_coord = st.number_input(
                        "Z2 - Конец (м)",
                        value=-100.0,
                        step=1.0,
                        format="%.2f",
                        help="Глубина конца скважины (отрицательное значение)"
                    )

                # Опция ML предсказаний
                use_ml = st.checkbox(
                    "🤖 Сгенерировать ML предсказания коллекторских зон",
                    value=True,
                    help="Автоматически предсказать коллекторские свойства по глубине скважины"
                )

                # Параметры ML (показываются только если чекбокс выбран)
                if use_ml:
                    st.markdown("**Параметры ML предсказаний:**")
                    ml_col1, ml_col2 = st.columns(2)
                    with ml_col1:
                        ml_depth_step = st.slider(
                            "Шаг глубины (м)",
                            min_value=1.0,
                            max_value=10.0,
                            value=5.0,
                            step=1.0,
                            help="Расстояние между точками предсказания"
                        )
                    with ml_col2:
                        ml_confidence = st.slider(
                            "Уровень доверия",
                            min_value=0.1,
                            max_value=1.0,
                            value=0.7,
                            step=0.1,
                            help="Минимальная вероятность для определения коллектора"
                        )

                # Кнопка добавления
                submitted = st.form_submit_button("➕ Добавить скважину", type="primary")
                
                if submitted:
                    # Валидация
                    if z2_coord >= z1_coord:
                        st.error("❌ Z2 должно быть меньше Z1 (скважина идет вниз)")
                    elif well_name in st.session_state.trajectories:
                        st.error(f"❌ Скважина {well_name} уже существует")
                    else:
                        # Создаем траекторию вертикальной скважины
                        depth_range = abs(z2_coord - z1_coord)
                        num_points = max(int(depth_range / 10) + 1, 2)
                        
                        z_points = np.linspace(z1_coord, z2_coord, num_points)
                        md_points = np.linspace(0, depth_range, num_points)
                        
                        # Создаем массив траектории [X, Y, Z, MD]
                        new_trajectory = np.column_stack([
                            np.full(num_points, x_coord),
                            np.full(num_points, y_coord),
                            z_points,
                            md_points
                        ])
                        
                        # Добавляем в траектории (в памяти)
                        st.session_state.trajectories[well_name] = new_trajectory
                        
                        # Сохраняем в базу данных
                        if DB_AVAILABLE and st.session_state.db_manager:
                            try:
                                # Сохраняем скважину
                                well_id = st.session_state.db_manager.save_well(
                                    well_name,
                                    x_coord,
                                    y_coord,
                                    z1_coord,
                                    None,  # H
                                    None   # EFF_H
                                )
                                
                                if well_id:
                                    # Сохраняем траекторию
                                    success = st.session_state.db_manager.save_trajectory(
                                        well_name,
                                        new_trajectory
                                    )
                                    
                                    if success:
                                        # Если включены ML предсказания, генерируем и сохраняем их
                                        if use_ml:
                                            try:
                                                st.info("🤖 Генерация ML предсказаний...")

                                                # ВАЖНО: Добавляем префикс ML_ если его нет
                                                if not well_name.startswith("ML_"):
                                                    well_name = f"ML_{well_name}"
                                                    # Обновляем траекторию с новым именем
                                                    st.session_state.trajectories[well_name] = st.session_state.trajectories.pop(list(st.session_state.trajectories.keys())[-1])

                                                # ВАЖНО: Используем MD (measured depth), а не Z координаты
                                                # MD всегда положительный (от 0 до длины скважины)
                                                depth_range_meters = abs(z2_coord - z1_coord)
                                                num_predictions = max(int(depth_range_meters / ml_depth_step) + 1, 10)

                                                # Генерируем предсказания с MD диапазоном (0 до длины скважины)
                                                predictions = ml_predictor.predict_collector_zones(
                                                    well_name=well_name,
                                                    x=x_coord,
                                                    y=y_coord,
                                                    depth_range=(0, depth_range_meters),  # MD от 0 до длины
                                                    num_points=num_predictions
                                                )

                                                # ВАЖНО: Убеждаемся что предсказания строго 0 или 1
                                                # Преобразуем в бинарные значения (на случай если есть float)
                                                binary_predictions = np.where(predictions['prediction'] >= 0.5, 1, 0).astype(int)

                                                # ВАЖНО: Убеждаемся что depth (MD) положительный
                                                depth_md = np.abs(predictions['depth'])  # MD всегда положительный
                                                
                                                # Отладка: проверяем что сохраняем
                                                print(f"🔍 DEBUG: Сохранение LAS данных для {well_name}")
                                                print(f"  depth_md диапазон: {depth_md.min():.2f} - {depth_md.max():.2f}")
                                                print(f"  binary_predictions: {np.unique(binary_predictions)}")
                                                
                                                # Сохраняем предсказания как LAS данные
                                                las_success = st.session_state.db_manager.save_las_data(
                                                    well_name=well_name,
                                                    depth=depth_md,
                                                    curve=binary_predictions
                                                )

                                                if las_success:
                                                    # Обновляем session state с новыми LAS данными
                                                    st.session_state.las_data[well_name] = {
                                                        'well_name': well_name,
                                                        'depth': depth_md,  # Используем положительный MD
                                                        'curve': binary_predictions,
                                                        'null_value': -999.25
                                                    }
                                                    
                                                    # Проверяем что данные добавлены
                                                    collector_count = np.sum(binary_predictions == 1)
                                                    non_collector_count = np.sum(binary_predictions == 0)
                                                    
                                                    st.success("✅ ML предсказания сгенерированы и сохранены!")
                                                    st.info(f"💡 Добавлено {len(binary_predictions)} точек: {collector_count} коллектор, {non_collector_count} неколлектор")
                                                    st.info("📊 Данные доступны для визуализации в режимах '3D пласты коллекторов', '2D проекция скважины' и 'Разрезы'")
                                                else:
                                                    st.warning("⚠️ Скважина добавлена, но ML предсказания не сохранены")

                                            except Exception as e:
                                                st.warning(f"⚠️ Ошибка генерации ML предсказаний: {e}")

                                        st.success(f"✅ Скважина {well_name} добавлена и сохранена в БД!")
                                        st.info("💡 Перейдите в режим '3D пласты коллекторов' для просмотра")
                                    else:
                                        st.warning(f"⚠️ Скважина {well_name} добавлена, но траектория не сохранена в БД")
                                else:
                                    st.warning(f"⚠️ Скважина {well_name} добавлена, но не сохранена в БД")
                            except Exception as e:
                                st.warning(f"⚠️ Скважина {well_name} добавлена, но ошибка сохранения в БД: {e}")
                        else:
                            st.success(f"✅ Скважина {well_name} успешно добавлена!")
                            st.info("💡 Перейдите в режим '3D пласты коллекторов' для просмотра")
                        
                        st.balloons()

                        # Показываем информацию
                        st.markdown("#### Информация о добавленной скважине:")
                        info_col1, info_col2, info_col3 = st.columns(3)
                        with info_col1:
                            st.metric("Координаты", f"X: {x_coord:.1f}, Y: {y_coord:.1f}")
                        with info_col2:
                            st.metric("Глубина", f"{depth_range:.1f} м")
                        with info_col3:
                            st.metric("Точек", num_points)

                        # Если были сгенерированы ML предсказания, показываем их
                        if use_ml and 'predictions' in locals():
                            st.markdown("#### 🤖 ML Предсказания:")
                            pred_collector_ratio = np.mean(predictions['prediction'] > ml_confidence)

                            ml_col1, ml_col2, ml_col3 = st.columns(3)
                            with ml_col1:
                                st.metric("Доля коллектора", f"{pred_collector_ratio:.1%}")
                            with ml_col2:
                                st.metric("Точек анализа", len(predictions['prediction']))
                            with ml_col3:
                                st.metric("Шаг глубины", f"{ml_depth_step:.1f} м")

                            # Показываем график предсказаний
                            fig_pred = create_ml_prediction_details(predictions)
                            st.plotly_chart(fig_pred, use_container_width=True)

                            st.info(f"💡 ML модель предсказала **{pred_collector_ratio:.1%}** коллекторских зон в скважине {well_name}")

        
        with col2:
            st.markdown("### Справка")
            st.markdown("""
            **Координаты:**
            - **X, Y** - горизонтальное положение устья скважины
            - **Z1** - глубина начала (обычно 0)
            - **Z2** - глубина конца (отрицательное значение)
            
            **Примеры:**
            - Скважина 100м: Z1=0, Z2=-100
            - Скважина 50м: Z1=10, Z2=-40
            
            **Особенности:**
            - Скважина вертикальная (X и Y постоянны)
            - MD рассчитывается автоматически
            - Точки с шагом ~10 метров
            
            **После добавления:**
            - Появится на всех графиках
            - Доступна в "3D траектории"
            - Доступна в "2D проекции"
            """)
            
            st.markdown("### Текущие скважины")
            if st.session_state.trajectories:
                st.metric("Всего скважин", len(st.session_state.trajectories))
                well_names = list(st.session_state.trajectories.keys())[-5:]
                st.markdown("**Последние:**")
                for wn in well_names:
                    st.text(f"• {wn}")

# Футер
st.divider()
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Хакатон: Технологии разработки ПО**")
with col2:
    st.markdown("**Команда 2 курса**")
with col3:
    st.markdown(f"Данные загружены: {st.session_state.data_loaded}")
