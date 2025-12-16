"""
Тестовый скрипт для проверки маппинга MD -> Z
Проверяет корректность преобразования Measured Depth в Z координаты
"""
import numpy as np
import sys
sys.path.append('.')

from frontend.modules.data_loader import load_welltrajectories, load_all_las_files, combine_all_data

def test_md_mapping():
    """Тестирует маппинг MD -> Z для всех скважин"""
    
    print("=" * 80)
    print("ТЕСТ МАППИНГА MD -> Z")
    print("=" * 80)
    
    # Загружаем данные
    print("\n1. Загрузка данных...")
    trajectories = load_welltrajectories('src_data/INKL/траектории')
    las_data = load_all_las_files('src_data')
    well_data = combine_all_data('src_data/dot_dtv/H', 'src_data/dot_dtv/EFF_H')
    
    print(f"   Загружено траекторий: {len(trajectories)}")
    print(f"   Загружено LAS файлов: {len(las_data)}")
    print(f"   Загружено скважин в well_data: {len(well_data)}")
    
    # Проверяем несколько скважин
    test_wells = ['WELL_002', 'WELL_034', 'WELL_001']
    
    print("\n2. Анализ диапазонов MD и Z:")
    print("-" * 80)
    
    issues_found = []
    
    for well_name in test_wells:
        if well_name not in trajectories or well_name not in las_data:
            print(f"\n❌ {well_name}: Нет данных")
            continue
        
        print(f"\n📊 {well_name}:")
        
        # Данные траектории
        traj = trajectories[well_name]
        traj_md = traj[:, 3]
        traj_z = traj[:, 2]
        traj_x = traj[:, 0]
        traj_y = traj[:, 1]
        
        print(f"   Траектория:")
        print(f"      MD диапазон: [{traj_md.min():.2f}, {traj_md.max():.2f}] м")
        print(f"      Z диапазон:  [{traj_z.min():.2f}, {traj_z.max():.2f}] м")
        print(f"      X диапазон:  [{traj_x.min():.2f}, {traj_x.max():.2f}] м")
        print(f"      Y диапазон:  [{traj_y.min():.2f}, {traj_y.max():.2f}] м")
        print(f"      Точек: {len(traj)}")
        
        # Проверка на вертикальность
        x_var = np.std(traj_x)
        y_var = np.std(traj_y)
        is_vertical = x_var < 1.0 and y_var < 1.0
        print(f"      Тип: {'Вертикальная' if is_vertical else 'Наклонная'} (σ_x={x_var:.2f}, σ_y={y_var:.2f})")
        
        # Данные LAS
        las = las_data[well_name]
        las_depth = las['depth']
        las_curve = las['curve']
        null_value = las.get('null_value', -999.25)
        
        # Фильтруем валидные данные
        valid_mask = (las_curve != null_value) & (~np.isnan(las_curve))
        las_depth_valid = las_depth[valid_mask]
        las_curve_valid = las_curve[valid_mask]
        
        print(f"   LAS данные:")
        print(f"      MD диапазон: [{las_depth_valid.min():.2f}, {las_depth_valid.max():.2f}] м")
        print(f"      Валидных точек: {len(las_depth_valid)} из {len(las_depth)}")
        print(f"      Коллектор (1): {np.sum(las_curve_valid == 1)} точек")
        print(f"      Неколлектор (0): {np.sum(las_curve_valid == 0)} точек")
        
        # Проверка соответствия диапазонов
        las_md_min, las_md_max = las_depth_valid.min(), las_depth_valid.max()
        traj_md_min, traj_md_max = traj_md.min(), traj_md.max()
        
        md_overlap = not (las_md_min < traj_md_min - 1.0 or las_md_max > traj_md_max + 1.0)
        
        print(f"   Проверка соответствия:")
        if md_overlap:
            print(f"      ✅ Диапазоны MD совпадают")
        else:
            print(f"      ⚠️  Диапазоны MD НЕ совпадают!")
            print(f"         LAS MD выходит за пределы траектории")
            print(f"         Смещение начала: {las_md_min - traj_md_min:.2f} м")
            print(f"         Смещение конца: {las_md_max - traj_md_max:.2f} м")
            issues_found.append(f"{well_name}: MD диапазоны не совпадают")
        
        # Тестируем интерполяцию
        z_interpolated = np.interp(las_depth_valid, traj_md, traj_z)
        z_range = z_interpolated.max() - z_interpolated.min()
        
        print(f"   Результат интерполяции MD -> Z:")
        print(f"      Z диапазон: [{z_interpolated.min():.2f}, {z_interpolated.max():.2f}] м")
        print(f"      Размах Z: {z_range:.2f} м")
        
        # Проверка на корректность
        if well_name in well_data['Well'].values:
            well_row = well_data[well_data['Well'] == well_name].iloc[0]
            expected_h = well_row['H']
            z_top = well_row['Z']
            
            print(f"   Сравнение с well_data:")
            print(f"      Ожидаемая мощность H: {expected_h:.2f} м")
            print(f"      Кровля Z: {z_top:.2f} м")
            print(f"      Расчетная мощность: {z_range:.2f} м")
            
            if abs(z_range - expected_h) > expected_h * 0.5:
                print(f"      ⚠️  БОЛЬШОЕ расхождение в мощности!")
                issues_found.append(f"{well_name}: Расхождение мощности {abs(z_range - expected_h):.2f} м")
            elif abs(z_range - expected_h) > expected_h * 0.2:
                print(f"      ⚠️  Заметное расхождение в мощности")
            else:
                print(f"      ✅ Мощность соответствует ожиданиям")
    
    # Итоги
    print("\n" + "=" * 80)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 80)
    
    if issues_found:
        print(f"\n⚠️  Обнаружено проблем: {len(issues_found)}")
        for issue in issues_found:
            print(f"   - {issue}")
    else:
        print("\n✅ Все проверки пройдены успешно!")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    test_md_mapping()