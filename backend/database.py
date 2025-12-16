"""
Модуль для работы с PostgreSQL базой данных
"""
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from psycopg2.pool import SimpleConnectionPool


class DatabaseManager:
    """Менеджер для работы с базой данных PostgreSQL"""
    
    def __init__(self, database_url: str = None):
        """
        Инициализация менеджера БД
        
        Args:
            database_url: URL подключения к БД (по умолчанию из переменной окружения)
        """
        self.database_url = database_url or os.getenv(
            'DATABASE_URL',
            'postgresql://pygeo_user:pygeo_password@localhost:5432/pygeo_db'
        )
        self.pool = None
        self._init_pool()
    
    def _init_pool(self):
        """Инициализация пула соединений"""
        try:
            self.pool = SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=self.database_url
            )
        except Exception as e:
            print(f"Ошибка подключения к БД: {e}")
            self.pool = None
    
    def get_connection(self):
        """Получить соединение из пула"""
        if self.pool:
            return self.pool.getconn()
        return None
    
    def put_connection(self, conn):
        """Вернуть соединение в пул"""
        if self.pool and conn:
            self.pool.putconn(conn)
    
    def close_all(self):
        """Закрыть все соединения"""
        if self.pool:
            self.pool.closeall()
    
    # ========== WELLS ==========
    
    def save_well(self, name: str, x: float, y: float, z: float,
                  h: float = None, eff_h: float = None) -> int:
        """
        Сохранить скважину в БД
        
        Returns:
            ID созданной скважины
        """
        conn = self.get_connection()
        if not conn:
            return None
        
        try:
            with conn.cursor() as cur:
                # Конвертируем numpy типы в Python типы
                x = float(x) if x is not None else None
                y = float(y) if y is not None else None
                z = float(z) if z is not None else None
                h = float(h) if h is not None and not pd.isna(h) else None
                eff_h = float(eff_h) if eff_h is not None and not pd.isna(eff_h) else None
                
                collector_ratio = (eff_h / h) if (h and eff_h and h > 0) else None
                
                cur.execute("""
                    INSERT INTO wells (name, x, y, z, h, eff_h, collector_ratio)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (name) DO UPDATE SET
                        x = EXCLUDED.x,
                        y = EXCLUDED.y,
                        z = EXCLUDED.z,
                        h = EXCLUDED.h,
                        eff_h = EXCLUDED.eff_h,
                        collector_ratio = EXCLUDED.collector_ratio,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id
                """, (name, x, y, z, h, eff_h, collector_ratio))
                
                well_id = cur.fetchone()[0]
                conn.commit()
                return well_id
        except Exception as e:
            conn.rollback()
            print(f"Ошибка сохранения скважины {name}: {e}")
            return None
        finally:
            self.put_connection(conn)
    
    def get_all_wells(self) -> pd.DataFrame:
        """Получить все скважины"""
        conn = self.get_connection()
        if not conn:
            return pd.DataFrame()
        
        try:
            query = """
                SELECT name as "Well", x as "X", y as "Y", z as "Z", 
                       h as "H", eff_h as "EFF_H", 
                       collector_ratio as "Доля_коллектора"
                FROM wells
                ORDER BY name
            """
            df = pd.read_sql_query(query, conn)
            return df
        except Exception as e:
            print(f"Ошибка получения скважин: {e}")
            return pd.DataFrame()
        finally:
            self.put_connection(conn)
    
    def get_well_by_name(self, name: str) -> Optional[Dict]:
        """Получить скважину по имени"""
        conn = self.get_connection()
        if not conn:
            return None
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, name, x, y, z, h, eff_h, collector_ratio
                    FROM wells WHERE name = %s
                """, (name,))
                row = cur.fetchone()
                if row:
                    return {
                        'id': row[0], 'name': row[1], 'x': row[2], 'y': row[3],
                        'z': row[4], 'h': row[5], 'eff_h': row[6], 'collector_ratio': row[7]
                    }
                return None
        finally:
            self.put_connection(conn)
    
    # ========== TRAJECTORIES ==========
    
    def save_trajectory(self, well_name: str, trajectory: np.ndarray):
        """
        Сохранить траекторию скважины
        
        Args:
            well_name: Название скважины
            trajectory: Массив [X, Y, Z, MD]
        """
        conn = self.get_connection()
        if not conn:
            return False
        
        try:
            with conn.cursor() as cur:
                # Получаем ID скважины используя то же соединение
                cur.execute("SELECT id FROM wells WHERE name = %s", (well_name,))
                row = cur.fetchone()
                
                if not row:
                    # Создаем скважину если её нет
                    cur.execute("""
                        INSERT INTO wells (name, x, y, z)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id
                    """, (well_name, float(trajectory[0, 0]), float(trajectory[0, 1]), float(trajectory[0, 2])))
                    well_id = cur.fetchone()[0]
                else:
                    well_id = row[0]
                
                # Удаляем старые точки траектории
                cur.execute("DELETE FROM trajectories WHERE well_id = %s", (well_id,))
                
                # Вставляем новые точки
                data = [
                    (well_id, i, float(point[0]), float(point[1]), float(point[2]), float(point[3]))
                    for i, point in enumerate(trajectory)
                ]
                
                execute_batch(cur, """
                    INSERT INTO trajectories (well_id, point_index, x, y, z, md)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, data)
                
                conn.commit()
                return True
        except Exception as e:
            conn.rollback()
            print(f"Ошибка сохранения траектории {well_name}: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.put_connection(conn)
    
    def get_all_trajectories(self) -> Dict[str, np.ndarray]:
        """Получить все траектории"""
        conn = self.get_connection()
        if not conn:
            return {}
        
        try:
            query = """
                SELECT w.name, t.x, t.y, t.z, t.md
                FROM trajectories t
                JOIN wells w ON t.well_id = w.id
                ORDER BY w.name, t.point_index
            """
            
            with conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()
            
            # Группируем по скважинам
            trajectories = {}
            current_well = None
            current_points = []
            
            for row in rows:
                well_name = row[0]
                point = [row[1], row[2], row[3], row[4]]
                
                if well_name != current_well:
                    if current_well and current_points:
                        trajectories[current_well] = np.array(current_points)
                    current_well = well_name
                    current_points = [point]
                else:
                    current_points.append(point)
            
            # Добавляем последнюю скважину
            if current_well and current_points:
                trajectories[current_well] = np.array(current_points)
            
            return trajectories
        except Exception as e:
            print(f"Ошибка получения траекторий: {e}")
            return {}
        finally:
            self.put_connection(conn)
    
    # ========== LAS DATA ==========
    
    def save_las_data(self, well_name: str, depth: np.ndarray, curve: np.ndarray):
        """Сохранить LAS данные"""
        conn = self.get_connection()
        if not conn:
            return False
        
        try:
            well = self.get_well_by_name(well_name)
            if not well:
                return False
            
            well_id = well['id']
            
            with conn.cursor() as cur:
                # Удаляем старые данные
                cur.execute("DELETE FROM las_data WHERE well_id = %s", (well_id,))
                
                # Вставляем новые данные
                data = [
                    (well_id, float(d), float(c))
                    for d, c in zip(depth, curve)
                    if c != -999.25 and not np.isnan(c)
                ]
                
                execute_batch(cur, """
                    INSERT INTO las_data (well_id, depth, curve_value)
                    VALUES (%s, %s, %s)
                """, data)
                
                conn.commit()
                return True
        except Exception as e:
            conn.rollback()
            print(f"Ошибка сохранения LAS данных: {e}")
            return False
        finally:
            self.put_connection(conn)
    
    def get_all_las_data(self) -> Dict[str, Dict]:
        """Получить все LAS данные"""
        conn = self.get_connection()
        if not conn:
            return {}
        
        try:
            query = """
                SELECT w.name, l.depth, l.curve_value
                FROM las_data l
                JOIN wells w ON l.well_id = w.id
                ORDER BY w.name, l.depth
            """
            
            with conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()
            
            # Группируем по скважинам
            las_data = {}
            current_well = None
            depths = []
            curves = []
            
            for row in rows:
                well_name = row[0]
                
                if well_name != current_well:
                    if current_well and depths:
                        las_data[current_well] = {
                            'well_name': current_well,
                            'depth': np.array(depths),
                            'curve': np.array(curves),
                            'null_value': -999.25
                        }
                    current_well = well_name
                    depths = [row[1]]
                    curves = [row[2]]
                else:
                    depths.append(row[1])
                    curves.append(row[2])
            
            # Добавляем последнюю скважину
            if current_well and depths:
                las_data[current_well] = {
                    'well_name': current_well,
                    'depth': np.array(depths),
                    'curve': np.array(curves),
                    'null_value': -999.25
                }
            
            return las_data
        except Exception as e:
            print(f"Ошибка получения LAS данных: {e}")
            return {}
        finally:
            self.put_connection(conn)
    
    # ========== BULK OPERATIONS ==========
    
    def load_data_from_files_to_db(self, well_data: pd.DataFrame,
                                     trajectories: Dict[str, np.ndarray],
                                     las_data: Dict[str, Dict]) -> bool:
        """
        Загрузить данные из файлов в БД
        
        Args:
            well_data: DataFrame с данными скважин
            trajectories: Словарь с траекториями
            las_data: Словарь с LAS данными
        
        Returns:
            True если успешно
        """
        wells_saved = 0
        wells_failed = 0
        trajectories_saved = 0
        trajectories_failed = 0
        las_saved = 0
        las_failed = 0
        
        try:
            # Сохраняем скважины
            print(f"Сохранение {len(well_data)} скважин...")
            for _, row in well_data.iterrows():
                try:
                    well_id = self.save_well(
                        row['Well'],
                        row['X'],
                        row['Y'],
                        row['Z'],
                        row.get('H'),
                        row.get('EFF_H')
                    )
                    if well_id:
                        wells_saved += 1
                    else:
                        wells_failed += 1
                        print(f"  ⚠️ Не удалось сохранить скважину {row['Well']}")
                except Exception as e:
                    wells_failed += 1
                    print(f"  ❌ Ошибка сохранения скважины {row['Well']}: {e}")
            
            print(f"✅ Скважины: {wells_saved} сохранено, {wells_failed} ошибок")
            
            # Сохраняем траектории
            print(f"Сохранение {len(trajectories)} траекторий...")
            for well_name, trajectory in trajectories.items():
                try:
                    success = self.save_trajectory(well_name, trajectory)
                    if success:
                        trajectories_saved += 1
                    else:
                        trajectories_failed += 1
                        print(f"  ⚠️ Не удалось сохранить траекторию {well_name}")
                except Exception as e:
                    trajectories_failed += 1
                    print(f"  ❌ Ошибка сохранения траектории {well_name}: {e}")
            
            print(f"✅ Траектории: {trajectories_saved} сохранено, {trajectories_failed} ошибок")
            
            # Сохраняем LAS данные
            print(f"Сохранение {len(las_data)} LAS файлов...")
            for well_name, las in las_data.items():
                try:
                    success = self.save_las_data(well_name, las['depth'], las['curve'])
                    if success:
                        las_saved += 1
                    else:
                        las_failed += 1
                        print(f"  ⚠️ Не удалось сохранить LAS данные {well_name}")
                except Exception as e:
                    las_failed += 1
                    print(f"  ❌ Ошибка сохранения LAS данных {well_name}: {e}")
            
            print(f"✅ LAS данные: {las_saved} сохранено, {las_failed} ошибок")
            
            # Итоговая статистика
            print(f"\n📊 Итого:")
            print(f"  Скважины: {wells_saved}/{len(well_data)}")
            print(f"  Траектории: {trajectories_saved}/{len(trajectories)}")
            print(f"  LAS файлы: {las_saved}/{len(las_data)}")
            
            # Считаем успешным если сохранено хотя бы 50% данных
            total_expected = len(well_data) + len(trajectories) + len(las_data)
            total_saved = wells_saved + trajectories_saved + las_saved
            success_rate = (total_saved / total_expected * 100) if total_expected > 0 else 0
            
            print(f"  Успешность: {success_rate:.1f}%")
            
            return success_rate >= 50
        except Exception as e:
            print(f"❌ Критическая ошибка загрузки данных в БД: {e}")
            import traceback
            traceback.print_exc()
            return False