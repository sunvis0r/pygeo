-- Инициализация базы данных PyGeo

-- Таблица скважин
CREATE TABLE IF NOT EXISTS wells (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    x FLOAT NOT NULL,
    y FLOAT NOT NULL,
    z FLOAT NOT NULL,
    h FLOAT,
    eff_h FLOAT,
    collector_ratio FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Таблица траекторий скважин
CREATE TABLE IF NOT EXISTS trajectories (
    id SERIAL PRIMARY KEY,
    well_id INTEGER REFERENCES wells(id) ON DELETE CASCADE,
    point_index INTEGER NOT NULL,
    x FLOAT NOT NULL,
    y FLOAT NOT NULL,
    z FLOAT NOT NULL,
    md FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(well_id, point_index)
);

-- Таблица LAS данных (каротаж)
CREATE TABLE IF NOT EXISTS las_data (
    id SERIAL PRIMARY KEY,
    well_id INTEGER REFERENCES wells(id) ON DELETE CASCADE,
    depth FLOAT NOT NULL,
    curve_value FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Индексы для оптимизации запросов
CREATE INDEX IF NOT EXISTS idx_wells_name ON wells(name);
CREATE INDEX IF NOT EXISTS idx_trajectories_well_id ON trajectories(well_id);
CREATE INDEX IF NOT EXISTS idx_trajectories_point_index ON trajectories(well_id, point_index);
CREATE INDEX IF NOT EXISTS idx_las_data_well_id ON las_data(well_id);
CREATE INDEX IF NOT EXISTS idx_las_data_depth ON las_data(well_id, depth);

-- Функция для автоматического обновления updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Триггер для обновления updated_at
CREATE TRIGGER update_wells_updated_at BEFORE UPDATE ON wells
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Комментарии к таблицам
COMMENT ON TABLE wells IS 'Основная информация о скважинах';
COMMENT ON TABLE trajectories IS 'Траектории скважин (точки пути)';
COMMENT ON TABLE las_data IS 'Данные каротажа из LAS файлов';

COMMENT ON COLUMN wells.name IS 'Название скважины (например, WELL_001)';
COMMENT ON COLUMN wells.x IS 'Координата X устья скважины';
COMMENT ON COLUMN wells.y IS 'Координата Y устья скважины';
COMMENT ON COLUMN wells.z IS 'Координата Z (кровля пласта)';
COMMENT ON COLUMN wells.h IS 'Мощность пласта (м)';
COMMENT ON COLUMN wells.eff_h IS 'Эффективная мощность (м)';
COMMENT ON COLUMN wells.collector_ratio IS 'Доля коллектора (EFF_H / H)';

COMMENT ON COLUMN trajectories.point_index IS 'Индекс точки в траектории';
COMMENT ON COLUMN trajectories.md IS 'Measured Depth - длина по стволу скважины';

COMMENT ON COLUMN las_data.depth IS 'Глубина измерения (MD)';
COMMENT ON COLUMN las_data.curve_value IS 'Значение кривой ГИС (0=неколлектор, 1=коллектор)';