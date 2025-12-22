# Как работает автоматическая инициализация PostgreSQL

## Механизм Docker Entrypoint

PostgreSQL официальный Docker образ имеет встроенный механизм автоматической инициализации базы данных.

### Принцип работы:

1. **При первом запуске контейнера** (когда volume `/var/lib/postgresql/data` пустой):
   - PostgreSQL создает новый кластер БД
   - Выполняет все скрипты из директории `/docker-entrypoint-initdb.d/`
   - Скрипты выполняются в алфавитном порядке

2. **При последующих запусках** (когда volume уже содержит данные):
   - Скрипты инициализации **НЕ выполняются**
   - PostgreSQL просто запускается с существующими данными

## Наша конфигурация

### В docker-compose.yml:

```yaml
postgres:
  image: postgres:15-alpine
  volumes:
    - postgres_data:/var/lib/postgresql/data              # Персистентное хранилище
    - ./backend/init_db.sql:/docker-entrypoint-initdb.d/init_db.sql  # Скрипт инициализации
```

### Что происходит:

```
┌─────────────────────────────────────────────────────────────┐
│  1. Docker запускает контейнер postgres                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  2. Проверка: volume postgres_data пустой?                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
                    ┌─────┴─────┐
                    │           │
                   ДА          НЕТ
                    │           │
                    ↓           ↓
    ┌───────────────────────┐  ┌──────────────────────────┐
    │ 3. Инициализация:     │  │ 3. Пропуск инициализации │
    │ - Создание кластера   │  │ - Использование          │
    │ - Создание БД         │  │   существующих данных    │
    │ - Создание пользователя│ │                          │
    │ - Выполнение init_db.sql│ │                          │
    └───────────────────────┘  └──────────────────────────┘
                    │           │
                    └─────┬─────┘
                          ↓
            ┌─────────────────────────────┐
            │ 4. PostgreSQL готов к работе│
            └─────────────────────────────┘
```

## Содержимое init_db.sql

Наш скрипт создает:

### 1. Таблицы:
```sql
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

CREATE TABLE IF NOT EXISTS las_data (
    id SERIAL PRIMARY KEY,
    well_id INTEGER REFERENCES wells(id) ON DELETE CASCADE,
    depth FLOAT NOT NULL,
    curve_value FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 2. Индексы для оптимизации:
```sql
CREATE INDEX IF NOT EXISTS idx_wells_name ON wells(name);
CREATE INDEX IF NOT EXISTS idx_trajectories_well_id ON trajectories(well_id);
CREATE INDEX IF NOT EXISTS idx_las_data_well_id ON las_data(well_id);
```

### 3. Триггеры:
```sql
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_wells_updated_at BEFORE UPDATE ON wells
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
```

### 4. Комментарии к таблицам и колонкам

## Когда нужно пересоздать БД

### Сценарии:

1. **Изменили init_db.sql** - нужно пересоздать БД
2. **Таблицы не создались** - volume уже существовал
3. **Хотите начать с чистого листа** - удалить все данные

### Решение:

```bash
# Способ 1: Автоматический (рекомендуется)
chmod +x reset_db.sh
./reset_db.sh

# Способ 2: Ручной
docker-compose down -v                    # Остановить и удалить volumes
docker volume rm pygeo_postgres_data      # Удалить volume (если остался)
docker-compose up -d                      # Запустить заново
```

## Проверка результата

```bash
# Способ 1: Скрипт проверки
chmod +x check_db.sh
./check_db.sh

# Способ 2: Ручная проверка
docker-compose exec postgres psql -U pygeo_user -d pygeo_db -c "\dt"
```

Должны увидеть:
```
             List of relations
 Schema |      Name      | Type  |   Owner    
--------+----------------+-------+------------
 public | las_data       | table | pygeo_user
 public | trajectories   | table | pygeo_user
 public | wells          | table | pygeo_user
```

## Важные моменты

### ✅ Правильно:
- Скрипт выполняется **автоматически** при первом запуске
- Используется `CREATE TABLE IF NOT EXISTS` - безопасно
- Volume обеспечивает **персистентность** данных

### ❌ Частые ошибки:
- Запуск `docker-compose up` без удаления volume - скрипт не выполнится
- Изменение init_db.sql без пересоздания БД - изменения не применятся
- Неправильный healthcheck - приложение не дождется готовности БД

## Альтернативные подходы

### 1. Миграции (для production):
```python
# Использовать Alembic или Django migrations
alembic upgrade head
```

### 2. Ручное выполнение:
```bash
docker-compose exec postgres psql -U pygeo_user -d pygeo_db -f /docker-entrypoint-initdb.d/init_db.sql
```

### 3. Программная инициализация:
```python
# В backend/database.py
def init_database():
    with open('backend/init_db.sql', 'r') as f:
        sql = f.read()
    conn.execute(sql)
```

## Логи инициализации

Посмотреть что происходило при инициализации:

```bash
docker-compose logs postgres | grep -A 20 "database system is ready"
```

Должны увидеть:
```
postgres_1  | PostgreSQL init process complete; ready for start up.
postgres_1  | 
postgres_1  | 2025-12-16 14:00:00.000 UTC [1] LOG:  database system is ready to accept connections
```

## Резюме

**init_db.sql используется автоматически:**
1. Docker монтирует файл в `/docker-entrypoint-initdb.d/`
2. PostgreSQL выполняет его при первом запуске
3. Создаются таблицы, индексы, триггеры
4. Данные сохраняются в volume `postgres_data`
5. При последующих запусках используются существующие данные

**Для пересоздания БД:**
```bash
./reset_db.sh
```

**Для проверки:**
```bash
./check_db.sh