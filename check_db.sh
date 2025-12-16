#!/bin/bash

echo "=== Проверка состояния Docker контейнеров ==="
docker-compose ps

echo ""
echo "=== Проверка здоровья PostgreSQL ==="
docker-compose exec postgres pg_isready -U pygeo_user -d pygeo_db

echo ""
echo "=== Проверка логов PostgreSQL (последние 20 строк) ==="
docker-compose logs postgres | tail -20

echo ""
echo "=== Проверка таблиц в базе данных ==="
docker-compose exec -T postgres psql -U pygeo_user -d pygeo_db -c "\dt"

echo ""
echo "=== Проверка схемы таблицы wells ==="
docker-compose exec -T postgres psql -U pygeo_user -d pygeo_db -c "\d wells"

echo ""
echo "=== Количество записей в таблицах ==="
docker-compose exec -T postgres psql -U pygeo_user -d pygeo_db -c "
SELECT
    'wells' as table_name, COUNT(*) as count FROM wells
UNION ALL
SELECT
    'trajectories' as table_name, COUNT(*) as count FROM trajectories
UNION ALL
SELECT
    'las_data' as table_name, COUNT(*) as count FROM las_data;
"

echo ""
echo "=== Проверка завершена ==="