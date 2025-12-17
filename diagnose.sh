#!/bin/bash

echo "=========================================="
echo "ДИАГНОСТИКА POSTGRESQL"
echo "=========================================="
echo ""

echo "1. Проверка запущенных контейнеров:"
docker-compose ps
echo ""

echo "2. Проверка volumes:"
docker volume ls | grep pygeo
echo ""

echo "3. Проверка файла init_db.sql в контейнере:"
docker-compose exec postgres ls -la /docker-entrypoint-initdb.d/ 2>/dev/null || echo "Контейнер не запущен"
echo ""

echo "4. Полные логи PostgreSQL:"
docker-compose logs postgres
echo ""

echo "5. Попытка подключения к БД:"
docker-compose exec postgres pg_isready -U pygeo_user -d pygeo_db 2>/dev/null || echo "Не удалось подключиться"
echo ""

echo "6. Список баз данных:"
docker-compose exec -T postgres psql -U pygeo_user -d pygeo_db -c "\l" 2>/dev/null || echo "Не удалось получить список БД"
echo ""

echo "7. Список таблиц:"
docker-compose exec -T postgres psql -U pygeo_user -d pygeo_db -c "\dt" 2>/dev/null || echo "Не удалось получить список таблиц"
echo ""

echo "8. Проверка содержимого volume:"
docker volume inspect pygeo_postgres_data 2>/dev/null || echo "Volume не найден"
echo ""

echo "=========================================="
echo "ДИАГНОСТИКА ЗАВЕРШЕНА"
echo "=========================================="