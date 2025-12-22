#!/bin/bash

set -e  # Остановить при ошибке

echo "=========================================="
echo "ПОЛНЫЙ СБРОС БАЗЫ ДАННЫХ POSTGRESQL"
echo "=========================================="
echo ""

echo "Шаг 1: Остановка контейнеров..."
docker-compose down
echo "✓ Контейнеры остановлены"
echo ""

echo "Шаг 2: Удаление volumes..."
docker-compose down -v
docker volume rm pygeo_postgres_data 2>/dev/null && echo "✓ Volume pygeo_postgres_data удален" || echo "⚠ Volume не существовал"
echo ""

echo "Шаг 3: Проверка что volume удален..."
if docker volume ls | grep -q pygeo_postgres_data; then
    echo "❌ ОШИБКА: Volume все еще существует!"
    echo "Попытка принудительного удаления..."
    docker volume rm -f pygeo_postgres_data
else
    echo "✓ Volume успешно удален"
fi
echo ""

echo "Шаг 4: Проверка файла init_db.sql..."
if [ -f "./backend/init_db.sql" ]; then
    echo "✓ Файл backend/init_db.sql найден"
    echo "Размер: $(wc -l < ./backend/init_db.sql) строк"
else
    echo "❌ ОШИБКА: Файл backend/init_db.sql не найден!"
    exit 1
fi
echo ""

echo "Шаг 5: Запуск контейнеров..."
docker-compose up -d
echo "✓ Контейнеры запущены"
echo ""

echo "Шаг 6: Ожидание инициализации PostgreSQL..."
echo "Это может занять до 60 секунд..."
for i in {1..60}; do
    if docker-compose exec -T postgres pg_isready -U pygeo_user -d pygeo_db > /dev/null 2>&1; then
        echo "✓ PostgreSQL готов к работе (через $i секунд)"
        break
    fi
    echo -n "."
    sleep 1
done
echo ""
echo ""

echo "Шаг 7: Проверка выполнения init_db.sql..."
echo "Логи инициализации:"
docker-compose logs postgres | grep -A 5 "init_db.sql" || echo "⚠ Упоминаний init_db.sql в логах не найдено"
echo ""

echo "Шаг 8: Проверка созданных таблиц..."
echo ""
docker-compose exec -T postgres psql -U pygeo_user -d pygeo_db -c "\dt"
echo ""

echo "Шаг 9: Детальная информация о таблицах..."
TABLES=$(docker-compose exec -T postgres psql -U pygeo_user -d pygeo_db -t -c "SELECT tablename FROM pg_tables WHERE schemaname='public';" | tr -d ' ')

if [ -z "$TABLES" ]; then
    echo "❌ ТАБЛИЦЫ НЕ СОЗДАНЫ!"
    echo ""
    echo "Полные логи PostgreSQL:"
    docker-compose logs postgres
    echo ""
    echo "Содержимое /docker-entrypoint-initdb.d/:"
    docker-compose exec postgres ls -la /docker-entrypoint-initdb.d/
    exit 1
else
    echo "✓ Найдены таблицы:"
    echo "$TABLES"
    echo ""
    for table in $TABLES; do
        echo "Структура таблицы $table:"
        docker-compose exec -T postgres psql -U pygeo_user -d pygeo_db -c "\d $table"
        echo ""
    done
fi

echo "=========================================="
echo "✓ СБРОС ЗАВЕРШЕН УСПЕШНО!"
echo "=========================================="
echo ""
echo "База данных готова к использованию."
echo "Приложение доступно на http://localhost:8501"