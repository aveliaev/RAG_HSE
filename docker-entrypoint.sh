#!/bin/sh
set -e
mkdir -p "$DATA_DIR"
# При первом запуске кладём в volume кеш с блэклистом из репозитория
if [ ! -f "$DATA_DIR/dynamic_cache.json" ] && [ -f /app/bot/dynamic_cache.json ]; then
    cp /app/bot/dynamic_cache.json "$DATA_DIR/dynamic_cache.json"
fi
exec "$@"
