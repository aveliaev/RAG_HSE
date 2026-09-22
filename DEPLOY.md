# Развёртывание на сервере (Yandex Cloud)

Бот работает через polling, поэтому домен, SSL и открытые порты не нужны.
На сервере запускаются два контейнера: `bot` (Telegram-бот) и `dashboard` (Streamlit).
Логи, кеш и индекс ChromaDB хранятся в Docker-volume `bot-data` и не пропадают при обновлении.

## 1. Создать виртуальную машину

console.yandex.cloud → **Compute Cloud** → **Создать ВМ**:

- ОС: **Ubuntu 24.04 LTS**
- vCPU: **2** (гарантированная доля 100%), RAM: **4 ГБ**, диск: **30 ГБ** SSD
- Публичный IP: **автоматически**
- Доступ: логин (например `fkn`) и **публичный SSH-ключ**: содержимое `~/.ssh/id_ed25519.pub`

## 2. Подготовить сервер

```bash
ssh fkn@<IP_СЕРВЕРА>
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
exit   # перелогиниться, чтобы docker работал без sudo
```

## 3. Запустить бота

```bash
ssh fkn@<IP_СЕРВЕРА>
git clone https://github.com/aveliaev/RAG_HSE.git
cd RAG_HSE
cp bot/.env.example bot/.env
nano bot/.env          # вписать TELEGRAM_TOKEN, YANDEX_API_KEY, YANDEX_FOLDER_ID и т.д.
docker compose up -d --build
docker compose logs -f bot   # дождаться «Индекс готов» и «Polling...», выход — Ctrl+C
```

⚠️ Перед запуском на сервере останови бота локально. Если два экземпляра с одним токеном
одновременно делают polling, Telegram возвращает ошибку `Conflict`.

## Если Telegram недоступен с сервера (серверы в РФ)

Если в логах бота `telegram.error.TimedOut`, значит `api.telegram.org` с сервера не открывается.
В этом случае бот выходит в интернет через OpenVPN-контейнер:

```bash
scp мой_vpn.ovpn fkn@<IP_СЕРВЕРА>:RAG_HSE/vpn/client.ovpn   # с локальной машины
echo "COMPOSE_FILE=docker-compose.yml:docker-compose.vpn.yml" > .env   # на сервере, в корне проекта
docker compose up -d --build
```

Через VPN идёт только контейнер бота. SSH и остальная ВМ работают напрямую.

## Дашборд

Дашборд доступен только с самого сервера (он без пароля, а в логах вопросы пользователей).
Открыть его у себя можно через SSH-туннель:

```bash
ssh -L 8501:localhost:8501 fkn@<IP_СЕРВЕРА>
```

После этого дашборд открывается по адресу http://localhost:8501.

## Обновление

```bash
cd RAG_HSE && git pull && docker compose up -d --build
```

Если изменился `dataset/`, индекс нужно пересобрать (при старте бот пересобирает его только когда индекс пустой):

```bash
docker compose run --rm bot python _ri4.py
docker compose restart bot
```

## Полезное

```bash
docker compose ps                  # статус контейнеров
docker compose logs --tail 100 bot # последние логи
docker compose down                # остановить (данные в volume сохраняются)
```
