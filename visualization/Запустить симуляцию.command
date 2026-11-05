#!/bin/bash
# Двойной клик по этому файлу запускает интерфейс и открывает браузер.
# Окно терминала остаётся открытым: в нём видно ошибки и им же всё гасится.
cd "$(dirname "$0")" || exit 1

echo "Симуляция оборота контейнеров"
echo "=============================="
echo

if ! command -v uv >/dev/null 2>&1; then
    for candidate in "$HOME/.local/bin" "$HOME/.cargo/bin" /opt/homebrew/bin /usr/local/bin; do
        [ -x "$candidate/uv" ] && export PATH="$candidate:$PATH" && break
    done
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "Не найден uv. Поставьте его одной командой и запустите этот файл снова:"
    echo
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo
    read -r -p "Enter чтобы закрыть..." _
    exit 1
fi

if [ ! -d "model" ] && [ -z "$SIMVIZ_MODEL_PATH" ]; then
    echo "Отсутствует папка model/ (этого не должно было произойти)"
    echo
    read -r -p "Enter чтобы закрыть..." _
    exit 1
fi

echo "Готовлю окружение, в первый раз это займёт минуту..."
if ! uv sync --quiet; then
    echo
    echo "Не удалось собрать окружение, смотрите ошибку выше."
    read -r -p "Enter чтобы закрыть..." _
    exit 1
fi

PORT=8050
while lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; do
    PORT=$((PORT + 1))
    [ $PORT -gt 8070 ] && { echo "Все порты 8050-8070 заняты."; read -r _; exit 1; }
done

echo "Запускаю на http://127.0.0.1:$PORT"
echo "Первый расчёт идёт около четырёх секунд."
echo
echo "Чтобы остановить, закройте это окно или нажмите Ctrl+C."
echo

( sleep 6; open "http://127.0.0.1:$PORT" ) &
uv run simviz --port "$PORT"
