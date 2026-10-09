#!/bin/bash
# Double-click this file to open Pianoscribe (macOS; also works on Linux).
cd "$(dirname "$0")" || exit 1

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is not installed."
  echo "Download it from https://www.python.org/downloads/ , install it,"
  echo "then double-click this file again."
  open "https://www.python.org/downloads/" 2>/dev/null || xdg-open "https://www.python.org/downloads/" 2>/dev/null
  read -r -p "Press Enter to close this window."
  exit 1
fi

if [ ! -x .venv/bin/python ] || ! .venv/bin/python -c "import pianoscribe.app" >/dev/null 2>&1; then
  echo "First start: setting up Pianoscribe. This takes a few minutes and only happens once..."
  python3 -m venv .venv \
    && .venv/bin/python -m pip install --quiet --upgrade pip \
    && .venv/bin/python -m pip install --quiet -e . \
    || { echo; echo "Setup failed - see the messages above."; read -r -p "Press Enter to close this window."; exit 1; }
fi

exec .venv/bin/python -m pianoscribe.app
