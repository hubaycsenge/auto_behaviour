#!/usr/bin/env bash
# Install the ABC client (the GUI) on the machine you drive it from.
#
#   setup/install_client.sh            GUI + local media probing
#   setup/install_client.sh --ssh      + paramiko, for a client with no NAS mount
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="${ABC_CLIENT_VENV:-$root/.venv-client}"
want_ssh=0
for arg in "$@"; do
  case "$arg" in
    --ssh) want_ssh=1 ;;
    --venv=*) venv="${arg#*=}" ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

echo "== ABC client install =="
echo "   venv: $venv"

if [ ! -x "$venv/bin/python" ]; then
  python3 -m venv "$venv" 2>/dev/null || {
    python3 -m venv --without-pip "$venv"
    curl -sS https://bootstrap.pypa.io/get-pip.py -o "$venv/get-pip.py"
    "$venv/bin/python" "$venv/get-pip.py" -q
    rm -f "$venv/get-pip.py"
  }
fi

pip="$venv/bin/python -m pip"
$pip install -q --upgrade pip wheel
# PySide6-Essentials rather than the full PySide6: ABC uses QtWidgets only, and
# the full package adds ~700 MB of WebEngine and 3D modules it never touches.
$pip install -q "PySide6-Essentials>=6.6" "av>=11" "Pillow>=10" "numpy>=1.24"

if [ "$want_ssh" = 1 ]; then
  $pip install -q "paramiko>=3.4"
fi

echo
echo "== done =="
echo "Launch the GUI:  ABC_PYTHON=$venv/bin/python $root/bin/abc-gui"
