#!/usr/bin/env bash
set -euo pipefail

REPO="${RECALL_REPO:-jashan/recall}"
VERSION="${RECALL_VERSION:-latest}"
INSTALL_DIR="${RECALL_INSTALL_DIR:-$HOME/.local/bin}"
DATA_DIR="${RECALL_DATA_DIR:-$HOME/.local/share/recall}"
BACKEND_VENV="$DATA_DIR/backend-venv"
TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "$TMP_DIR"
}

trap cleanup EXIT

require() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

require curl
require tar
require python3

python3 - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11 or newer is required")
PY

case "$(uname -s)" in
  Linux)
    os_name="linux"
    ;;
  Darwin)
    os_name="darwin"
    ;;
  *)
    echo "Unsupported platform: $(uname -s)" >&2
    exit 1
    ;;
esac

case "$(uname -m)" in
  x86_64|amd64)
    arch_name="x86_64"
    ;;
  arm64|aarch64)
    arch_name="arm64"
    ;;
  *)
    echo "Unsupported architecture: $(uname -m)" >&2
    exit 1
    ;;
esac

if [[ "$VERSION" == "latest" ]]; then
  api_url="https://api.github.com/repos/$REPO/releases/latest"
  VERSION="$(curl -fsSL "$api_url" | python3 -c 'import json, sys; print(json.load(sys.stdin)["tag_name"])')"
fi

release_base="https://github.com/$REPO/releases/download/$VERSION"
cli_asset="recall-cli-${os_name}-${arch_name}.tar.gz"
backend_asset="recall-backend-py3-none-any.whl"

mkdir -p "$INSTALL_DIR" "$DATA_DIR" "$BACKEND_VENV"

curl -fsSL -o "$TMP_DIR/$cli_asset" "$release_base/$cli_asset"
curl -fsSL -o "$TMP_DIR/$backend_asset" "$release_base/$backend_asset"

tar -xzf "$TMP_DIR/$cli_asset" -C "$TMP_DIR"
install -m 755 "$TMP_DIR/recall" "$INSTALL_DIR/recall"

python3 -m venv "$BACKEND_VENV"
"$BACKEND_VENV/bin/python" -m pip install --upgrade pip >/dev/null
"$BACKEND_VENV/bin/python" -m pip install "$TMP_DIR/$backend_asset" >/dev/null

cat > "$INSTALL_DIR/recall-backend" <<EOF
#!/usr/bin/env bash
exec "$BACKEND_VENV/bin/recall-backend" "\$@"
EOF
chmod 755 "$INSTALL_DIR/recall-backend"

if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
  echo "Installed to $INSTALL_DIR"
  echo "Add $INSTALL_DIR to your PATH if it is not already there."
fi

echo "Recall $VERSION installed successfully."