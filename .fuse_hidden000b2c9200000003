#!/usr/bin/env bash
#
# Publica a plataforma na internet por um túnel da Cloudflare e mostra um
# QR code para acesso pelo celular.
#
#   ./publicar.sh                 túnel rápido, endereço trycloudflare.com
#   ./publicar.sh --porta 8502    usa outra porta local
#   ./publicar.sh --nomeado NOME  túnel nomeado (exige conta e login prévios)
#   ./publicar.sh --so-qrcode URL gera o QR code de um endereço já existente
#
# Sobre o túnel rápido: a Cloudflare sorteia um subdomínio de trycloudflare.com,
# sem necessidade de conta nem de domínio próprio. O endereço muda a cada
# execução e o túnel morre junto com o script. Serve para demonstração, banca e
# congresso. Para um endereço fixo, use --nomeado com uma conta Cloudflare.
#
# ATENÇÃO: qualquer pessoa com o endereço acessa a plataforma. Não a use para
# processar imagens de pacientes: o projeto não tem aprovação de CEP.
#
set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RAIZ"

PORTA=8501
NOMEADO=""
SO_QRCODE=""
DESTINO="dist"

while [ $# -gt 0 ]; do
  case "$1" in
    --porta)     PORTA="$2"; shift 2 ;;
    --nomeado)   NOMEADO="$2"; shift 2 ;;
    --so-qrcode) SO_QRCODE="$2"; shift 2 ;;
    -h|--help)   sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "opção desconhecida: $1" >&2; exit 2 ;;
  esac
done

titulo() { printf '\n\033[1;36m▸ %s\033[0m\n' "$1"; }
ok()     { printf '  \033[0;32m✓\033[0m %s\n' "$1"; }
aviso()  { printf '  \033[0;33m!\033[0m %s\n' "$1"; }
erro()   { printf '\n\033[0;31m✗ %s\033[0m\n' "$1" >&2; exit 1; }

PID_STREAMLIT=""
PID_TUNEL=""
encerrar() {
  printf '\n'
  titulo "Encerrando"
  [ -n "$PID_TUNEL" ] && kill "$PID_TUNEL" 2>/dev/null && ok "túnel fechado"
  [ -n "$PID_STREAMLIT" ] && kill "$PID_STREAMLIT" 2>/dev/null && ok "servidor parado"
  exit 0
}
trap encerrar INT TERM

mkdir -p "$DESTINO"

# ---------------------------------------------------------------------------
# Geração do QR code
# ---------------------------------------------------------------------------
gerar_qrcode() {
  local url="$1"
  local arquivo="$DESTINO/qrcode_plataforma.png"

  python3 - "$url" "$arquivo" <<'PY'
import subprocess
import sys

url, saida = sys.argv[1], sys.argv[2]

try:
    import qrcode
except ImportError:
    print("  instalando a biblioteca qrcode…")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "qrcode[pil]"],
        check=True,
    )
    import qrcode

from qrcode.constants import ERROR_CORRECT_M

qr = qrcode.QRCode(version=None, error_correction=ERROR_CORRECT_M,
                   box_size=12, border=3)
qr.add_data(url)
qr.make(fit=True)

# Versão para o terminal, útil quando o script roda por SSH
qr.print_ascii(invert=True)

imagem = qr.make_image(fill_color="#14232F", back_color="white")
imagem.save(saida)
print(f"  QR code salvo em {saida}")
PY
}

# ---------------------------------------------------------------------------
# Modo "só QR code"
# ---------------------------------------------------------------------------
if [ -n "$SO_QRCODE" ]; then
  titulo "Gerando QR code para $SO_QRCODE"
  gerar_qrcode "$SO_QRCODE"
  exit 0
fi

# ---------------------------------------------------------------------------
titulo "Conferindo o ambiente"

command -v python3 >/dev/null 2>&1 || erro "python3 não encontrado"
python3 -c "import streamlit" 2>/dev/null || erro \
  "Streamlit não instalado. Rode: pip install -r requirements.txt"
ok "streamlit disponível"

if ! command -v cloudflared >/dev/null 2>&1; then
  aviso "cloudflared não encontrado; instalando na pasta do projeto"
  ARQ=""
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)  ARQ="linux-amd64" ;;
    Linux-aarch64) ARQ="linux-arm64" ;;
    Darwin-arm64)  ARQ="darwin-arm64" ;;
    Darwin-x86_64) ARQ="darwin-amd64" ;;
    *) erro "arquitetura não reconhecida. Instale o cloudflared manualmente:
    https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/" ;;
  esac
  BASE="https://github.com/cloudflare/cloudflared/releases/latest/download"
  if [ "${ARQ#darwin}" != "$ARQ" ]; then
    curl -fL "$BASE/cloudflared-$ARQ.tgz" -o /tmp/cf.tgz \
      && tar -xzf /tmp/cf.tgz -C "$RAIZ" && chmod +x "$RAIZ/cloudflared"
  else
    curl -fL "$BASE/cloudflared-$ARQ" -o "$RAIZ/cloudflared" \
      && chmod +x "$RAIZ/cloudflared"
  fi
  export PATH="$RAIZ:$PATH"
  command -v cloudflared >/dev/null 2>&1 || erro "falha ao instalar o cloudflared"
fi
ok "cloudflared disponível"

if [ ! -f "$RAIZ/app.py" ]; then
  erro "app.py não encontrado em $RAIZ"
fi

# ---------------------------------------------------------------------------
titulo "Subindo o servidor local na porta $PORTA"

LOG_STREAMLIT="$(mktemp)"
streamlit run app.py \
  --server.port "$PORTA" \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false \
  > "$LOG_STREAMLIT" 2>&1 &
PID_STREAMLIT=$!

for _ in $(seq 1 45); do
  if curl -sf "http://localhost:$PORTA/_stcore/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$PID_STREAMLIT" 2>/dev/null; then
    echo; cat "$LOG_STREAMLIT"
    erro "o servidor não subiu"
  fi
  sleep 1
done
curl -sf "http://localhost:$PORTA/_stcore/health" >/dev/null 2>&1 \
  || erro "o servidor não respondeu ao teste de saúde a tempo"
ok "servidor no ar em http://localhost:$PORTA"

# ---------------------------------------------------------------------------
titulo "Abrindo o túnel da Cloudflare"

LOG_TUNEL="$(mktemp)"
if [ -n "$NOMEADO" ]; then
  cloudflared tunnel run --url "http://localhost:$PORTA" "$NOMEADO" \
    > "$LOG_TUNEL" 2>&1 &
  PID_TUNEL=$!
  sleep 6
  URL_PUBLICA="(endereço configurado no painel da Cloudflare para o túnel $NOMEADO)"
  ok "túnel nomeado '$NOMEADO' em execução"
else
  cloudflared tunnel --url "http://localhost:$PORTA" --no-autoupdate \
    > "$LOG_TUNEL" 2>&1 &
  PID_TUNEL=$!

  URL_PUBLICA=""
  for _ in $(seq 1 40); do
    URL_PUBLICA=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG_TUNEL" \
                  | head -n1 || true)
    [ -n "$URL_PUBLICA" ] && break
    if ! kill -0 "$PID_TUNEL" 2>/dev/null; then
      echo; cat "$LOG_TUNEL"
      erro "o túnel não subiu"
    fi
    sleep 1
  done
  [ -n "$URL_PUBLICA" ] || { cat "$LOG_TUNEL"; erro "não consegui ler o endereço público"; }
  ok "endereço público: $URL_PUBLICA"
  printf '%s\n' "$URL_PUBLICA" > "$DESTINO/endereco_publico.txt"
fi

# ---------------------------------------------------------------------------
titulo "QR code de acesso"

if [ -n "$NOMEADO" ]; then
  aviso "túnel nomeado: informe o endereço final com --so-qrcode https://seu.dominio"
else
  gerar_qrcode "$URL_PUBLICA"
fi

# ---------------------------------------------------------------------------
printf '\n\033[1;32mNo ar.\033[0m\n\n'
printf '  Endereço  %s\n' "$URL_PUBLICA"
printf '  Local     http://localhost:%s\n' "$PORTA"
printf '  QR code   %s/qrcode_plataforma.png\n\n' "$DESTINO"
printf '  O endereço do túnel rápido é temporário e muda a cada execução.\n'
printf '  Qualquer pessoa com o link acessa a plataforma: não a utilize para\n'
printf '  processar imagens de pacientes.\n\n'
printf '  Ctrl+C encerra o túnel e o servidor.\n\n'

wait "$PID_STREAMLIT"
