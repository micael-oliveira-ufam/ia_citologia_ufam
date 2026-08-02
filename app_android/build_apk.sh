#!/usr/bin/env bash
#
# Gera o APK do CitoPred, aplicativo de apoio ao diagnóstico citológico.
#
#   ./build_apk.sh                  APKs por arquitetura (menores, recomendado)
#   ./build_apk.sh --universal      APK único, roda em qualquer aparelho
#   ./build_apk.sh --debug          build de depuração, para testar rápido
#   ./build_apk.sh --limpar         apaga artefatos antes de compilar
#
# Assinatura de release (opcional): exporte as quatro variáveis abaixo e o
# script escreve o android/key.properties para você.
#   KEYSTORE_ARQUIVO  KEYSTORE_SENHA  KEY_ALIAS  KEY_SENHA
#
set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RAIZ"

MODELO="assets/model/citologia_convnext.onnx"
CHECKPOINT_PADRAO="convnext_liquid_based_citology_IARC_digital_atlas_01_08_26.pt"
# Mesmo endereço usado pela plataforma web: os dois precisam dos mesmos pesos.
CHECKPOINT_URL="https://github.com/micael-oliveira-ufam/ia_citologia_ufam/releases/download/v.1.1/$CHECKPOINT_PADRAO"
MIN_SDK=24
DESTINO="dist"

UNIVERSAL=0
MODO="release"
LIMPAR=0
for arg in "$@"; do
  case "$arg" in
    --universal) UNIVERSAL=1 ;;
    --debug)     MODO="debug" ;;
    --limpar)    LIMPAR=1 ;;
    -h|--help)   sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "opção desconhecida: $arg" >&2; exit 2 ;;
  esac
done

titulo() { printf '\n\033[1;36m▸ %s\033[0m\n' "$1"; }
ok()     { printf '  \033[0;32m✓\033[0m %s\n' "$1"; }
aviso()  { printf '  \033[0;33m!\033[0m %s\n' "$1"; }
erro()   { printf '\n\033[0;31m✗ %s\033[0m\n' "$1" >&2; exit 1; }

# ---------------------------------------------------------------------------
titulo "Conferindo o ambiente"

command -v flutter >/dev/null 2>&1 || erro \
  "Flutter não encontrado no PATH. Instale em https://docs.flutter.dev/get-started/install"
ok "flutter $(flutter --version 2>/dev/null | head -n1 | awk '{print $2}')"

# O texto do flutter doctor muda entre versões e vem com códigos de cor, então
# a checagem é tolerante: só reclama se não houver sinal algum de Android.
DOCTOR=$(flutter doctor 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' || true)
if printf '%s' "$DOCTOR" | grep -qiE "^\[.\].*Android (toolchain|Studio)"; then
  if printf '%s' "$DOCTOR" | grep -qiE "^\[✓\].*Android toolchain"; then
    ok "Android toolchain confirmado"
  else
    aviso "o Android toolchain aparece com pendências no flutter doctor"
    aviso "rode 'flutter doctor -v' se a compilação falhar"
  fi
else
  aviso "não encontrei o Android toolchain no flutter doctor; a compilação pode falhar"
fi

if [ -z "${JAVA_HOME:-}" ] && ! command -v java >/dev/null 2>&1; then
  erro "Java não encontrado. Instale o JDK 17 e defina JAVA_HOME."
fi
ok "Java disponível"

# ---------------------------------------------------------------------------
titulo "Conferindo o modelo embarcado"

if [ ! -f "$MODELO" ]; then
  aviso "modelo ausente em $MODELO"
  if [ ! -f "$CHECKPOINT_PADRAO" ]; then
    echo "  Checkpoint não encontrado. Baixando do mesmo endereço usado pela"
    echo "  plataforma web, para garantir que os dois usem os mesmos pesos…"
    if command -v curl >/dev/null 2>&1; then
      curl -fL --progress-bar "$CHECKPOINT_URL" -o "$CHECKPOINT_PADRAO.parcial" \
        && mv "$CHECKPOINT_PADRAO.parcial" "$CHECKPOINT_PADRAO" \
        || { rm -f "$CHECKPOINT_PADRAO.parcial"; erro "falha ao baixar o checkpoint"; }
    elif command -v wget >/dev/null 2>&1; then
      wget -q --show-progress -O "$CHECKPOINT_PADRAO.parcial" "$CHECKPOINT_URL" \
        && mv "$CHECKPOINT_PADRAO.parcial" "$CHECKPOINT_PADRAO" \
        || { rm -f "$CHECKPOINT_PADRAO.parcial"; erro "falha ao baixar o checkpoint"; }
    else
      erro "nem curl nem wget disponíveis. Baixe manualmente:
    $CHECKPOINT_URL"
    fi
    ok "checkpoint baixado"
  else
    ok "checkpoint encontrado na raiz"
  fi
  echo "  Exportando para ONNX…"
  python3 tools/export_onnx.py \
    --checkpoint "$CHECKPOINT_PADRAO" \
    --saida "$MODELO" \
    --exemplos assets/exemplos
fi

TAM_MODELO=$(du -m "$MODELO" | cut -f1)
ok "modelo presente (${TAM_MODELO} MB)"
[ "$TAM_MODELO" -gt 95 ] && aviso \
  "modelo acima de 95 MB, o APK ficará pesado para distribuir. Considere quantizar."

# ---------------------------------------------------------------------------
titulo "Preparando o projeto Android"

if [ ! -f "android/settings.gradle" ] && [ ! -f "android/settings.gradle.kts" ]; then
  echo "  Gerando o esqueleto nativo…"
  # flutter create não sobrescreve arquivos existentes sem --overwrite, mas o
  # lib/ e o pubspec.yaml são o coração do projeto: copiamos por precaução.
  RESGATE=$(mktemp -d)
  cp -r lib pubspec.yaml "$RESGATE"/ 2>/dev/null || true
  PROGUARD_SALVO=""
  [ -f "android/app/proguard-rules.pro" ] && \
    PROGUARD_SALVO=$(cat android/app/proguard-rules.pro)

  flutter create --platforms=android --project-name citopred \
    --org br.ufam.fcf . >/dev/null

  cp -r "$RESGATE"/lib . 2>/dev/null || true
  cp "$RESGATE"/pubspec.yaml . 2>/dev/null || true
  rm -rf "$RESGATE"
  mkdir -p android/app
  [ -n "$PROGUARD_SALVO" ] && printf '%s\n' "$PROGUARD_SALVO" > android/app/proguard-rules.pro
  ok "esqueleto android/ criado (applicationId br.ufam.fcf.citopred)"
else
  ok "android/ já existe"
fi

# O ONNX Runtime exige API 24 ou superior.
GRADLE_APP=""
for candidato in android/app/build.gradle.kts android/app/build.gradle; do
  [ -f "$candidato" ] && GRADLE_APP="$candidato" && break
done
[ -n "$GRADLE_APP" ] || erro "não encontrei android/app/build.gradle(.kts)"

if grep -q "minSdk = flutter.minSdkVersion" "$GRADLE_APP"; then
  sed -i.bak "s/minSdk = flutter.minSdkVersion/minSdk = $MIN_SDK/" "$GRADLE_APP"
  ok "minSdk fixado em $MIN_SDK"
elif grep -q "minSdkVersion flutter.minSdkVersion" "$GRADLE_APP"; then
  sed -i.bak "s/minSdkVersion flutter.minSdkVersion/minSdkVersion $MIN_SDK/" "$GRADLE_APP"
  ok "minSdk fixado em $MIN_SDK"
elif grep -qE "minSdk(Version)? *=? *(2[4-9]|3[0-9])" "$GRADLE_APP"; then
  ok "minSdk já compatível"
else
  aviso "não consegui ajustar o minSdk automaticamente: confira $GRADLE_APP (precisa ser >= $MIN_SDK)"
fi
rm -f "${GRADLE_APP}.bak"

# --- Nome visível do aplicativo ---
MANIFESTO="android/app/src/main/AndroidManifest.xml"
if [ -f "$MANIFESTO" ]; then
  if grep -q 'android:label="citopred"' "$MANIFESTO"; then
    sed -i.bak 's/android:label="citopred"/android:label="CitoPred"/' "$MANIFESTO"
    rm -f "$MANIFESTO.bak"
    ok "rótulo do aplicativo definido como CitoPred"
  elif grep -q 'android:label="CitoPred"' "$MANIFESTO"; then
    ok "rótulo já é CitoPred"
  else
    aviso "não consegui ajustar android:label; confira $MANIFESTO"
  fi
fi

# --- Ícone de microscópio ---
if command -v python3 >/dev/null 2>&1; then
  python3 tools/gerar_icone.py >/dev/null 2>&1 \
    && ok "ícone de microscópio gerado nas densidades do Android" \
    || aviso "não consegui gerar o ícone (é preciso Pillow: pip install pillow)"
fi

if ! grep -q "onnxruntime" android/app/proguard-rules.pro 2>/dev/null; then
  echo "-keep class ai.onnxruntime.** { *; }" >> android/app/proguard-rules.pro
  ok "regra do proguard adicionada"
else
  ok "regra do proguard já presente"
fi

# Assinatura própria, se as variáveis estiverem definidas.
if [ -n "${KEYSTORE_ARQUIVO:-}" ]; then
  cat > android/key.properties <<PROPS
storeFile=$KEYSTORE_ARQUIVO
storePassword=${KEYSTORE_SENHA:-}
keyAlias=${KEY_ALIAS:-}
keyPassword=${KEY_SENHA:-}
PROPS
  ok "android/key.properties escrito"
  aviso "confirme que android/app/build.gradle lê esse arquivo no buildType release"
else
  [ "$MODO" = "release" ] && aviso \
    "sem keystore configurado: o APK sairá assinado com a chave de depuração, válido só para instalação manual"
fi

# ---------------------------------------------------------------------------
titulo "Compilando"

if [ "$LIMPAR" -eq 1 ]; then
  flutter clean >/dev/null
  ok "artefatos anteriores removidos"
fi

flutter pub get
ok "dependências resolvidas"

if [ "$MODO" = "debug" ]; then
  flutter build apk --debug
  ARTEFATOS=(build/app/outputs/flutter-apk/app-debug.apk)
elif [ "$UNIVERSAL" -eq 1 ]; then
  flutter build apk --release
  ARTEFATOS=(build/app/outputs/flutter-apk/app-release.apk)
else
  flutter build apk --release --split-per-abi
  ARTEFATOS=(build/app/outputs/flutter-apk/app-*-release.apk)
fi

# ---------------------------------------------------------------------------
titulo "Publicando os artefatos"

mkdir -p "$DESTINO"
CARIMBO=$(date +%Y%m%d)
for caminho in "${ARTEFATOS[@]}"; do
  [ -f "$caminho" ] || continue
  base=$(basename "$caminho" .apk)
  alvo="$DESTINO/citopred-${base#app-}-${CARIMBO}.apk"
  cp "$caminho" "$alvo"
  tamanho=$(du -h "$alvo" | cut -f1)
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$alvo" | awk '{print $1}' > "$alvo.sha256"
  fi
  ok "$alvo  ($tamanho)"
done

# ---------------------------------------------------------------------------
titulo "Verificando que o aplicativo é mesmo offline"

AAPT=$(find "${ANDROID_HOME:-$HOME/Android/Sdk}" -name aapt2 -type f 2>/dev/null | sort | tail -n1 || true)
PRIMEIRO=$(ls "$DESTINO"/*"$CARIMBO".apk 2>/dev/null | head -n1 || true)

if [ -n "$AAPT" ] && [ -n "$PRIMEIRO" ]; then
  PERMISSOES=$("$AAPT" dump permissions "$PRIMEIRO" 2>/dev/null || true)
  if echo "$PERMISSOES" | grep -q "android.permission.INTERNET"; then
    aviso "o APK declara a permissão INTERNET: investigue antes de distribuir"
  else
    ok "nenhuma permissão de INTERNET declarada: a inferência é local por construção"
  fi
else
  aviso "aapt2 não encontrado; pulei a conferência de permissões"
fi

printf '\n\033[1;32mPronto.\033[0m Instale com:  adb install -r %s\n\n' \
  "${PRIMEIRO:-$DESTINO/<arquivo>.apk}"
