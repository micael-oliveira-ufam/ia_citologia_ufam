#!/usr/bin/env bash
#
# Gera o APK do aplicativo Colo do Útero com IA (pacientes e profissionais).
#
#   ./build_apk.sh                  APKs por arquitetura (menores, recomendado)
#   ./build_apk.sh --universal      APK único, roda em qualquer aparelho
#   ./build_apk.sh --debug          build de depuração, para testar rápido
#   ./build_apk.sh --limpar         apaga artefatos antes de compilar
#
# O modelo ONNX da citologia NÃO vai no APK: o aplicativo o baixa do GitHub
# Pages na primeira vez que o profissional pede (precisa de internet nessa vez).
#
# Assinatura de release (opcional): exporte as quatro variáveis abaixo e o
# script escreve o android/key.properties para você.
#   KEYSTORE_ARQUIVO  KEYSTORE_SENHA  KEY_ALIAS  KEY_SENHA
#
set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RAIZ"

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
    -h|--help)   sed -n '2,24p' "$0"; exit 0 ;;
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

if ! flutter doctor 2>/dev/null | grep -q "Android toolchain"; then
  aviso "flutter doctor não confirmou o Android toolchain; a compilação pode falhar"
fi

if [ -z "${JAVA_HOME:-}" ] && ! command -v java >/dev/null 2>&1; then
  erro "Java não encontrado. Instale o JDK 17 e defina JAVA_HOME."
fi
ok "Java disponível"

# ---------------------------------------------------------------------------
titulo "Modelo de citologia"

# O modelo NÃO vai dentro do APK: o aplicativo o baixa do GitHub Pages na
# primeira vez (veja docs/model/ e tools/publicar_modelo.py).
ok "o modelo é baixado pelo aplicativo (URL_BASE_MODELO em lib/config.dart)"
if [ ! -f "../docs/model/manifesto.json" ]; then
  aviso "docs/model/manifesto.json não existe: ainda não publicaram o modelo; o app mostrará 'modelo ainda não publicado'."
  aviso "Para publicar: python3 tools/export_onnx.py ... && python3 tools/publicar_modelo.py --onnx ARQUIVO.onnx"
fi

# ---------------------------------------------------------------------------
titulo "Preparando o projeto Android"

if [ ! -f "android/settings.gradle" ] && [ ! -f "android/settings.gradle.kts" ]; then
  echo "  Gerando o esqueleto nativo…"
  # flutter create não sobrescreve arquivos existentes sem --overwrite, mas o
  # lib/ e o pubspec.yaml são o coração do projeto: copiamos por precaução.
  RESGATE=$(mktemp -d)
  cp -r lib test pubspec.yaml "$RESGATE"/ 2>/dev/null || true
  PROGUARD_SALVO=""
  [ -f "android/app/proguard-rules.pro" ] && \
    PROGUARD_SALVO=$(cat android/app/proguard-rules.pro)

  flutter create --platforms=android,ios --project-name colo_ia \
    --org br.ufam.fcf . >/dev/null

  cp -r "$RESGATE"/lib . 2>/dev/null || true
  cp -r "$RESGATE"/test . 2>/dev/null || true
  cp "$RESGATE"/pubspec.yaml . 2>/dev/null || true
  rm -rf "$RESGATE"
  mkdir -p android/app
  [ -n "$PROGUARD_SALVO" ] && printf '%s\n' "$PROGUARD_SALVO" > android/app/proguard-rules.pro
  ok "esqueletos android/ e ios/ criados (applicationId br.ufam.fcf.colo_ia)"
else
  ok "android/ já existe"
fi

# Nome exibido do aplicativo e textos de permissão do iOS (câmera e galeria).
if python3 tools/ajustar_plataformas.py; then
  ok "nome do aplicativo e permissões do iOS ajustados"
else
  aviso "não consegui ajustar nome/permissões (tools/ajustar_plataformas.py); confira o ios/Runner/Info.plist"
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
  aviso "não consegui ajustar o minSdk automaticamente — confira $GRADLE_APP (precisa ser >= $MIN_SDK)"
fi
rm -f "${GRADLE_APP}.bak"

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
  alvo="$DESTINO/colo-ia-${base#app-}-${CARIMBO}.apk"
  cp "$caminho" "$alvo"
  tamanho=$(du -h "$alvo" | cut -f1)
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$alvo" | awk '{print $1}' > "$alvo.sha256"
  fi
  ok "$alvo  ($tamanho)"
done

# ---------------------------------------------------------------------------
titulo "Conferindo a permissão de internet"

AAPT=$(find "${ANDROID_HOME:-$HOME/Android/Sdk}" -name aapt2 -type f 2>/dev/null | sort | tail -n1 || true)
PRIMEIRO=$(ls "$DESTINO"/*"$CARIMBO".apk 2>/dev/null | head -n1 || true)

if [ -n "$AAPT" ] && [ -n "$PRIMEIRO" ]; then
  PERMISSOES=$("$AAPT" dump permissions "$PRIMEIRO" 2>/dev/null || true)
  if echo "$PERMISSOES" | grep -q "android.permission.INTERNET"; then
    ok "o APK declara INTERNET (necessária: baixar o modelo e acessar o catálogo)"
  else
    aviso "o APK NÃO declara INTERNET: o download do modelo e o catálogo vão falhar (rode tools/ajustar_plataformas.py)"
  fi
else
  aviso "aapt2 não encontrado; pulei a conferência de permissões"
fi

printf '\n\033[1;32mPronto.\033[0m Instale com:  adb install -r %s\n\n' \
  "${PRIMEIRO:-$DESTINO/<arquivo>.apk}"
printf 'iOS: veja o README (seção "iOS"); o build exige macOS e Xcode.\n\n'
