#!/usr/bin/env bash
#
# Cria a chave de assinatura (keystore) do aplicativo e mostra os valores para
# cadastrar como segredos no GitHub (Settings > Secrets and variables > Actions).
#
#   ./tools/criar_keystore.sh [caminho_do_arquivo.jks]
#
# As senhas são digitadas aqui, no seu terminal, e nunca passam por nenhuma ferramenta
# de IA. GUARDE o arquivo e as senhas em local seguro (gerenciador de senhas): quem
# perder a chave não consegue mais atualizar o app já instalado nem na Play Store.
# Não envie o .jks ao repositório (o .gitignore já ignora *.jks/key.properties).
#
set -euo pipefail

ARQ="${1:-$HOME/colo-ia-release.jks}"
ALIAS="${KEY_ALIAS:-colo_ia}"

command -v keytool >/dev/null 2>&1 || { echo "keytool não encontrado: instale o JDK 17." >&2; exit 1; }
[ -e "$ARQ" ] && { echo "O arquivo $ARQ já existe; escolha outro caminho ou apague-o de propósito." >&2; exit 1; }

read -r -s -p "Senha do keystore (mín. 6 caracteres): " SENHA; echo
read -r -s -p "Repita a senha: " SENHA2; echo
[ "$SENHA" = "$SENHA2" ] || { echo "As senhas não conferem." >&2; exit 1; }
[ "${#SENHA}" -ge 6 ] || { echo "Senha curta demais." >&2; exit 1; }

keytool -genkeypair -v -keystore "$ARQ" -alias "$ALIAS" -keyalg RSA -keysize 2048 \
  -validity 10000 -storepass "$SENHA" -keypass "$SENHA" \
  -dname "CN=Colo do Utero IA, OU=FCF, O=UFAM, L=Manaus, ST=AM, C=BR"

if base64 --help 2>&1 | grep -q -- "-w"; then B64=$(base64 -w0 "$ARQ"); else B64=$(base64 < "$ARQ" | tr -d '\n'); fi

cat <<TXT

Keystore criado em: $ARQ

Cadastre estes 4 segredos em  GitHub > repositório > Settings > Secrets and variables > Actions:

  ANDROID_KEYSTORE_PASSWORD = (a senha que você acabou de digitar)
  ANDROID_KEY_PASSWORD      = (a mesma senha)
  ANDROID_KEY_ALIAS         = $ALIAS
  ANDROID_KEYSTORE_BASE64   = o conteúdo salvo em ${ARQ%.jks}.base64.txt

TXT
printf '%s' "$B64" > "${ARQ%.jks}.base64.txt"
chmod 600 "$ARQ" "${ARQ%.jks}.base64.txt"
echo "Apague ${ARQ%.jks}.base64.txt depois de colar no GitHub."
