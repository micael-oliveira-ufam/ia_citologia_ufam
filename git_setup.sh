#!/bin/bash

# ==========================================
# Configurações do GitHub
# ==========================================
GITHUB_USER="micael-oliveira-ufam"
REPO_NAME="ia_citologia_ufam"
BRANCH_NAME="main"

echo "=========================================="
echo "   ⚠️ Automação de Deploy FORÇADO"
echo "=========================================="
echo "ATENÇÃO: Este script irá sobrescrever o repositório remoto."
echo "Tudo o que estiver no GitHub e não estiver na sua máquina será APAGADO."
echo "=========================================="

# ==========================================
# Autenticação Segura
# ==========================================
read -s -p "🔑 Digite ou cole seu Personal Access Token (PAT): " GITHUB_TOKEN
echo "" 

if [ -z "$GITHUB_TOKEN" ]; then
    echo "❌ Erro: O Token não pode estar vazio."
    exit 1
fi

REMOTE_URL="https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git"
CLEAN_URL="https://github.com/${GITHUB_USER}/${REPO_NAME}.git"

# ==========================================
# Validação e Configuração Inicial
# ==========================================
if [ ! -d ".git" ]; then
    echo "🔄 Inicializando um novo repositório Git local..."
    git init
    git branch -M "$BRANCH_NAME"
fi

if git remote | grep -q "^origin$"; then
    git remote set-url origin "$REMOTE_URL" > /dev/null 2>&1
else
    git remote add origin "$REMOTE_URL" > /dev/null 2>&1
fi

trap 'git remote set-url origin "$CLEAN_URL" > /dev/null 2>&1; echo "🔒 Credenciais limpas do ambiente local."' EXIT

# ==========================================
# Adição e Commit
# ==========================================
echo "📦 Verificando arquivos modificados..."
git add .

if git status --porcelain | grep -q "^"; then
    read -p "💬 Digite a mensagem do commit (ou aperte Enter para data atual): " COMMIT_MSG
    if [ -z "$COMMIT_MSG" ]; then
        COMMIT_MSG="Substituição forçada da versão: $(date +'%Y-%m-%d %H:%M:%S')"
    fi
    echo "💾 Criando o commit..."
    git commit -m "$COMMIT_MSG"
else
    echo "✅ Nenhuma alteração nova detectada nos arquivos locais."
fi

# ==========================================
# Envio Forçado (Overwrite Remoto)
# ==========================================
echo "🚀 FORÇANDO o envio do código para o GitHub..."

# O parâmetro --force (-f) diz ao GitHub para aceitar a sua versão local 
# como a única verdade e descartar qualquer histórico divergente online.
if git push -u origin "$BRANCH_NAME" --force --quiet; then
    echo "✅ Repositório SOBSCRITO e atualizado com sucesso no GitHub!"
else
    echo "❌ Ocorreu um erro ao enviar para o GitHub."
    echo "Verifique se a sua branch principal aceita 'force push' nas configurações do GitHub."
    exit 1
fi
