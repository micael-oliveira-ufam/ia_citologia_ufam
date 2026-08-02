#!/bin/bash

# ==========================================
# Configurações do GitHub
# ==========================================
GITHUB_USER="micael-oliveira-ufam"
REPO_NAME="ia_citologia_ufam"
BRANCH_NAME="main"

# ==========================================
# Autenticação Segura
# ==========================================

# Solicita o token de forma oculta (-s) para não vazar no terminal
read -s -p "🔑 Digite ou cole seu Personal Access Token (PAT) do GitHub: " GITHUB_TOKEN
echo "" # Quebra de linha após a digitação oculta

if [ -z "$GITHUB_TOKEN" ]; then
    echo "❌ Erro: O Token não pode estar vazio."
    exit 1
fi

# Constrói a URL de autenticação segura
REMOTE_URL="https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git"

# ==========================================
# Validação e Configuração Inicial
# ==========================================

# 1. Verifica se já é um repositório git local. Se não, inicializa.
if [ ! -d ".git" ]; then
    echo "🔄 Inicializando um novo repositório Git local..."
    git init
    git branch -M "$BRANCH_NAME"
fi

# 2. Configura ou atualiza a URL do repositório remoto de forma silenciosa
# (Redirecionando a saída para /dev/null para não imprimir a URL com o token)
if git remote | grep -q "^origin$"; then
    echo "🔄 Atualizando as credenciais do repositório remoto 'origin'..."
    git remote set-url origin "$REMOTE_URL" > /dev/null 2>&1
else
    echo "➕ Adicionando o repositório remoto 'origin'..."
    git remote add origin "$REMOTE_URL" > /dev/null 2>&1
fi

# ==========================================
# Adição e Commit
# ==========================================

echo "📦 Adicionando arquivos modificados..."
git add .

# Verifica se há algo novo para "commitar"
if git status --porcelain | grep -q "^"; then
    # Pede o nome do commit; se deixar em branco, usa a data atual
    read -p "💬 Digite a mensagem do commit (ou pressione Enter para usar a data atual): " COMMIT_MSG
    
    if [ -z "$COMMIT_MSG" ]; then
        COMMIT_MSG="Atualização automática: $(date +'%Y-%m-%d %H:%M:%S')"
    fi
    
    echo "💾 Criando o commit..."
    git commit -m "$COMMIT_MSG"
else
    echo "✅ Nenhuma alteração detectada para criar um novo commit."
fi

# ==========================================
# Sincronização e Envio (Proteção de Histórico)
# ==========================================

echo "🔄 Sincronizando com o GitHub..."

# Tenta puxar alterações remotas para evitar divergências.
git pull origin "$BRANCH_NAME" --rebase || true

echo "🚀 Enviando código para o GitHub..."

# Envio padrão. Se houver um conflito estrutural, o Git irá bloquear o envio em vez de apagar o histórico.
# Redirecionamos os erros e acertos para evitar que o Git vaze a URL com o token em caso de falha.
if git push -u origin "$BRANCH_NAME" --quiet; then
    echo "✅ Repositório atualizado com sucesso no GitHub!"
else
    echo "❌ Ocorreu um erro ao enviar para o GitHub."
    echo "Verifique se o Token tem as permissões corretas (repo) ou se há conflitos manuais que precisam ser resolvidos."
    
    # Remove a URL com token do repositório remoto por segurança após a falha
    git remote set-url origin "https://github.com/${GITHUB_USER}/${REPO_NAME}.git"
    exit 1
fi

# Limpa a URL do remote após o sucesso para não deixar o token salvo nas configurações locais do .git/config
git remote set-url origin "https://github.com/${GITHUB_USER}/${REPO_NAME}.git"
echo "🔒 Credenciais limpas do ambiente local."
