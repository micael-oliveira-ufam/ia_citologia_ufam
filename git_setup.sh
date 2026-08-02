#!/bin/bash

# Configurações do GitHub
GITHUB_USER="micael-oliveira-ufam"
REPO_NAME="ia_citologia_ufam"

# SUBST_TOKEN: Insira o seu NOVO Personal Access Token (PAT) do GitHub entre as aspas abaixo
# ATENÇÃO: Nunca compartilhe este token publicamente.
GITHUB_TOKEN=""

echo "=========================================================================="
echo " Iniciando configuração do Git para: $REPO_NAME"
echo "=========================================================================="

# 1. Criação do arquivo descritivo (opcional, caso não exista)
if [ ! -f README.md ]; then
    echo "# ia_citologia_ufam" >> README.md
    echo "[+] Arquivo README.md criado."
fi

# 2. Inicialização do repositório local
git init
echo "[+] Repositório Git inicializado localmente."

# 3. Adição de TODOS os arquivos do diretório ao stage
git add .
echo "[+] Todos os arquivos do diretório local foram adicionados ao stage."

# 4. Primeiro commit local
git commit -m "Atualização para o modelo ConvNext e desenvolvimento do app em Flutter"

# 5. Definição da branch principal para 'main'
git branch -M main
echo "[+] Branch principal configurada como 'main'."

# 6. Configuração da URL remota injetando o Token de Autenticação de forma segura
REMOTE_URL="https://${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git"

# Remove a origin caso ela já exista de alguma tentativa anterior
git remote remove origin 2>/dev/null
git remote add origin "$REMOTE_URL"
echo "[+] URL remota configurada com credenciais de autenticação."

# 7. Upload do código para o servidor remoto do GitHub (USANDO --force)
echo "[...] Realizando push forçado para a branch main remota..."
git push -u origin main --force

if [ $? -eq 0 ]; then
    echo "=========================================================================="
    echo " Repositório configurado e todos os arquivos enviados com sucesso!"
    echo "=========================================================================="
else
    echo "=========================================================================="
    echo " Erro ao realizar o push. Verifique se o seu Token possui permissão de 'repo'."
    echo "=========================================================================="
fi
