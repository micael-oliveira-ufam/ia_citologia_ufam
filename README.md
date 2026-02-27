# 🔬 IA Citologia: Sistema de Apoio ao Diagnóstico Citológico

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Status](https://img.shields.io/badge/Status-Ativo-success)

[![Acessar Aplicação](https://img.shields.io/badge/🌐_Acessar_Aplicação_Online-FF4B4B?style=for-the-badge&logo=streamlit)](https://ia-citologia-ufam.streamlit.app/)

Plataforma online baseada em Inteligência Artificial para análise e classificação automatizada de lâminas de citologia em meio líquido, focada no rastreio precoce do câncer do colo do útero.

---

## 📖 Sobre o Projeto

O câncer do colo do útero é uma doença altamente evitável, porém continua sendo um desafio crítico na saúde pública. Esta aplicação atua como um **Sistema de Suporte à Decisão Clínica (CDSS)**, desenhado para auxiliar citopatologistas na triagem de exames preventivos.

Utilizando Redes Neurais Convolucionais (arquitetura **ResNet50**), o modelo lê imagens digitalizadas de lâminas citológicas e as categoriza segundo as diretrizes do **Sistema Bethesda**:
* **NILM** (Negativo para lesão intraepitelial ou malignidade)
* **LSIL** (Lesão intraepitelial escamosa de baixo grau)
* **HSIL** (Lesão intraepitelial escamosa de alto grau)
* **SCC** (Carcinoma de células escamosas)

## ✨ Funcionalidades Principais

* **Classificação Automatizada:** Predição rápida da categoria da lesão celular.
* **Nível de Confiança:** Exibição gráfica da probabilidade para cada uma das classes clínicas.
* **Interpretabilidade (Grad-CAM):** Geração de mapas de calor que destacam em vermelho/laranja as regiões da célula (como o núcleo alterado) que mais influenciaram a decisão da IA, eliminando o efeito "caixa preta".
* **Data Augmentation Visual:** Simulação de variações de microscopia (rotação, cor e brilho) para validar a robustez da detecção.

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python
* **Deep Learning:** PyTorch, Torchvision (ResNet50)
* **Interface Web:** Streamlit
* **Visão Computacional:** OpenCV, PIL, Albumentations
* **Visualização de Dados:** Matplotlib, NumPy

## 👨‍🔬 Equipe e Créditos

Este projeto é fruto de pesquisa acadêmica de Iniciação Científica (PIBIC) voltada ao avanço tecnológico na saúde pública.

* **Desenvolvedor:** Micael Davi Lima de Oliveira (Iniciação Científica)
* **Coordenação:** Prof. Dr. Toni Ricardo Martins
* **Instituição:** Faculdade de Ciências Farmacêuticas - Universidade Federal do Amazonas (UFAM)
* **Parceria Institucional:** Laboratório Sebastião Marinho (SEMSA)

## 🚀 Como Executar o Projeto Localmente

Caso deseje rodar o código-fonte na sua própria máquina, siga os passos abaixo:

**1. Clone o repositório:**
```bash
git clone [https://github.com/micael-oliveira-ufam/ia_citologia_ufam.git](https://github.com/micael-oliveira-ufam/ia_citologia_ufam.git)
cd NOME_DO_REPOSITORIO
