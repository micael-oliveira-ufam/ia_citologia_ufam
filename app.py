"""
Sistema de Apoio ao Diagnóstico Citológico | UFAM, IComp e SEMSA Manaus
Citologia em meio líquido, Sistema Bethesda, ConvNeXt Large + Grad-CAM.

Execução:
    streamlit run app.py

Estrutura esperada:
    app.py
    logo_ufam.png, logo-icomp.png, semsa.png
    assets/matriz_confusao.png
    assets/historico_treinamento.png        (opcional)
    exemplos/*.jpg
    .streamlit/config.toml                  (tema claro)
"""

import json
import os
import urllib.request

import albumentations as A
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torchvision import models

matplotlib.use("Agg")

# =============================================================================
# SEÇÃO 1. CONFIGURAÇÃO
# =============================================================================

st.set_page_config(
    page_title="IA Citologia UFAM & SEMSA | Citopatologia com Inteligência Artificial",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
EXEMPLOS_DIR = os.path.join(BASE_DIR, "exemplos")

ANO_DESENVOLVIMENTO = "2026"

MODEL_FILE = "convnext_liquid_based_citology_IARC_digital_atlas_01_08_26.pt"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILE)
MODEL_URL = (
    "https://github.com/micael-oliveira-ufam/ia_citologia_ufam/releases/download/"
    f"v.1.1/{MODEL_FILE}"
)

LOGO_UFAM = os.path.join(BASE_DIR, "logo_ufam.png")
LOGO_ICOMP = os.path.join(BASE_DIR, "logo-icomp.png")
LOGO_SEMSA = os.path.join(BASE_DIR, "semsa.png")
FIG_MATRIZ = os.path.join(ASSETS_DIR, "matriz_confusao.png")
FIG_HISTORICO = os.path.join(ASSETS_DIR, "historico_treinamento.png")

# -----------------------------------------------------------------------------
# PARÂMETROS COMPARTILHADOS COM O APLICATIVO ANDROID
# Ver PARAMETROS_COMPARTILHADOS.md. Os dois precisam classificar a mesma imagem
# do mesmo jeito; mexer aqui sem mexer no Flutter faz os resultados divergirem.
# -----------------------------------------------------------------------------

# Ordem das classes: a mesma do treinamento. As pastas do ImageFolder
# (Carcinoma / HSIL / LSIL / Normal) são ordenadas alfabeticamente, e é essa a
# ordem dos eixos da matriz de confusão. Alterar esta lista inverte
# diagnósticos sem gerar erro visível.
CLASSES = ["SCC", "HSIL", "LSIL", "NILM"]

# Pré-processamento de inferência (sem os aumentos de treino).
LADO_ENTRADA = 224
IMAGENET_MEDIA = (0.485, 0.456, 0.406)
IMAGENET_DESVIO = (0.229, 0.224, 0.225)

# Limiares de alerta, espelhados em lib/servicos/classificador.dart.
LIMIAR_CONFIANCA = 0.60
LIMIAR_CONCORDANCIA = 0.75

CLASS_INFO = {
    "NILM": {
        "nome": "Negativo para lesão intraepitelial ou malignidade",
        "cor": "#2E7D62",
        "gravidade": 0,
        "resumo": "Sem alterações sugestivas de lesão precursora ou câncer.",
    },
    "LSIL": {
        "nome": "Lesão intraepitelial escamosa de baixo grau",
        "cor": "#C08A1E",
        "gravidade": 1,
        "resumo": "Alterações associadas ao HPV. Maioria regride, mas exige seguimento.",
    },
    "HSIL": {
        "nome": "Lesão intraepitelial escamosa de alto grau",
        "cor": "#D9642E",
        "gravidade": 2,
        "resumo": "Lesão precursora com risco relevante de progressão.",
    },
    "SCC": {
        "nome": "Carcinoma de células escamosas",
        "cor": "#B23A3A",
        "gravidade": 3,
        "resumo": "Achados compatíveis com carcinoma invasor.",
    },
}

ORDEM_GRAVIDADE = ["NILM", "LSIL", "HSIL", "SCC"]

# -----------------------------------------------------------------------------
# CONTEÚDO DIDÁTICO SOBRE O SISTEMA BETHESDA
# -----------------------------------------------------------------------------
BETHESDA_DETALHE = {
    "NILM": {
        "traducao": "Negative for Intraepithelial Lesion or Malignancy",
        "o_que_e": (
            "Categoria de normalidade. O material é adequado e não há células com "
            "alterações sugestivas de lesão precursora ou de câncer. Não significa "
            "ausência de qualquer achado: infecções, alterações reativas por "
            "inflamação, atrofia ou uso de DIU são relatadas dentro de NILM."
        ),
        "morfologia": (
            "Células escamosas superficiais e intermediárias com núcleos pequenos, "
            "regulares e cromatina fina. Relação núcleo/citoplasma baixa. Células "
            "endocervicais colunares preservadas, em paliçada ou favo de mel."
        ),
        "conduta": (
            "Retorno ao rastreio de rotina, no intervalo previsto pelo programa. "
            "Achados infecciosos ou reativos podem motivar tratamento próprio, sem "
            "mudar a conduta de rastreio."
        ),
        "frequencia": "É o resultado da grande maioria dos exames de rastreio.",
    },
    "LSIL": {
        "traducao": "Low-grade Squamous Intraepithelial Lesion",
        "o_que_e": (
            "Reúne o efeito citopático do HPV (coilocitose) e a neoplasia "
            "intraepitelial cervical de grau 1 (NIC 1). Representa infecção "
            "produtiva pelo vírus, e não ainda uma lesão com autonomia "
            "proliferativa."
        ),
        "morfologia": (
            "Células escamosas maduras com núcleos aumentados, mais de três vezes o "
            "núcleo de uma célula intermediária normal, hipercromáticos e de "
            "contorno irregular. O achado mais característico é o coilócito: halo "
            "perinuclear amplo e bem delimitado, com borda citoplasmática densa."
        ),
        "conduta": (
            "Seguimento citológico ou colposcopia, conforme a idade e o protocolo "
            "vigente. A maioria das lesões regride espontaneamente em até dois "
            "anos, sobretudo em mulheres jovens."
        ),
        "frequencia": "A alteração mais comum entre os exames alterados.",
    },
    "HSIL": {
        "traducao": "High-grade Squamous Intraepithelial Lesion",
        "o_que_e": (
            "Agrupa NIC 2 e NIC 3, incluindo o carcinoma in situ. É a verdadeira "
            "lesão precursora: há proliferação de células imaturas com potencial "
            "real de progressão para carcinoma invasor se não tratada."
        ),
        "morfologia": (
            "Células menores e imaturas, isoladas ou em agregados sinciciais, com "
            "relação núcleo/citoplasma alta, com o núcleo ocupando a maior parte da "
            "célula. Cromatina grosseira e irregularmente distribuída, contornos "
            "nucleares recortados, citoplasma escasso e frequentemente basofílico."
        ),
        "conduta": (
            "Encaminhamento para colposcopia com biópsia dirigida, e tratamento "
            "excisional quando confirmada. É a categoria que o rastreio existe "
            "para encontrar."
        ),
        "frequencia": "Menos frequente que LSIL, com impacto clínico bem maior.",
    },
    "SCC": {
        "traducao": "Squamous Cell Carcinoma",
        "o_que_e": (
            "Carcinoma invasor de células escamosas: a neoplasia rompeu a membrana "
            "basal e invadiu o estroma. Não é mais lesão precursora, e sim câncer "
            "estabelecido."
        ),
        "morfologia": (
            "Acentuado pleomorfismo celular e nuclear, nucléolos evidentes, "
            "cromatina irregular e grumosa. Podem aparecer células queratinizadas "
            "de formas bizarras: fusiformes, em girino, em fibra. O fundo costuma "
            "trazer a chamada diátese tumoral: sangue, debris necróticos e "
            "material proteináceo granular."
        ),
        "conduta": (
            "Encaminhamento imediato para confirmação histopatológica e "
            "estadiamento. Prioridade máxima na fila."
        ),
        "frequencia": "Raro no rastreio organizado, e é justamente esse o objetivo.",
    },
}

TEXTO_BETHESDA_INTRO = """
O Sistema Bethesda é a linguagem padronizada para relatar exames citopatológicos
do colo do útero. Foi criado em 1988, nos Estados Unidos, e revisado em 1991,
2001 e 2014. Antes dele, cada laboratório usava a classificação de Papanicolaou
em cinco classes numéricas, que dizia pouco sobre a conduta a tomar e variava de
serviço para serviço.

A mudança central foi abandonar números e adotar categorias que se traduzem
diretamente em decisão clínica. O laudo passou a informar também se a amostra
era adequada para leitura. Uma lâmina com material escasso ou obscurecido por
sangue não é um resultado negativo, é uma lâmina que precisa ser repetida.

As quatro categorias abaixo são as que este modelo classifica. Elas formam uma
escala ordinal: da ausência de lesão até o carcinoma invasor. Essa ordem importa
na leitura dos erros do algoritmo, porque confundir categorias vizinhas tem
consequência clínica diferente de confundir categorias distantes.
"""

TEXTO_BETHESDA_ALEM = """
O Sistema Bethesda completo é mais amplo que estas quatro categorias. Um laudo
real também pode registrar:

- **ASC-US**: atipias em células escamosas de significado indeterminado, quando
  as alterações existem mas não bastam para caracterizar LSIL.
- **ASC-H**: atipias escamosas em que não se pode excluir lesão de alto grau.
- **AGC**: atipias em células glandulares, de origem endocervical ou endometrial.
- **AIS**: adenocarcinoma in situ.
- **Adenocarcinoma**: de origem endocervical, endometrial ou extrauterina.
- **Avaliação da adequabilidade**: satisfatória ou insatisfatória para análise.

Este modelo **não** classifica essas categorias: foi treinado apenas nas quatro
acima. Um campo com atipia glandular ou com material insatisfatório será
forçosamente encaixado numa das quatro classes conhecidas, com confiança que
pode ser alta e ainda assim errada. É uma limitação de escopo, não um defeito de
treinamento, e uma das razões pelas quais a leitura humana continua indispensável.
"""

# -----------------------------------------------------------------------------
# Aumentos usados NO TREINAMENTO. Na inferência: apenas Resize + Normalize.
# -----------------------------------------------------------------------------
AUMENTOS_TREINO = [
    ("Resize(224, 224)", "Padroniza a entrada da rede"),
    ("HorizontalFlip(p=0.5)", "A lâmina não tem orientação preferencial"),
    ("VerticalFlip(p=0.5)", "Idem, no eixo vertical"),
    ("RandomRotate90(p=0.5)", "Rotações múltiplas de 90°"),
    ("ColorJitter(0.2, 0.2, 0.2, hue=0.1, p=0.3)", "Variação de coloração entre lotes"),
    ("Normalize(ImageNet)", "Estatísticas do pré-treino"),
    ("ToTensorV2()", "Conversão para tensor"),
]

# -----------------------------------------------------------------------------
# VALIDAÇÃO CRUZADA 5-FOLDS, derivada da matriz de confusão global.
# Linhas = classe verdadeira, na ordem de ORDEM_GRAVIDADE.
# -----------------------------------------------------------------------------
MATRIZ_CONFUSAO = [
    #        NILM  LSIL  HSIL  SCC
    [634, 0, 3, 0],      # NILM verdadeiro
    [6, 139, 2, 0],      # LSIL verdadeiro
    [0, 1, 214, 1],      # HSIL verdadeiro
    [0, 0, 7, 89],       # SCC verdadeiro
]

METRICAS = {
    "protocolo": "Validação cruzada estratificada, 5 folds",
    "conjunto": "IARC Digital Atlas e Liquid Based Cytology Pap Smear (Mendeley Data)",
    "n_imagens": 1096,
    "acertos": 1076,
    "acuracia": 0.9818,
    "ic95": (0.9720, 0.9882),
    "f1_macro": 0.9717,
    "precisao_macro": 0.9798,
    "recall_macro": 0.9647,
    "kappa": 0.9694,
    "por_classe": {
        "NILM": {"precisao": 0.9906, "recall": 0.9953, "f1": 0.9930, "suporte": 637},
        "LSIL": {"precisao": 0.9929, "recall": 0.9456, "f1": 0.9686, "suporte": 147},
        "HSIL": {"precisao": 0.9469, "recall": 0.9907, "f1": 0.9683, "suporte": 216},
        "SCC": {"precisao": 0.9889, "recall": 0.9271, "f1": 0.9570, "suporte": 96},
    },
}

# HTML puro: markdown NÃO é interpretado dentro de blocos com unsafe_allow_html,
# por isso o negrito usa <strong> em vez de asteriscos.
AVISO_ETICO_HTML = (
    "Ferramenta acadêmica em desenvolvimento. O modelo foi treinado "
    "<strong>exclusivamente com acervos públicos de imagens</strong>. "
    "nenhuma amostra de paciente do serviço foi utilizada. Por isso o projeto "
    "<strong>ainda não possui aprovação de Comitê de Ética em Pesquisa "
    "(CEP/CONEP)</strong> e <strong>não pode ser usado para decisão clínica, "
    "laudo ou triagem assistencial</strong>."
)

# -----------------------------------------------------------------------------
# EQUIPE E PARCERIAS
# -----------------------------------------------------------------------------
EQUIPE = [
    {
        "nome": "Micael Davi Lima de Oliveira",
        "papel": "Desenvolvimento do modelo e das aplicações",
        "vinculo": "Iniciação Científica, Faculdade de Ciências Farmacêuticas, UFAM",
    },
    {
        "nome": "Prof. Dr. Toni Ricardo Martins",
        "papel": "Coordenação e orientação",
        "vinculo": "Faculdade de Ciências Farmacêuticas, UFAM",
    },
    {
        "nome": "Profa. Dra. Fabíola Nakamura",
        "papel": "Pesquisadora parceira em computação",
        "vinculo": "Instituto de Computação (IComp), UFAM",
    },
    {
        "nome": "Prof. Dr. Felipe Gomes de Oliveira",
        "papel": "Pesquisador parceiro em computação",
        "vinculo": "Instituto de Computação (IComp), UFAM",
    },
    {
        "nome": "Dra. Ivanete",
        "papel": "Especialista clínica e citopatologista",
        "vinculo": "SEMSA Manaus",
    },
    {
        "nome": "Dra. Carol",
        "papel": "Especialista clínica e citopatologista",
        "vinculo": "SEMSA Manaus",
    },
]

INSTITUICOES = [
    {
        "sigla": "UFAM / FCF",
        "nome": "Faculdade de Ciências Farmacêuticas, Universidade Federal do Amazonas",
        "papel": "Instituição executora. Concepção, treinamento e validação do modelo.",
    },
    {
        "sigla": "UFAM / IComp",
        "nome": "Instituto de Computação, Universidade Federal do Amazonas",
        "papel": (
            "Parceria em visão computacional e aprendizado profundo, com os "
            "professores pesquisadores Dra. Fabíola Nakamura e Dr. Felipe Gomes "
            "de Oliveira."
        ),
    },
    {
        "sigla": "SEMSA Manaus",
        "nome": "Secretaria Municipal de Saúde de Manaus, Laboratório Sebastião Marinho",
        "papel": (
            "Parceria institucional de validação clínica e citopatológica, "
            "promovendo suporte prático e validação independente do catálogo "
            "digital de lâminas. Especialistas clínicas e citopatologistas "
            "envolvidas: Dra. Ivanete e Dra. Carol."
        ),
    },
]

# -----------------------------------------------------------------------------
# IMAGENS DE TESTE. O campo "referencia" traz o gabarito do atlas.
# -----------------------------------------------------------------------------
EXEMPLOS = [
    {
        "arquivo": "cyto5940.jpg",
        "titulo": "Ectocérvice normal",
        "referencia": "NILM",
        "descricao": (
            "Ectocérvice normal: células escamosas intermediárias e superficiais, "
            "basofílicas ou eosinofílicas. Presença de alguns polimorfonucleares. "
            "(obj. 10x)"
        ),
    },
    {
        "arquivo": "cyt14686a.jpg",
        "titulo": "Carcinoma invasivo",
        "referencia": "SCC",
        "descricao": (
            "Carcinoma de células escamosas invasivo: agrupamento de células "
            "malignas pleomórficas, predominantemente pouco diferenciadas, e "
            "células isoladas mais diferenciadas, necróticas ou queratinizadas, "
            "com formas anômalas (elipses). Inflamação, sangue e necrose ao fundo. "
            "(obj. 20x)"
        ),
    },
    {
        "arquivo": "cyto5950.jpg",
        "titulo": "Campo adicional 1",
        "referencia": None,
        "descricao": "Escamosas maduras, núcleos pequenos, flora bacilar.",
    },
    {
        "arquivo": "cyto2870.jpg",
        "titulo": "Campo adicional 2",
        "referencia": None,
        "descricao": "Colunares endocervicais em paliçada, fundo inflamatório.",
    },
    {
        "arquivo": "cyt10131a.jpg",
        "titulo": "Campo adicional 3",
        "referencia": None,
        "descricao": "Halos perinucleares indicados por setas, exsudato abundante.",
    },
    {
        "arquivo": "cyt16243.jpg",
        "titulo": "Campo adicional 4",
        "referencia": None,
        "descricao": "Agrupamentos densos, alta relação núcleo/citoplasma.",
    },
]

# -----------------------------------------------------------------------------
# CONJUNTOS DE DADOS DE TREINAMENTO
# -----------------------------------------------------------------------------
DATASETS = [
    {
        "nome": "IARC Digital Atlas of Cervical Cytology",
        "sigla": "IARC",
        "instituicao": (
            "Agência Internacional de Pesquisa em Câncer, vinculada à Organização "
            "Mundial da Saúde"
        ),
        "url": "https://screening.iarc.fr/atlascyto.php",
        "descricao": (
            "Atlas digital de referência em citologia cervical, com campos "
            "anotados por citopatologistas e usado internacionalmente para ensino."
        ),
        "referencia": None,
    },
    {
        "nome": (
            "Liquid based cytology Pap smear images for multi-class diagnosis "
            "of cervical cancer"
        ),
        "sigla": "Mendeley Data",
        "instituicao": "Institute of Advanced Study in Science and Technology, Índia",
        "url": "https://data.mendeley.com/datasets/zddtpgzv63/2",
        "descricao": (
            "Conjunto de imagens de citologia em meio líquido preparadas e "
            "anotadas para classificação multiclasse de lesões pré-cancerosas e "
            "de câncer cervical."
        ),
        "referencia": (
            "HUSSAIN, Elima; MAHANTA, Lipi B.; BORAH, Himakshi; DAS, Chandana Ray. "
            "Liquid based-cytology Pap smear dataset for automated multi-class "
            "diagnosis of pre-cancerous and cervical cancer lesions. *Data in "
            "Brief*, v. 30, p. 105589, jun. 2020. "
            "DOI: https://doi.org/10.1016/j.dib.2020.105589"
        ),
    },
]

# -----------------------------------------------------------------------------
# FERRAMENTAS DE APOIO AO DESENVOLVIMENTO
# Transparência sobre o processo: o código foi escrito com apoio de assistente
# de IA generativa, sob revisão e responsabilidade da equipe.
# -----------------------------------------------------------------------------
FERRAMENTAS_APOIO = (
    "O código desta plataforma e do aplicativo Android foi desenvolvido com "
    "apoio do modelo de IA generativa **Claude Opus 5**, da Anthropic, usado "
    "como assistente de programação. A concepção científica, a escolha da "
    "arquitetura, o treinamento, a curadoria dos dados, a validação e a revisão "
    "de todo o código são de responsabilidade da equipe do projeto."
)

MENUS = [
    "Análise",
    "Sistema Bethesda",
    "Validação do algoritmo",
    "Imagens de teste",
    "Equipe e parcerias",
    "Dados e créditos",
]


# -----------------------------------------------------------------------------
# DESCOBERTA EM BUSCADORES
# O Streamlit monta o <head> sozinho e não expõe meta tags, então elas são
# injetadas no documento a partir do próprio corpo da página.
# -----------------------------------------------------------------------------
SEO_TITULO = (
    "IA Citologia UFAM & SEMSA | Citopatologia com Inteligência Artificial"
)
SEO_DESCRICAO = (
    "Plataforma de apoio ao diagnóstico citológico em meio líquido pelo Sistema "
    "Bethesda com inteligência artificial. Projeto da Faculdade de Ciências "
    "Farmacêuticas e do Instituto de Computação da UFAM em parceria com a SEMSA "
    "Manaus. Classificação de NILM, LSIL, HSIL e carcinoma escamoso com ConvNeXt "
    "e mapa de atenção Grad-CAM."
)
SEO_PALAVRAS_CHAVE = [
    "IA Citologia UFAM & SEMSA",
    "IA Citologia UFAM",
    "Citopatologia SEMSA",
    "Citopatologia IA",
    "Citopatologia UFAM",
    "Citopatologia FCF",
    "Citopatologia ICOMP",
    "citologia em meio líquido",
    "citologia em meio líquido inteligência artificial",
    "Sistema Bethesda",
    "rastreio câncer do colo do útero",
    "câncer cervical Amazonas",
    "Papanicolaou inteligência artificial",
    "ConvNeXt citologia",
    "Grad-CAM citopatologia",
    "IARC Digital Atlas",
    "deep learning citopatologia",
    "UFAM Faculdade de Ciências Farmacêuticas",
    "Instituto de Computação UFAM",
    "SEMSA Manaus",
    "Laboratório Sebastião Marinho",
    "saúde digital Amazonas",
    "CitoPred",
]


def injetar_metadados():
    """Insere título, descrição e dados estruturados no <head> do documento."""
    palavras = ", ".join(SEO_PALAVRAS_CHAVE)
    dados_estruturados = json.dumps({
        "@context": "https://schema.org",
        "@type": "WebApplication",
        "name": SEO_TITULO,
        "alternateName": ["IA Citologia UFAM", "CitoPred"],
        "description": SEO_DESCRICAO,
        "applicationCategory": "HealthApplication",
        "inLanguage": "pt-BR",
        "keywords": palavras,
        "isAccessibleForFree": True,
        "creator": [
            {"@type": "Person", "name": p["nome"], "affiliation": p["vinculo"]}
            for p in EQUIPE
        ],
        "publisher": {
            "@type": "CollegeOrUniversity",
            "name": "Universidade Federal do Amazonas",
            "department": [
                "Faculdade de Ciências Farmacêuticas",
                "Instituto de Computação",
            ],
        },
        "about": [
            "Citopatologia", "Citologia em meio líquido", "Sistema Bethesda",
            "Rastreio do câncer do colo do útero", "Inteligência artificial",
        ],
    }, ensure_ascii=False)

    st.markdown(
        f"""
<script type="application/ld+json">{dados_estruturados}</script>
<script>
(function () {{
  var doc = window.parent ? window.parent.document : document;
  function meta(atributo, chave, valor) {{
    var el = doc.querySelector('meta[' + atributo + '="' + chave + '"]');
    if (!el) {{
      el = doc.createElement('meta');
      el.setAttribute(atributo, chave);
      doc.head.appendChild(el);
    }}
    el.setAttribute('content', valor);
  }}
  doc.title = {json.dumps(SEO_TITULO)};
  doc.documentElement.lang = 'pt-BR';
  meta('name', 'description', {json.dumps(SEO_DESCRICAO)});
  meta('name', 'keywords', {json.dumps(palavras)});
  meta('name', 'author', 'Universidade Federal do Amazonas');
  meta('name', 'robots', 'index, follow');
  meta('property', 'og:title', {json.dumps(SEO_TITULO)});
  meta('property', 'og:description', {json.dumps(SEO_DESCRICAO)});
  meta('property', 'og:type', 'website');
  meta('property', 'og:locale', 'pt_BR');
  meta('name', 'twitter:card', 'summary_large_image');
  meta('name', 'twitter:title', {json.dumps(SEO_TITULO)});
  meta('name', 'twitter:description', {json.dumps(SEO_DESCRICAO)});
}})();
</script>
""",
        unsafe_allow_html=True,
    )


# =============================================================================
# SEÇÃO 2. IDENTIDADE VISUAL (TEMA CLARO)
# =============================================================================

def injetar_css():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;600&display=swap');

:root {
  --porcelana: #F4F7F9;
  --superficie: #FFFFFF;
  --tinta: #14232F;
  --tinta-fraca: #566B7A;
  --linha: #DCE5EB;
  --teal: #0E6B7B;
  --teal-claro: #E4F1F3;
}

.stApp { background: var(--porcelana); }

html, body, [class*="css"], .stMarkdown, .stText, .stDataFrame {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  color: var(--tinta);
}

/* padding-top generoso: com pouco espaço, a linha "UFAM · IComp · SEMSA"
   estourava para cima do contêiner e aparecia cortada ao meio. */
.block-container { padding-top: 3.2rem; padding-bottom: 1rem; max-width: 1400px; }
.block-container > div { overflow: visible; }

/* A navegação por barra lateral foi removida. */
section[data-testid="stSidebar"] { display: none; }

h1, h2, h3, h4 { color: var(--tinta); letter-spacing: -0.015em; }
h1 {
  font-size: 1.95rem;
  font-weight: 800;
  margin-bottom: .2rem;
  padding-top: 0 !important;   /* o padrão do Streamlit empurra o título */
  line-height: 1.2;
}

.cartao {
  background: var(--superficie);
  border: 1px solid var(--linha);
  border-radius: 14px;
  padding: 1.15rem 1.35rem;
  height: 100%;
  box-shadow: 0 1px 2px rgba(20,35,47,.04);
}

.sobrancelha {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .72rem;
  letter-spacing: .15em;
  text-transform: uppercase;
  color: var(--teal);
  margin: 0 0 .35rem 0;
  padding-top: .15rem;
  line-height: 1.6;          /* sem isso, as maiúsculas ficam rentes ao topo */
  display: block;
  overflow: visible;
}

.subtitulo { color: var(--tinta-fraca); font-size: 1.02rem; margin-top: -.2rem; }

/* --- Créditos no cabeçalho --- */
.creditos {
  display: flex;
  flex-wrap: wrap;
  gap: 0;
  background: var(--superficie);
  border: 1px solid var(--linha);
  border-radius: 12px;
  overflow: hidden;
  margin: .6rem 0 .9rem 0;
}
.credito-bloco {
  flex: 1 1 220px;
  padding: .85rem 1.1rem;
  border-right: 1px solid var(--linha);
}
.credito-bloco:last-child { border-right: none; }
.credito-titulo {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .66rem;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--teal);
  font-weight: 600;
  margin-bottom: .35rem;
}
.credito-nome { font-size: .88rem; font-weight: 600; line-height: 1.45; }
.credito-sub { font-size: .78rem; color: var(--tinta-fraca); line-height: 1.4; }

.faixa-etica {
  background: #FFF6E8;
  border: 1px solid #F0D6A8;
  border-left: 4px solid #C08A1E;
  border-radius: 10px;
  padding: .8rem 1.05rem;
  font-size: .89rem;
  color: #5A4415;
  line-height: 1.55;
}
.faixa-etica strong { color: #4A3510; font-weight: 700; }

.numero {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 1.7rem;
  font-weight: 600;
  line-height: 1.15;
}
.rotulo-numero {
  font-size: .73rem;
  color: var(--tinta-fraca);
  text-transform: uppercase;
  letter-spacing: .09em;
  font-weight: 600;
}
.nota-numero { font-size: .74rem; color: var(--tinta-fraca); margin-top: .3rem; }

/* --- Régua de gravidade do Sistema Bethesda --- */
.regua { display: flex; gap: 6px; margin: .5rem 0 .3rem 0; }
.degrau {
  flex: 1;
  border-radius: 9px;
  padding: .55rem .55rem .5rem .6rem;
  border: 1px solid var(--linha);
  background: var(--superficie);
  opacity: .45;
}
.degrau.ativo { opacity: 1; border-width: 2px; }
.degrau .sigla { font-weight: 800; font-size: .95rem; }
.degrau .desc { font-size: .69rem; color: var(--tinta-fraca); line-height: 1.3; }

.pessoa {
  border-left: 3px solid var(--teal-claro);
  padding: .1rem 0 .1rem .8rem;
  margin-bottom: .9rem;
}
.pessoa .nome { font-weight: 700; font-size: .97rem; }
.pessoa .papel { color: var(--teal); font-size: .86rem; font-weight: 600; }
.pessoa .vinculo { color: var(--tinta-fraca); font-size: .83rem; }

.etiqueta {
  display: inline-block;
  background: var(--teal-claro);
  color: var(--teal);
  font-size: .73rem;
  font-weight: 700;
  letter-spacing: .05em;
  padding: .18rem .6rem;
  border-radius: 6px;
}

.stButton > button {
  border-radius: 10px;
  border: 1px solid var(--linha);
  font-weight: 600;
}
.stButton > button[kind="primary"] { background: var(--teal); border-color: var(--teal); }

/* --- NAVEGAÇÃO PRINCIPAL: st.radio com aparência de abas separadas ---
   O seletor usa a classe gerada por st.container(key="navegacao"). O segundo
   seletor é uma rede de segurança para versões do Streamlit que não geram
   `st-key-*`: como esta é a única lista de opções da página, não há conflito. */
.st-key-navegacao div[role="radiogroup"],
div[data-testid="stMain"] div[role="radiogroup"] {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: .4rem;
}
.st-key-navegacao div[role="radiogroup"] > label,
div[data-testid="stMain"] div[role="radiogroup"] > label {
  background: var(--superficie);
  border: 1px solid var(--linha);
  border-radius: 11px;
  padding: .62rem 1.25rem;
  margin: 0 !important;
  cursor: pointer;
  transition: border-color .15s, background .15s, box-shadow .15s;
}
.st-key-navegacao div[role="radiogroup"] > label:hover,
div[data-testid="stMain"] div[role="radiogroup"] > label:hover {
  border-color: var(--teal);
  background: var(--teal-claro);
}
/* Esconde a bolinha do radio: queremos aparência de aba, não de formulário. */
.st-key-navegacao div[role="radiogroup"] > label > div:first-child,
div[data-testid="stMain"] div[role="radiogroup"] > label > div:first-child {
  display: none;
}
.st-key-navegacao div[role="radiogroup"] label p,
div[data-testid="stMain"] div[role="radiogroup"] label p {
  font-size: .93rem;
  font-weight: 600;
  color: var(--tinta-fraca);
  margin: 0;
  white-space: nowrap;
}
.st-key-navegacao div[role="radiogroup"] > label:has(input:checked),
div[data-testid="stMain"] div[role="radiogroup"] > label:has(input:checked) {
  background: var(--teal);
  border-color: var(--teal);
  box-shadow: 0 2px 7px rgba(14,107,123,.22);
}
.st-key-navegacao div[role="radiogroup"] > label:has(input:checked) p,
div[data-testid="stMain"] div[role="radiogroup"] > label:has(input:checked) p {
  color: #FFFFFF;
}

/* --- Rodapé --- */
.rodape {
  margin-top: 2.6rem;
  padding: 1.35rem 0 .7rem 0;
  border-top: 1px solid var(--linha);
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: .8rem;
  font-size: .8rem;
  color: var(--tinta-fraca);
  line-height: 1.6;
}
.rodape .ano { font-family: 'IBM Plex Mono', monospace; font-weight: 600; color: var(--teal); }
.rodape code { font-size: .76rem; }

hr { border-color: var(--linha); margin: 1.2rem 0; }
footer, #MainMenu { visibility: hidden; }
</style>
""",
        unsafe_allow_html=True,
    )


def figura_clara(*args, **kwargs):
    fig, ax = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor("#FFFFFF")
    for a in np.atleast_1d(ax).ravel():
        a.set_facecolor("#FFFFFF")
    return fig, ax


# =============================================================================
# SEÇÃO 3. MODELO
# =============================================================================

def modelo_disponivel():
    """O arquivo já está no diretório de trabalho e tem tamanho plausível?"""
    return os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000


@st.cache_resource(show_spinner=False)
def garantir_modelo(_url=MODEL_URL):
    """Baixa o modelo uma única vez, e só se ele não estiver no diretório.

    A função é cacheada por `st.cache_resource`, então roda no máximo uma vez
    por processo do servidor. Cliques em menu, troca de seção e qualquer outro
    rerun reaproveitam o resultado em vez de reabrir a barra de progresso.
    A checagem de `os.path.exists` vem antes de tudo: se o arquivo já foi
    baixado numa execução anterior do servidor, nada é transferido.
    """
    if modelo_disponivel():
        return MODEL_PATH

    caixa = st.container()
    with caixa:
        st.info(
            f"O modelo `{MODEL_FILE}` não está no diretório de trabalho. "
            "Baixando uma única vez. Nas próximas execuções ele é lido do disco."
        )
        barra = st.progress(0.0, text="Preparando o download…")

        def report(blocos, tam_bloco, total):
            if total > 0:
                frac = min(blocos * tam_bloco / total, 1.0)
                barra.progress(
                    frac,
                    text=f"Baixando o modelo: {frac*100:.0f}% de {total/1_048_576:.0f} MB",
                )

        tmp = MODEL_PATH + ".parcial"
        try:
            urllib.request.urlretrieve(_url, tmp, reporthook=report)
            # Só promove o arquivo definitivo depois do download completo:
            # assim uma queda de conexão não deixa um .pt truncado no disco,
            # que seria tratado como "já existe" na próxima execução.
            os.replace(tmp, MODEL_PATH)
        except Exception as erro:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise RuntimeError(
                f"Falha ao baixar o modelo de {_url}. Verifique a conexão ou "
                f"coloque o arquivo manualmente em {BASE_DIR}. Detalhe: {erro}"
            ) from erro

    caixa.empty()
    return MODEL_PATH


def _extrair_state_dict(objeto):
    sd = objeto
    if isinstance(objeto, dict) and not any(torch.is_tensor(v) for v in objeto.values()):
        for chave in ("state_dict", "model_state_dict", "model"):
            if chave in objeto and isinstance(objeto[chave], dict):
                sd = objeto[chave]
                break
    return {k.replace("module.", "", 1): v for k, v in sd.items()}


def _detectar_variante(sd):
    stem = sd.get("features.0.0.weight")
    if stem is None:
        return "convnext_large"
    largura = stem.shape[0]
    if largura == 192:
        return "convnext_large"
    if largura == 128:
        return "convnext_base"
    blocos = {int(k.split(".")[2]) for k in sd if k.startswith("features.5.")}
    return "convnext_small" if len(blocos) > 12 else "convnext_tiny"


def _cabeca_com_dropout(sd):
    """O treinamento usou classifier[2] = Sequential(Dropout, Linear)."""
    return any(k.startswith("classifier.2.1.") for k in sd)


def construir_modelo(variante, com_dropout, num_classes):
    modelo = getattr(models, variante)(weights=None)
    in_features = modelo.classifier[2].in_features
    if com_dropout:
        modelo.classifier[2] = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(in_features, num_classes)
        )
    else:
        modelo.classifier[2] = nn.Linear(in_features, num_classes)
    return modelo


@st.cache_resource(show_spinner=False)
def carregar_modelo():
    """Monta a rede a partir do checkpoint. Cacheada: roda uma vez por processo.

    Não desenha nada na tela. Avisos são devolvidos como texto para quem
    chamar decidir onde mostrar. Isso permite que a interface inteira apareça
    antes de o modelo existir.
    """
    caminho = garantir_modelo()

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        bruto = torch.load(caminho, map_location=dispositivo, weights_only=True)
    except Exception:
        bruto = torch.load(caminho, map_location=dispositivo, weights_only=False)

    sd = _extrair_state_dict(bruto)
    variante = _detectar_variante(sd)
    modelo = construir_modelo(variante, _cabeca_com_dropout(sd), len(CLASSES))

    faltando, sobrando = modelo.load_state_dict(sd, strict=False)
    aviso = None
    if faltando or sobrando:
        aviso = (
            f"O arquivo do modelo não bate exatamente com a arquitetura montada "
            f"({len(faltando)} pesos ausentes, {len(sobrando)} inesperados). "
            "Confira a variante do ConvNeXt e o número de classes."
        )

    modelo.to(dispositivo).eval()
    return modelo, dispositivo, variante, aviso


def obter_modelo():
    """Carrega o modelo sob demanda e devolve (modelo, dispositivo, variante).

    Chamada apenas quando há uma análise para rodar. Em caso de falha, mostra a
    mensagem no lugar certo da página e devolve None, de modo que o restante da interface
    continua utilizável.
    """
    try:
        modelo, dispositivo, variante, aviso = carregar_modelo()
    except Exception as erro:
        st.error(str(erro))
        return None
    if aviso:
        st.warning(aviso)
    return modelo, dispositivo, variante


def variante_conhecida():
    """Nome da arquitetura para exibir em textos, sem forçar o carregamento."""
    if st.session_state.get("variante"):
        return st.session_state["variante"]
    return "ConvNeXt Large"



def transform_inferencia():
    return A.Compose([
        A.Resize(LADO_ENTRADA, LADO_ENTRADA),   # cv2.INTER_LINEAR, como no Android
        A.Normalize(mean=IMAGENET_MEDIA, std=IMAGENET_DESVIO),
        ToTensorV2(),
    ])


def transforms_ilustrativos():
    return {
        "Giro horizontal": A.HorizontalFlip(p=1.0),
        "Giro vertical": A.VerticalFlip(p=1.0),
        "Rotação 90°": A.RandomRotate90(p=1.0),
        "Variação de cor": A.ColorJitter(0.2, 0.2, 0.2, hue=0.1, p=1.0),
    }


# =============================================================================
# SEÇÃO 4. GRAD-CAM E INFERÊNCIA
# =============================================================================

class GradCAM:
    """Grad-CAM sobre o último bloco convolucional do ConvNeXt.

    O gradiente é capturado por um hook no TENSOR de ativação, e não por
    register_full_backward_hook no módulo. As duas formas dão o mesmo resultado
    numérico, mas o hook de módulo tem um contrato traiçoeiro: qualquer valor
    devolvido substitui o grad_input, cuja forma difere do grad_output neste
    bloco, e o autograd aborta com "hook has changed the size of value". O hook
    de tensor não tem esse contrato. Esta é a mesma implementação usada pelo
    exportador ONNX do aplicativo Android, para garantir paridade.
    """

    def __init__(self, modelo, camada_alvo):
        self.modelo = modelo
        self.ativacoes = None
        self.gradientes = None
        self.alca = camada_alvo.register_forward_hook(self._ao_passar)

    def _ao_passar(self, _modulo, _entrada, saida):
        self.ativacoes = saida
        if saida.requires_grad:
            saida.register_hook(self._guardar_gradiente)
        # Devolver None é obrigatório: qualquer retorno substitui a saída.
        return None

    def _guardar_gradiente(self, gradiente):
        self.gradientes = gradiente

    def __call__(self, tensor, indice_classe):
        self.modelo.zero_grad(set_to_none=True)
        saida = self.modelo(tensor)
        saida[0, indice_classe].backward()

        if self.ativacoes is None or self.gradientes is None:
            raise RuntimeError(
                "não foi possível capturar ativação e gradiente da camada alvo"
            )

        # Sem operação in-place: o autograd ainda referencia self.ativacoes.
        pesos = self.gradientes.mean(dim=(2, 3), keepdim=True)
        mapa = F.relu((pesos * self.ativacoes).sum(dim=1)).squeeze(0)
        maximo = mapa.max()
        if maximo > 0:
            mapa = mapa / maximo
        return mapa.detach().cpu().numpy()

    def liberar(self):
        self.alca.remove()


def _oito_vistas(tensor):
    """Grupo diedral D4: as mesmas simetrias vistas no treinamento."""
    vistas = []
    for k in range(4):
        rot = torch.rot90(tensor, k, dims=(2, 3))
        vistas.append(rot)
        vistas.append(torch.flip(rot, dims=(3,)))
    return torch.cat(vistas, dim=0)


def analisar(modelo, dispositivo, imagem_pil, verificar_robustez=True):
    img = np.array(imagem_pil)
    tensor = transform_inferencia()(image=img)["image"].unsqueeze(0).to(dispositivo)

    with torch.no_grad():
        prob = F.softmax(modelo(tensor), dim=1)[0].cpu().numpy()

    idx = int(np.argmax(prob))
    prob_tta, concordancia, idx_tta = None, None, idx

    if verificar_robustez:
        with torch.no_grad():
            prob_vistas = F.softmax(modelo(_oito_vistas(tensor)), dim=1).cpu().numpy()
        prob_tta = prob_vistas.mean(axis=0)
        idx_tta = int(np.argmax(prob_tta))
        concordancia = float((prob_vistas.argmax(axis=1) == idx).mean())

    cam = GradCAM(modelo, modelo.features[-1][-1])
    try:
        mapa = cam(tensor, idx)
    finally:
        # O modelo fica em cache entre execuções: um hook esquecido aqui
        # contaminaria todas as análises seguintes.
        cam.liberar()

    mapa_grande = cv2.resize(mapa, (img.shape[1], img.shape[0]))
    colorido = cv2.cvtColor(
        cv2.applyColorMap(np.uint8(255 * mapa_grande), cv2.COLORMAP_INFERNO),
        cv2.COLOR_BGR2RGB,
    )
    sobreposto = cv2.addWeighted(img, 0.6, colorido, 0.4, 0)

    return {
        "img": img,
        "prob": prob,
        "idx": idx,
        "classe": CLASSES[idx],
        "confianca": float(prob[idx]),
        "prob_tta": prob_tta,
        "classe_tta": CLASSES[idx_tta],
        "concordancia": concordancia,
        "sobreposto": sobreposto,
    }


# =============================================================================
# SEÇÃO 5. COMPONENTES DE INTERFACE
# =============================================================================

def _logo(caminho, largura):
    if caminho and os.path.exists(caminho):
        st.image(caminho, width=largura)


def cabecalho():
    c1, c2, c3 = st.columns([1.1, 5.6, 2.6], vertical_alignment="top")
    with c1:
        _logo(LOGO_UFAM, 86)
    with c2:
        st.markdown(
            '<div class="sobrancelha">UFAM · IComp · SEMSA Manaus</div>',
            unsafe_allow_html=True,
        )
        st.markdown("# Sistema de Apoio ao Diagnóstico Citológico")
        st.markdown(
            '<p class="subtitulo">Citologia em meio líquido classificada pelo Sistema '
            'Bethesda, com ConvNeXt Large e mapa de atenção Grad-CAM.</p>',
            unsafe_allow_html=True,
        )
    with c3:
        l1, l2 = st.columns(2, vertical_alignment="center")
        with l1:
            _logo(LOGO_ICOMP, 135)
        with l2:
            _logo(LOGO_SEMSA, 125)

    # --- Créditos do projeto, no cabeçalho ------------------------------
    st.markdown(
        '<div class="creditos">'
        '<div class="credito-bloco">'
        '<div class="credito-titulo">Desenvolvimento</div>'
        '<div class="credito-nome">Micael Davi Lima de Oliveira</div>'
        '<div class="credito-sub">Iniciação Científica · FCF/UFAM</div>'
        '</div>'
        '<div class="credito-bloco">'
        '<div class="credito-titulo">Coordenação</div>'
        '<div class="credito-nome">Prof. Dr. Toni Ricardo Martins</div>'
        '<div class="credito-sub">Faculdade de Ciências Farmacêuticas · UFAM</div>'
        '</div>'
        '<div class="credito-bloco">'
        '<div class="credito-titulo">Pesquisadores parceiros · IComp/UFAM</div>'
        '<div class="credito-nome">Profa. Dra. Fabíola Nakamura<br>'
        'Prof. Dr. Felipe Gomes de Oliveira</div>'
        '</div>'
        '<div class="credito-bloco">'
        '<div class="credito-titulo">Validação clínica · SEMSA Manaus</div>'
        '<div class="credito-nome">Dra. Ivanete · Dra. Carol</div>'
        '<div class="credito-sub">Especialistas clínicas e citopatologistas</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="faixa-etica">{AVISO_ETICO_HTML}</div>', unsafe_allow_html=True
    )
    st.write("")


def rodape():
    st.markdown(
        '<div class="rodape">'
        '<div>'
        '<strong>Sistema de Apoio ao Diagnóstico Citológico</strong><br>'
        'Faculdade de Ciências Farmacêuticas e Instituto de Computação · UFAM<br>'
        'Secretaria Municipal de Saúde de Manaus · Laboratório Sebastião Marinho'
        '</div>'
        '<div style="text-align:right">'
        f'<span class="ano">© {ANO_DESENVOLVIMENTO}</span> · Projeto de pesquisa acadêmica<br>'
        f'Modelo <code>{variante_conhecida()}</code> · IARC Digital Atlas e Mendeley Data<br>'
        'Sem aprovação de CEP · Não emite laudo<br>'
        '<span style="font-size:.74rem">Código desenvolvido com apoio de '
        'Claude Opus 5 (Anthropic)</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def regua_bethesda(classe_ativa=None):
    degraus = []
    for sigla in ORDEM_GRAVIDADE:
        info = CLASS_INFO[sigla]
        ativo = "ativo" if sigla == classe_ativa else ""
        degraus.append(
            f'<div class="degrau {ativo}" style="border-color:{info["cor"]}">'
            f'<div class="sigla" style="color:{info["cor"]}">{sigla}</div>'
            f'<div class="desc">{info["nome"]}</div></div>'
        )
    st.markdown(f'<div class="regua">{"".join(degraus)}</div>', unsafe_allow_html=True)
    st.caption("Gravidade crescente da esquerda para a direita.")


def cartao_numero(coluna, rotulo, valor, nota=None, pct=True, cor=None):
    if valor is None:
        texto = "n/d"
    elif pct:
        texto = f"{valor:.2%}"
    elif isinstance(valor, float):
        texto = f"{valor:.4f}"
    else:
        texto = f"{valor:,}".replace(",", ".")
    estilo = f"color:{cor}" if cor else "color:var(--tinta)"
    linha_nota = f'<div class="nota-numero">{nota}</div>' if nota else ""
    with coluna:
        st.markdown(
            f'<div class="cartao"><div class="rotulo-numero">{rotulo}</div>'
            f'<div class="numero" style="{estilo}">{texto}</div>{linha_nota}</div>',
            unsafe_allow_html=True,
        )


def grafico_probabilidades(prob, idx):
    fig, ax = figura_clara(figsize=(6.2, 2.7))
    ordem = [CLASSES.index(s) for s in ORDEM_GRAVIDADE]
    valores = [prob[i] for i in ordem]
    rotulos = [CLASSES[i] for i in ordem]
    cores = ["#D6E3E8" if i != idx else CLASS_INFO[CLASSES[i]]["cor"] for i in ordem]

    barras = ax.barh(np.arange(len(ordem)), valores, color=cores, height=0.6)
    ax.set_yticks(np.arange(len(ordem)), rotulos, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Probabilidade", fontsize=9, color="#566B7A")
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color("#DCE5EB")
    ax.tick_params(colors="#566B7A", length=0)
    for barra, valor in zip(barras, valores):
        ax.text(valor + 0.02, barra.get_y() + barra.get_height() / 2,
                f"{valor:.1%}", va="center", fontsize=10, color="#14232F")
    fig.tight_layout()
    return fig


def matriz_confusao_figura(matriz, rotulos):
    m = np.array(matriz, dtype=float)
    normal = m / np.clip(m.sum(axis=1, keepdims=True), 1, None)
    fig, ax = figura_clara(figsize=(5.4, 4.6))
    ax.imshow(normal, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(rotulos)), rotulos, fontsize=9)
    ax.set_yticks(range(len(rotulos)), rotulos, fontsize=9)
    ax.set_xlabel("Predito pela IA", fontsize=9.5, color="#566B7A")
    ax.set_ylabel("Referência (citopatologista)", fontsize=9.5, color="#566B7A")
    for i in range(len(rotulos)):
        for j in range(len(rotulos)):
            ax.text(j, i, f"{int(m[i, j])}\n{normal[i, j]:.0%}", ha="center",
                    va="center", fontsize=9,
                    color="#14232F" if normal[i, j] < 0.55 else "#FFFFFF")
    ax.tick_params(length=0, colors="#566B7A")
    for lado in ax.spines.values():
        lado.set_visible(False)
    fig.tight_layout()
    return fig


def painel_aumentos(img):
    itens = list(transforms_ilustrativos().items())
    fig, eixos = figura_clara(1, len(itens), figsize=(3.0 * len(itens), 3.2))
    for eixo, (nome, transformacao) in zip(np.atleast_1d(eixos), itens):
        eixo.imshow(transformacao(image=img)["image"])
        eixo.set_title(nome, fontsize=10, color="#14232F")
        eixo.axis("off")
    fig.tight_layout()
    return fig


# =============================================================================
# SEÇÃO 6. ESTADO E NAVEGAÇÃO
# =============================================================================

def caminho_exemplo(arquivo):
    for pasta in (EXEMPLOS_DIR, BASE_DIR):
        caminho = os.path.join(pasta, arquivo)
        if os.path.exists(caminho):
            return caminho
    return None


def carregar_exemplo(ex):
    """Callback do botão 'Analisar' das imagens de teste.

    Callbacks rodam ANTES do próximo ciclo de renderização, então é aqui que dá
    para trocar o menu selecionado. É isso que faz a navegação saltar para a
    seção Análise com a inferência já disparada. Fazer o mesmo no corpo da
    página, como antes, não funcionava: o widget de navegação já havia sido
    desenhado quando o clique era processado.
    """
    caminho = caminho_exemplo(ex["arquivo"])
    if caminho is None:
        st.session_state["erro_exemplo"] = ex["arquivo"]
        return
    st.session_state["imagem"] = Image.open(caminho).convert("RGB")
    st.session_state["origem"] = ex["arquivo"]
    st.session_state["referencia"] = ex["referencia"]
    st.session_state["descricao"] = ex["descricao"]
    st.session_state["chave_upload"] = None
    st.session_state["rodar"] = True          # a análise dispara sozinha
    st.session_state["menu"] = "Análise"      # e a navegação salta para lá


def limpar_imagem():
    for chave in ("imagem", "origem", "referencia", "descricao", "rodar", "chave_upload"):
        st.session_state.pop(chave, None)


def navegacao():
    # st.container(key=...) gera a classe CSS `st-key-navegacao` em volta do
    # bloco. Envolver com st.markdown('<div>') não funcionaria: o Streamlit
    # fecha a div dentro do próprio contêiner de markdown, e o rádio ficaria
    # fora do seletor.
    with st.container(key="navegacao"):
        st.radio(
            "Navegação principal",
            MENUS,
            key="menu",
            horizontal=True,
            label_visibility="collapsed",
        )
    st.write("")


# =============================================================================
# SEÇÃO 7. ANÁLISE
# =============================================================================

def secao_analise():
    st.markdown("### Análise de campo citológico")

    # --- Entrada de dados, agora na própria área principal ---------------
    col_env, col_op = st.columns([1.6, 1])
    with col_env:
        enviado = st.file_uploader(
            "Carregue a lâmina digitalizada",
            type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
            help="Um campo digitalizado por vez. Não envie imagens de pacientes.",
        )
        if enviado is not None:
            chave = f"{enviado.name}:{enviado.size}"
            if st.session_state.get("chave_upload") != chave:
                st.session_state["chave_upload"] = chave
                st.session_state["imagem"] = Image.open(enviado).convert("RGB")
                st.session_state["origem"] = enviado.name
                st.session_state["referencia"] = None
                st.session_state["descricao"] = None
                st.session_state["rodar"] = False
    with col_op:
        st.checkbox(
            "Verificar robustez nas oito orientações",
            value=True,
            key="robustez",
            help="Analisa o campo girado e espelhado. Revela se a rede muda de "
                 "opinião só por causa da orientação. Deixa a análise mais lenta.",
        )
        st.caption(
            "Sem uma imagem própria? A seção **Imagens de teste** traz campos do "
            "IARC prontos para analisar."
        )
        if modelo_disponivel():
            tamanho = os.path.getsize(MODEL_PATH) / 1_048_576
            st.caption(f"Modelo local pronto · {tamanho:.0f} MB no diretório de trabalho.")
        else:
            st.caption(
                "O modelo ainda não está no diretório. Ele será baixado uma única "
                "vez, na primeira análise."
            )

    imagem = st.session_state.get("imagem")
    if imagem is None:
        st.markdown("---")
        tela_de_boas_vindas()
        return

    st.markdown("---")

    col_img, col_acao = st.columns([1, 1.35])
    with col_img:
        st.image(imagem, caption=st.session_state.get("origem", "Lâmina carregada"),
                 use_container_width=True)
    with col_acao:
        st.markdown("#### Lâmina carregada")
        st.write(
            f"Dimensões originais: {imagem.width} × {imagem.height} px. A imagem é "
            "redimensionada para 224 × 224 e normalizada antes de entrar na rede."
        )
        desc = st.session_state.get("descricao")
        ref = st.session_state.get("referencia")
        if desc:
            st.caption(desc)
        if ref:
            st.caption(f"Classe de referência do atlas: **{ref}**.")

        b1, b2 = st.columns([1.7, 1])
        with b1:
            if st.button("Executar análise citológica", type="primary",
                         use_container_width=True):
                st.session_state["rodar"] = True
        with b2:
            st.button("Limpar", on_click=limpar_imagem, use_container_width=True)

    if not st.session_state.get("rodar"):
        return

    st.markdown("---")

    # É aqui, e só aqui, que o modelo é necessário.
    with st.spinner("Preparando o modelo…"):
        carregado = obter_modelo()
    if carregado is None:
        return
    modelo, dispositivo, variante = carregado
    st.session_state["variante"] = variante

    with st.spinner("Extraindo características celulares…"):
        try:
            r = analisar(modelo, dispositivo, imagem,
                         verificar_robustez=st.session_state.get("robustez", True))
        except Exception as erro:
            st.error(f"A análise falhou. Detalhe técnico: {erro}")
            return

    mostrar_resultado(r)


def mostrar_resultado(r):
    info = CLASS_INFO[r["classe"]]
    linha_extra = ""
    if r["concordancia"] is not None:
        linha_extra = (f' · Concordância entre 8 orientações: '
                       f'<b>{r["concordancia"]:.0%}</b>')

    st.markdown(
        f'<div class="cartao" style="border-left:5px solid {info["cor"]}">'
        f'<div class="sobrancelha">Sugestão do modelo</div>'
        f'<div style="font-size:1.55rem;font-weight:800;color:{info["cor"]}">'
        f'{r["classe"]}. {info["nome"]}</div>'
        f'<div style="color:var(--tinta-fraca);margin-top:.35rem">'
        f'Confiança do algoritmo: <b>{r["confianca"]:.1%}</b>{linha_extra}</div>'
        f'<div style="color:var(--tinta-fraca);margin-top:.3rem">{info["resumo"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if (r["confianca"] < LIMIAR_CONFIANCA
            or (r["concordancia"] is not None
                and r["concordancia"] < LIMIAR_CONCORDANCIA)):
        st.warning(
            "Predição instável: a rede muda de opinião quando a imagem é girada ou "
            "espelhada. Trate este resultado como indefinido."
        )
    elif r["prob_tta"] is not None and r["classe_tta"] != r["classe"]:
        st.warning(
            f"A média sobre as oito orientações aponta {r['classe_tta']}, diferente "
            "da leitura direta. Vale repetir com outra imagem do mesmo campo."
        )

    ref = st.session_state.get("referencia")
    if ref:
        if ref == r["classe"]:
            st.success(f"A predição coincide com a referência do atlas ({ref}).")
        else:
            st.error(
                f"A predição ({r['classe']}) diverge da referência do atlas ({ref}). "
                "Divergências como esta são o material mais útil para melhorar o modelo."
            )

    st.write("")
    regua_bethesda(r["classe"])

    st.write("")
    col_a, col_b = st.columns([1.3, 1])
    with col_a:
        st.markdown("#### Onde a rede olhou")
        g1, g2 = st.columns(2)
        g1.image(r["img"], caption="Campo original", use_container_width=True)
        g2.image(r["sobreposto"], caption="Grad-CAM sobreposto", use_container_width=True)
        st.caption(
            "Tons claros marcam as regiões de maior influência na classe escolhida. "
            "Se o destaque cair em fundo, muco ou artefato de preparo, desconfie da predição."
        )
    with col_b:
        st.markdown("#### Confiança por classe")
        st.pyplot(grafico_probabilidades(r["prob"], r["idx"]))

    if r["prob_tta"] is not None:
        with st.expander("Teste de robustez: mesma lâmina, oito orientações"):
            st.write(
                "O treinamento usou giros e rotações de 90°, então a predição não "
                "deveria mudar com a orientação da imagem. A tabela compara a leitura "
                "direta com a média das oito simetrias."
            )
            st.dataframe(
                pd.DataFrame({
                    "Classe": CLASSES,
                    "Leitura direta": [f"{p:.1%}" for p in r["prob"]],
                    "Média das 8 orientações": [f"{p:.1%}" for p in r["prob_tta"]],
                }),
                hide_index=True, use_container_width=True,
            )
            st.pyplot(painel_aumentos(r["img"]))


def tela_de_boas_vindas():
    st.markdown("### Como funciona")
    passos = [
        ("Passo 1", "Carregue um campo",
         "Use o seletor acima ou abra uma das imagens de teste."),
        ("Passo 2", "Execute a análise",
         "A rede classifica a imagem nas quatro categorias do Sistema Bethesda."),
        ("Passo 3", "Leia o mapa de atenção",
         "O Grad-CAM mostra quais regiões pesaram na decisão."),
    ]
    colunas = st.columns(3)
    for coluna, (etapa, titulo, texto) in zip(colunas, passos):
        with coluna:
            st.markdown(
                f'<div class="cartao"><div class="sobrancelha">{etapa}</div>'
                f'<b>{titulo}</b><br><span style="color:var(--tinta-fraca)">{texto}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown("### Escala de gravidade do Sistema Bethesda")
    regua_bethesda()
    st.caption(
        "A seção **Sistema Bethesda** explica cada categoria em detalhe: o que "
        "significa, o que se vê ao microscópio e que conduta costuma decorrer dela."
    )

    st.write("")
    esq, dir_ = st.columns([1.15, 1])
    with esq:
        st.markdown("#### Por que rastrear")
        st.write(
            "O câncer do colo do útero é dos mais evitáveis, e ainda assim segue entre "
            "as principais causas de morte por câncer em mulheres no Amazonas. A "
            "citologia em meio líquido detecta lesões precursoras anos antes da "
            "malignização. O gargalo está no volume de lâminas por citopatologista. "
            "Uma triagem automatizada pode ordenar a fila, colocando os casos "
            "suspeitos na frente."
        )
    with dir_:
        st.markdown("#### Como o modelo foi treinado")
        st.write(
            "ConvNeXt Large com *transfer learning* a partir do ImageNet, ajustado "
            "em dois conjuntos públicos de citologia em meio líquido. Aumentos "
            "aplicados no treino:"
        )
        st.dataframe(
            pd.DataFrame(AUMENTOS_TREINO, columns=["Transformação", "Motivo"]),
            hide_index=True, use_container_width=True,
        )


# =============================================================================
# SEÇÃO 8. SISTEMA BETHESDA
# =============================================================================

def secao_bethesda():
    st.markdown("### O que é o Sistema Bethesda")
    col_txt, col_reg = st.columns([1.35, 1])
    with col_txt:
        st.markdown(TEXTO_BETHESDA_INTRO)
    with col_reg:
        st.markdown(
            '<div class="cartao">'
            '<div class="sobrancelha">Em uma frase</div>'
            '<div style="font-size:.95rem;line-height:1.6;color:var(--tinta-fraca)">'
            'Um vocabulário comum que transforma o que se vê ao microscópio numa '
            'categoria que orienta conduta clínica, e permite comparar resultados '
            'entre laboratórios, cidades e países.'
            '</div></div>',
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            '<div class="cartao">'
            '<div class="sobrancelha">Por que a ordem importa</div>'
            '<div style="font-size:.95rem;line-height:1.6;color:var(--tinta-fraca)">'
            'As categorias formam uma escala. Um erro entre vizinhas costuma adiar '
            'um seguimento; um erro entre extremos pode perder um câncer. É por isso '
            'que a matriz de confusão diz mais que a acurácia isolada.'
            '</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    regua_bethesda()

    st.markdown("---")
    st.markdown("### As quatro categorias em detalhe")
    st.caption(
        "Cada bloco reúne o que a categoria significa, o que se vê ao microscópio e "
        "o que costuma decorrer dela em termos de conduta."
    )
    st.write("")

    for sigla in ORDEM_GRAVIDADE:
        info = CLASS_INFO[sigla]
        det = BETHESDA_DETALHE[sigla]
        st.markdown(
            f'<div class="cartao" style="border-left:5px solid {info["cor"]};'
            f'margin-bottom:.5rem">'
            f'<div style="font-size:1.3rem;font-weight:800;color:{info["cor"]}">'
            f'{sigla}</div>'
            f'<div style="font-weight:600;font-size:1rem">{info["nome"]}</div>'
            f'<div style="color:var(--tinta-fraca);font-size:.85rem;font-style:italic;'
            f'margin-top:.15rem">{det["traducao"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**O que significa**")
            st.markdown(
                f'<div style="font-size:.9rem;line-height:1.55;color:var(--tinta-fraca)">'
                f'{det["o_que_e"]}</div>', unsafe_allow_html=True)
        with c2:
            st.markdown("**O que se vê ao microscópio**")
            st.markdown(
                f'<div style="font-size:.9rem;line-height:1.55;color:var(--tinta-fraca)">'
                f'{det["morfologia"]}</div>', unsafe_allow_html=True)
        with c3:
            st.markdown("**Conduta habitual**")
            st.markdown(
                f'<div style="font-size:.9rem;line-height:1.55;color:var(--tinta-fraca)">'
                f'{det["conduta"]}</div>', unsafe_allow_html=True)
            st.caption(det["frequencia"])
        st.markdown("---")

    st.markdown("### O que este modelo não classifica")
    st.markdown(TEXTO_BETHESDA_ALEM)

    st.info(
        "As descrições acima são material didático desta plataforma. Não substituem "
        "as definições oficiais do Sistema Bethesda nem as diretrizes do Ministério "
        "da Saúde e do INCA para conduta clínica."
    )


# =============================================================================
# SEÇÃO 9. VALIDAÇÃO
# =============================================================================

def secao_validacao():
    st.markdown("### Validação do algoritmo")
    st.caption(
        f"{METRICAS['protocolo']} · {METRICAS['conjunto']} · arquitetura: `{variante_conhecida()}`"
    )
    st.write(
        f"O modelo foi avaliado por validação cruzada estratificada em cinco partições, "
        f"somando **{METRICAS['n_imagens']} imagens**. Cada imagem foi classificada "
        "exatamente uma vez, por um modelo que não a viu no treino. É um protocolo mais "
        "exigente que uma única divisão treino/validação, porque toda a base entra na "
        "avaliação e o resultado não depende do sorteio de uma partição específica."
    )

    st.write("")
    ic = METRICAS["ic95"]
    c1, c2, c3, c4 = st.columns(4)
    cartao_numero(c1, "Acurácia global", METRICAS["acuracia"],
                  nota=f"IC 95%: {ic[0]:.1%} – {ic[1]:.1%}", cor="#0E6B7B")
    cartao_numero(c2, "F1 macro", METRICAS["f1_macro"])
    cartao_numero(c3, "Precisão macro", METRICAS["precisao_macro"])
    cartao_numero(c4, "Recall macro", METRICAS["recall_macro"])

    st.write("")
    c5, c6, c7 = st.columns(3)
    cartao_numero(c5, "Kappa de Cohen", METRICAS["kappa"], pct=False,
                  nota="Concordância além do acaso")
    cartao_numero(c6, "Imagens avaliadas", METRICAS["n_imagens"], pct=False,
                  nota=f"{METRICAS['acertos']} classificações corretas")
    cartao_numero(c7, "Erros", METRICAS["n_imagens"] - METRICAS["acertos"], pct=False,
                  nota="Nenhum deles subestimou um caso grave")

    st.markdown("---")
    st.markdown("#### Matriz de confusão global")
    col_fig, col_txt = st.columns([1.1, 1])
    with col_fig:
        st.pyplot(matriz_confusao_figura(MATRIZ_CONFUSAO, ORDEM_GRAVIDADE))
    with col_txt:
        st.markdown(
            "Cada linha é a classe de referência e cada coluna, a decisão da rede. "
            "A diagonal são os acertos.\n\n"
            "**O que mais importa não é o total de erros, é a direção deles.** Numa "
            "ferramenta de triagem, confundir LSIL com NILM adia um seguimento; "
            "confundir carcinoma com NILM perde um caso.\n\n"
            "Nesta validação, **nenhuma das 312 imagens de HSIL ou carcinoma foi "
            "classificada como normal**. O erro mais frequente foi o oposto, sete "
            "carcinomas lidos como HSIL, o que mantém a paciente na via de "
            "investigação. Seis campos de LSIL foram para NILM: é o achado que merece "
            "mais atenção, porque adia o seguimento."
        )

    with st.expander("Figura original do experimento"):
        if os.path.exists(FIG_MATRIZ):
            st.image(FIG_MATRIZ, use_container_width=True)
            st.caption(
                "Rótulos como saíram do treinamento: Carcinoma, HSIL, LSIL, Normal. "
                "ordem alfabética das pastas, que é a mesma dos índices do modelo."
            )
        else:
            st.info("Coloque `matriz_confusao.png` em `assets/` para exibir a figura original.")

    st.markdown("---")
    st.markdown("#### Desempenho por classe")
    linhas = []
    for sigla in ORDEM_GRAVIDADE:
        d = METRICAS["por_classe"][sigla]
        linhas.append({
            "Classe": sigla,
            "Descrição": CLASS_INFO[sigla]["nome"],
            "Precisão": f"{d['precisao']:.1%}",
            "Recall (sensibilidade)": f"{d['recall']:.1%}",
            "F1": f"{d['f1']:.1%}",
            "Imagens": d["suporte"],
        })
    st.dataframe(pd.DataFrame(linhas), hide_index=True, use_container_width=True)
    st.caption(
        "Em rastreio, o recall de HSIL e SCC é a métrica decisiva: mede quantos casos "
        "graves escapariam. HSIL alcançou 99,1% e carcinoma 92,7%, e as sete imagens "
        "de carcinoma não reconhecidas foram classificadas como HSIL, ou seja, "
        "permaneceram sinalizadas como alteração de alto grau."
    )

    if os.path.exists(FIG_HISTORICO):
        st.markdown("---")
        st.markdown("#### Curvas de treinamento")
        st.image(FIG_HISTORICO, use_container_width=True)
        st.caption(
            "A acurácia de treino satura cedo e a de validação oscila, comportamento "
            "esperado num conjunto curado. O checkpoint publicado corresponde ao melhor "
            "ponto de validação, não à última época."
        )

    st.markdown("---")
    st.markdown("#### O que estes números ainda não dizem")
    st.markdown(
        "- A validação é **interna aos dois conjuntos públicos usados no treino**. "
        "Desempenho em lâminas do "
        "Laboratório Sebastião Marinho, com outro scanner, outra coloração e outra "
        "prevalência, ainda não foi medido. É justamente o objeto da parceria com a "
        "SEMSA Manaus.\n"
        "- A prevalência aqui não é a de um programa de rastreio. No acervo, 42% dos "
        "campos são alterados; na rotina, a proporção de exames alterados é bem menor. "
        "O valor preditivo positivo em campo será menor que a precisão medida.\n"
        "- Métricas por imagem não equivalem a métricas por lâmina ou por paciente. "
        "Uma lâmina tem centenas de campos, e a decisão clínica é sobre a lâmina.\n"
        "- O modelo só conhece quatro categorias. Atipias de significado indeterminado "
        "(ASC-US, ASC-H), lesões glandulares (AGC, AIS) e amostras insatisfatórias não "
        "têm classe própria e serão forçadas para uma das quatro.\n"
        "- Não há comparação com dupla leitura humana, que é o padrão de referência real.\n"
        "- Sem aprovação de CEP, nenhum dado de paciente pode ser processado nesta "
        "ferramenta."
    )


# =============================================================================
# SEÇÃO 10. IMAGENS DE TESTE
# =============================================================================

def secao_exemplos():
    st.markdown("### Imagens de teste")
    st.write(
        "Campos do IARC Digital Atlas que acompanham a plataforma. Clique em "
        "**Analisar** para enviá-los direto ao modelo: a navegação salta para a seção "
        "Análise e o resultado já aparece pronto."
    )

    if st.session_state.pop("erro_exemplo", None):
        st.error("Arquivo não encontrado. Confira se os `.jpg` estão na pasta `exemplos/`.")

    st.write("")
    disponiveis = [e for e in EXEMPLOS if caminho_exemplo(e["arquivo"])]
    if not disponiveis:
        st.info("Nenhuma imagem encontrada. Coloque os arquivos `.jpg` em `exemplos/`.")
        return

    for i in range(0, len(disponiveis), 2):
        colunas = st.columns(2)
        for coluna, ex in zip(colunas, disponiveis[i:i + 2]):
            with coluna:
                caminho = caminho_exemplo(ex["arquivo"])
                st.image(caminho, use_container_width=True)
                if ex["referencia"]:
                    info = CLASS_INFO[ex["referencia"]]
                    st.markdown(
                        f'<span class="etiqueta" style="background:{info["cor"]}1A;'
                        f'color:{info["cor"]}">{ex["referencia"]} · referência do atlas'
                        f'</span>',
                        unsafe_allow_html=True,
                    )
                st.markdown(f"**{ex['titulo']}** · `{ex['arquivo']}`")
                st.caption(ex["descricao"])

                b1, b2 = st.columns([1, 1])
                with b1:
                    st.button(
                        "Analisar",
                        key=f"an_{ex['arquivo']}",
                        type="primary",
                        use_container_width=True,
                        on_click=carregar_exemplo,
                        args=(ex,),
                    )
                with b2:
                    with open(caminho, "rb") as arq:
                        st.download_button(
                            "Baixar", arq.read(), file_name=ex["arquivo"],
                            mime="image/jpeg", key=f"dl_{ex['arquivo']}",
                            use_container_width=True,
                        )
                st.write("")


# =============================================================================
# SEÇÃO 11. EQUIPE E PARCERIAS
# =============================================================================

def secao_equipe():
    st.markdown("### Equipe")
    colunas = st.columns(2)
    for i, pessoa in enumerate(EQUIPE):
        with colunas[i % 2]:
            st.markdown(
                f'<div class="pessoa"><div class="nome">{pessoa["nome"]}</div>'
                f'<div class="papel">{pessoa["papel"]}</div>'
                f'<div class="vinculo">{pessoa["vinculo"]}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("### Instituições parceiras")
    logos = {
        "UFAM / FCF": LOGO_UFAM,
        "UFAM / IComp": LOGO_ICOMP,
        "SEMSA Manaus": LOGO_SEMSA,
    }
    for inst in INSTITUICOES:
        col_logo, col_txt = st.columns([1, 6], vertical_alignment="center")
        with col_logo:
            _logo(logos.get(inst["sigla"]), 105)
        with col_txt:
            st.markdown(
                f'<div class="cartao">'
                f'<span class="etiqueta">{inst["sigla"]}</span>'
                f'<div style="font-weight:700;margin-top:.6rem">{inst["nome"]}</div>'
                f'<div style="color:var(--tinta-fraca);margin-top:.35rem;font-size:.93rem">'
                f'{inst["papel"]}</div></div>',
                unsafe_allow_html=True,
            )
        st.write("")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Ficha técnica")
        st.markdown(
            f"**Arquitetura** · `{variante_conhecida()}`, *transfer learning* do ImageNet  \n"
            "**Interpretabilidade** · Grad-CAM sobre o último bloco convolucional  \n"
            "**Dados** · IARC Digital Atlas e Liquid Based Cytology Pap Smear "
            "(Mendeley Data). Detalhes na seção Dados e créditos  \n"
            "**Entrada** · campo digitalizado de citologia em meio líquido, RGB  \n"
            f"**Validação** · {METRICAS['protocolo']}, {METRICAS['n_imagens']} imagens"
        )
    with col_b:
        st.markdown("#### Situação regulatória")
        st.markdown(
            "Todo o desenvolvimento usou **apenas o acervo público IARC**. Nenhuma "
            "lâmina, imagem ou dado de paciente atendido pela rede municipal foi "
            "acessado.\n\n"
            "- O projeto **ainda não foi aprovado por Comitê de Ética em Pesquisa**. "
            "A submissão ao CEP é pré-requisito da fase seguinte.\n"
            "- Não há registro na ANVISA como software como dispositivo médico (SaMD), "
            "exigido para uso assistencial.\n"
            "- A responsabilidade diagnóstica permanece integralmente com o "
            "citopatologista."
        )

    st.markdown("---")
    st.markdown("#### Próximos passos")
    st.markdown(
        "1. Submissão do protocolo ao CEP/UFAM.\n"
        "2. Validação externa independente com lâminas do Laboratório Sebastião "
        "Marinho, sob supervisão das citopatologistas da SEMSA Manaus, medindo "
        "desempenho por lâmina e por paciente.\n"
        "3. Concordância entre a IA e a dupla leitura humana.\n"
        "4. Estudo de impacto no tempo de fila e na taxa de detecção de HSIL ou pior."
    )


# =============================================================================
# SEÇÃO 11b. DADOS E CRÉDITOS
# =============================================================================

def secao_dados_creditos():
    st.markdown("### Conjuntos de dados de treinamento")
    st.write(
        "O modelo foi treinado e validado sobre dois acervos públicos de citologia "
        "em meio líquido. Nenhuma imagem de paciente atendido pela rede municipal "
        "foi utilizada em qualquer etapa."
    )
    st.write("")

    for ds in DATASETS:
        st.markdown(
            f'<div class="cartao" style="margin-bottom:.5rem">'
            f'<span class="etiqueta">{ds["sigla"]}</span>'
            f'<div style="font-weight:700;margin-top:.6rem;font-size:1.02rem">'
            f'{ds["nome"]}</div>'
            f'<div style="color:var(--tinta-fraca);font-size:.88rem;margin-top:.2rem">'
            f'{ds["instituicao"]}</div>'
            f'<div style="color:var(--tinta-fraca);margin-top:.5rem;font-size:.93rem">'
            f'{ds["descricao"]}</div>'
            f'<div style="margin-top:.55rem;font-size:.86rem">'
            f'<a href="{ds["url"]}" target="_blank" rel="noopener">{ds["url"]}</a>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        if ds["referencia"]:
            st.markdown(f"**Como citar:** {ds['referencia']}")
        st.write("")

    st.info(
        "A combinação dos dois acervos foi o que permitiu chegar às 1096 imagens da "
        "validação cruzada. Ela também amplia a variedade de preparo e coloração "
        "vista no treino, o que tende a ajudar na generalização. Ainda assim, "
        "continuam sendo acervos curados para ensino e pesquisa, com prevalência de "
        "lesões bem diferente da rotina de um programa de rastreio."
    )

    st.markdown("---")
    st.markdown("### Ferramentas de apoio ao desenvolvimento")
    st.markdown(FERRAMENTAS_APOIO)
    st.caption(
        "Registramos esta informação por transparência científica, do mesmo modo "
        "que se declara o uso de qualquer outra ferramenta computacional."
    )

    st.markdown("---")
    st.markdown("### Bibliotecas e tecnologias")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            "**Modelo e treinamento**  \n"
            "PyTorch, torchvision (ConvNeXt Large), Albumentations, "
            "scikit-learn (validação cruzada estratificada), NumPy"
        )
        st.markdown(
            "**Interpretabilidade**  \n"
            "Grad-CAM implementado sobre o último bloco convolucional"
        )
    with col_b:
        st.markdown(
            "**Plataforma web**  \n"
            "Streamlit, OpenCV, Matplotlib, pandas, Pillow"
        )
        st.markdown(
            "**Aplicativo Android**  \n"
            "Flutter, ONNX Runtime para inferência local no aparelho"
        )

    st.markdown("---")
    st.markdown("### Como citar esta plataforma")
    st.code(
        "OLIVEIRA, M. D. L.; MARTINS, T. R.; NAKAMURA, F.; OLIVEIRA, F. G. "
        "Sistema de apoio ao diagnóstico citológico em meio líquido assistido por "
        "inteligência artificial. Faculdade de Ciências Farmacêuticas e Instituto "
        "de Computação, Universidade Federal do Amazonas, em parceria com a "
        f"Secretaria Municipal de Saúde de Manaus, {ANO_DESENVOLVIMENTO}.",
        language=None,
    )
    st.caption(
        "Trabalho em andamento, ainda sem publicação revisada por pares e sem "
        "aprovação de Comitê de Ética em Pesquisa."
    )


# =============================================================================
# SEÇÃO 12. APLICAÇÃO
# =============================================================================

def main():
    injetar_css()
    injetar_metadados()

    if "menu" not in st.session_state:
        st.session_state["menu"] = MENUS[0]

    # A interface inteira é desenhada antes de qualquer coisa pesada: cabeçalho,
    # navegação e conteúdo aparecem imediatamente. O modelo só é carregado
    # quando há de fato uma análise para rodar (ver obter_modelo).
    cabecalho()
    navegacao()

    escolha = st.session_state["menu"]
    if escolha == "Análise":
        secao_analise()
    elif escolha == "Sistema Bethesda":
        secao_bethesda()
    elif escolha == "Validação do algoritmo":
        secao_validacao()
    elif escolha == "Imagens de teste":
        secao_exemplos()
    elif escolha == "Equipe e parcerias":
        secao_equipe()
    else:
        secao_dados_creditos()

    rodape()


if __name__ == "__main__":
    main()
