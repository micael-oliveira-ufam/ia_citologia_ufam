"""Conteúdo do site, portado de app.py (versão Streamlit) sem alterações de sentido.

Fonte única dos textos e números: o gerador (gerar_site.py) produz o HTML estático
e o docs/js/dados.js a partir daqui. Ao mudar um número da validação, mude AQUI e
rode `python ferramentas/gerar_site.py`; no Streamlit, o equivalente é METRICAS em app.py.
"""

ANO_DESENVOLVIMENTO = "2026"
URL_SITE = "https://micael-oliveira-ufam.github.io/ia_citologia_ufam/"
URL_STREAMLIT = "https://ia-citologia-ufam.streamlit.app/"
URL_REPO = "https://github.com/micael-oliveira-ufam/ia_citologia_ufam"

# Mesma ordem do treinamento (ver PARAMETROS_COMPARTILHADOS.md). NÃO reordenar.
CLASSES = ["SCC", "HSIL", "LSIL", "NILM"]
ORDEM_GRAVIDADE = ["NILM", "LSIL", "HSIL", "SCC"]
LIMIAR_CONFIANCA = 0.60
LIMIAR_CONCORDANCIA = 0.75

CLASS_INFO = {
    "NILM": {"nome": "Negativo para lesão intraepitelial ou malignidade", "cor": "#2E7D62", "gravidade": 0,
             "resumo": "Sem alterações sugestivas de lesão precursora ou câncer."},
    "LSIL": {"nome": "Lesão intraepitelial escamosa de baixo grau", "cor": "#C08A1E", "gravidade": 1,
             "resumo": "Alterações associadas ao HPV. Maioria regride, mas exige seguimento."},
    "HSIL": {"nome": "Lesão intraepitelial escamosa de alto grau", "cor": "#D9642E", "gravidade": 2,
             "resumo": "Lesão precursora com risco relevante de progressão."},
    "SCC": {"nome": "Carcinoma de células escamosas", "cor": "#B23A3A", "gravidade": 3,
            "resumo": "Achados compatíveis com carcinoma invasor."},
}

BETHESDA_DETALHE = {
    "NILM": {
        "traducao": "Negative for Intraepithelial Lesion or Malignancy",
        "o_que_e": ("Categoria de normalidade. O material é adequado e não há células com "
                    "alterações sugestivas de lesão precursora ou de câncer. Não significa "
                    "ausência de qualquer achado: infecções, alterações reativas por "
                    "inflamação, atrofia ou uso de DIU são relatadas dentro de NILM."),
        "morfologia": ("Células escamosas superficiais e intermediárias com núcleos pequenos, "
                       "regulares e cromatina fina. Relação núcleo/citoplasma baixa. Células "
                       "endocervicais colunares preservadas, em paliçada ou favo de mel."),
        "conduta": ("Retorno ao rastreio de rotina, no intervalo previsto pelo programa. "
                    "Achados infecciosos ou reativos podem motivar tratamento próprio, sem "
                    "mudar a conduta de rastreio."),
        "frequencia": "É o resultado da grande maioria dos exames de rastreio.",
    },
    "LSIL": {
        "traducao": "Low-grade Squamous Intraepithelial Lesion",
        "o_que_e": ("Reúne o efeito citopático do HPV (coilocitose) e a neoplasia "
                    "intraepitelial cervical de grau 1 (NIC 1). Representa infecção "
                    "produtiva pelo vírus, e não ainda uma lesão com autonomia "
                    "proliferativa."),
        "morfologia": ("Células escamosas maduras com núcleos aumentados, mais de três vezes o "
                       "núcleo de uma célula intermediária normal, hipercromáticos e de "
                       "contorno irregular. O achado mais característico é o coilócito: halo "
                       "perinuclear amplo e bem delimitado, com borda citoplasmática densa."),
        "conduta": ("Seguimento citológico ou colposcopia, conforme a idade e o protocolo "
                    "vigente. A maioria das lesões regride espontaneamente em até dois "
                    "anos, sobretudo em mulheres jovens."),
        "frequencia": "A alteração mais comum entre os exames alterados.",
    },
    "HSIL": {
        "traducao": "High-grade Squamous Intraepithelial Lesion",
        "o_que_e": ("Agrupa NIC 2 e NIC 3, incluindo o carcinoma in situ. É a verdadeira "
                    "lesão precursora: há proliferação de células imaturas com potencial "
                    "real de progressão para carcinoma invasor se não tratada."),
        "morfologia": ("Células menores e imaturas, isoladas ou em agregados sinciciais, com "
                       "relação núcleo/citoplasma alta, com o núcleo ocupando a maior parte da "
                       "célula. Cromatina grosseira e irregularmente distribuída, contornos "
                       "nucleares recortados, citoplasma escasso e frequentemente basofílico."),
        "conduta": ("Encaminhamento para colposcopia com biópsia dirigida, e tratamento "
                    "excisional quando confirmada. É a categoria que o rastreio existe "
                    "para encontrar."),
        "frequencia": "Menos frequente que LSIL, com impacto clínico bem maior.",
    },
    "SCC": {
        "traducao": "Squamous Cell Carcinoma",
        "o_que_e": ("Carcinoma invasor de células escamosas: a neoplasia rompeu a membrana "
                    "basal e invadiu o estroma. Não é mais lesão precursora, e sim câncer "
                    "estabelecido."),
        "morfologia": ("Acentuado pleomorfismo celular e nuclear, nucléolos evidentes, "
                       "cromatina irregular e grumosa. Podem aparecer células queratinizadas "
                       "de formas bizarras: fusiformes, em girino, em fibra. O fundo costuma "
                       "trazer a chamada diátese tumoral: sangue, debris necróticos e "
                       "material proteináceo granular."),
        "conduta": ("Encaminhamento imediato para confirmação histopatológica e "
                    "estadiamento. Prioridade máxima na fila."),
        "frequencia": "Raro no rastreio organizado, e é justamente esse o objetivo.",
    },
}

BETHESDA_INTRO = [
    ("O Sistema Bethesda é a linguagem padronizada para relatar exames citopatológicos "
     "do colo do útero. Foi criado em 1988, nos Estados Unidos, e revisado em 1991, "
     "2001 e 2014. Antes dele, cada laboratório usava a classificação de Papanicolaou "
     "em cinco classes numéricas, que dizia pouco sobre a conduta a tomar e variava de "
     "serviço para serviço."),
    ("A mudança central foi abandonar números e adotar categorias que se traduzem "
     "diretamente em decisão clínica. O laudo passou a informar também se a amostra "
     "era adequada para leitura. Uma lâmina com material escasso ou obscurecido por "
     "sangue não é um resultado negativo, é uma lâmina que precisa ser repetida."),
    ("As quatro categorias abaixo são as que este modelo classifica. Elas formam uma "
     "escala ordinal: da ausência de lesão até o carcinoma invasor. Essa ordem importa "
     "na leitura dos erros do algoritmo, porque confundir categorias vizinhas tem "
     "consequência clínica diferente de confundir categorias distantes."),
]

BETHESDA_ALEM_INTRO = "O Sistema Bethesda completo é mais amplo que estas quatro categorias. Um laudo real também pode registrar:"
BETHESDA_ALEM_ITENS = [
    ("ASC-US", "atipias em células escamosas de significado indeterminado, quando as alterações existem mas não bastam para caracterizar LSIL."),
    ("ASC-H", "atipias escamosas em que não se pode excluir lesão de alto grau."),
    ("AGC", "atipias em células glandulares, de origem endocervical ou endometrial."),
    ("AIS", "adenocarcinoma in situ."),
    ("Adenocarcinoma", "de origem endocervical, endometrial ou extrauterina."),
    ("Avaliação da adequabilidade", "satisfatória ou insatisfatória para análise."),
]
BETHESDA_ALEM_FIM = ("Este modelo <strong>não</strong> classifica essas categorias: foi treinado apenas nas quatro "
                     "acima. Um campo com atipia glandular ou com material insatisfatório será "
                     "forçosamente encaixado numa das quatro classes conhecidas, com confiança que "
                     "pode ser alta e ainda assim errada. É uma limitação de escopo, não um defeito de "
                     "treinamento, e uma das razões pelas quais a leitura humana continua indispensável.")

AUMENTOS_TREINO = [
    ("Resize(224, 224)", "Padroniza a entrada da rede"),
    ("HorizontalFlip(p=0.5)", "A lâmina não tem orientação preferencial"),
    ("VerticalFlip(p=0.5)", "Idem, no eixo vertical"),
    ("RandomRotate90(p=0.5)", "Rotações múltiplas de 90°"),
    ("ColorJitter(0.2, 0.2, 0.2, hue=0.1, p=0.3)", "Variação de coloração entre lotes"),
    ("Normalize(ImageNet)", "Estatísticas do pré-treino"),
    ("ToTensorV2()", "Conversão para tensor"),
]

MATRIZ_CONFUSAO = [  # linhas = verdadeiro; colunas = predito; ordem NILM, LSIL, HSIL, SCC
    [634, 0, 3, 0],
    [6, 139, 2, 0],
    [0, 1, 214, 1],
    [0, 0, 7, 89],
]

METRICAS = {
    "protocolo": "Validação cruzada estratificada, 5 folds",
    "conjunto": "IARC Digital Atlas e Liquid Based Cytology Pap Smear (Mendeley Data)",
    "n_imagens": 1096, "acertos": 1076,
    "acuracia": 0.9818, "ic95": (0.9720, 0.9882),
    "f1_macro": 0.9717, "precisao_macro": 0.9798, "recall_macro": 0.9647, "kappa": 0.9694,
    "por_classe": {
        "NILM": {"precisao": 0.9906, "recall": 0.9953, "f1": 0.9930, "suporte": 637},
        "LSIL": {"precisao": 0.9929, "recall": 0.9456, "f1": 0.9686, "suporte": 147},
        "HSIL": {"precisao": 0.9469, "recall": 0.9907, "f1": 0.9683, "suporte": 216},
        "SCC": {"precisao": 0.9889, "recall": 0.9271, "f1": 0.9570, "suporte": 96},
    },
}

AVISO_ETICO_HTML = (
    "Ferramenta acadêmica em desenvolvimento. O modelo foi treinado "
    "<strong>exclusivamente com acervos públicos de imagens</strong>; "
    "nenhuma amostra de paciente do serviço foi utilizada. Por isso o projeto "
    "<strong>ainda não possui aprovação de Comitê de Ética em Pesquisa "
    "(CEP/CONEP)</strong> e <strong>não pode ser usado para decisão clínica, "
    "laudo ou triagem assistencial</strong>."
)

EQUIPE = [
    {"nome": "Micael Davi Lima de Oliveira", "papel": "Desenvolvimento do modelo e das aplicações",
     "vinculo": "Iniciação Científica, Faculdade de Ciências Farmacêuticas, UFAM"},
    {"nome": "Prof. Dr. Toni Ricardo Martins", "papel": "Coordenação e orientação",
     "vinculo": "Faculdade de Ciências Farmacêuticas, UFAM"},
    {"nome": "Profa. Dra. Fabíola Nakamura", "papel": "Pesquisadora parceira em computação",
     "vinculo": "Instituto de Computação (IComp), UFAM"},
    {"nome": "Prof. Dr. Felipe Gomes de Oliveira", "papel": "Pesquisador parceiro em computação",
     "vinculo": "Instituto de Computação (IComp), UFAM"},
    {"nome": "Dra. Ivanete", "papel": "Especialista clínica e citopatologista", "vinculo": "SEMSA Manaus"},
    {"nome": "Dra. Carol", "papel": "Especialista clínica e citopatologista", "vinculo": "SEMSA Manaus"},
]

INSTITUICOES = [
    {"sigla": "UFAM / FCF", "logo": "img/logo_ufam.png",
     "nome": "Faculdade de Ciências Farmacêuticas, Universidade Federal do Amazonas",
     "papel": "Instituição executora. Concepção, treinamento e validação do modelo."},
    {"sigla": "UFAM / IComp", "logo": "img/logo-icomp.png",
     "nome": "Instituto de Computação, Universidade Federal do Amazonas",
     "papel": ("Parceria em visão computacional e aprendizado profundo, com os "
               "professores pesquisadores Dra. Fabíola Nakamura e Dr. Felipe Gomes de Oliveira.")},
    {"sigla": "SEMSA Manaus", "logo": "img/semsa.png",
     "nome": "Secretaria Municipal de Saúde de Manaus, Laboratório Sebastião Marinho",
     "papel": ("Parceria institucional de validação clínica e citopatológica, "
               "promovendo suporte prático e validação independente do catálogo "
               "digital de lâminas. Especialistas clínicas e citopatologistas "
               "envolvidas: Dra. Ivanete e Dra. Carol.")},
]

EXEMPLOS = [
    {"arquivo": "cyto5940.jpg", "titulo": "Ectocérvice normal", "referencia": "NILM",
     "descricao": ("Ectocérvice normal: células escamosas intermediárias e superficiais, "
                   "basofílicas ou eosinofílicas. Presença de alguns polimorfonucleares. (obj. 10x)")},
    {"arquivo": "cyt14686a.jpg", "titulo": "Carcinoma invasivo", "referencia": "SCC",
     "descricao": ("Carcinoma de células escamosas invasivo: agrupamento de células "
                   "malignas pleomórficas, predominantemente pouco diferenciadas, e "
                   "células isoladas mais diferenciadas, necróticas ou queratinizadas, "
                   "com formas anômalas (elipses). Inflamação, sangue e necrose ao fundo. (obj. 20x)")},
    {"arquivo": "cyto5950.jpg", "titulo": "Campo adicional 1", "referencia": None,
     "descricao": "Escamosas maduras, núcleos pequenos, flora bacilar."},
    {"arquivo": "cyto2870.jpg", "titulo": "Campo adicional 2", "referencia": None,
     "descricao": "Colunares endocervicais em paliçada, fundo inflamatório."},
    {"arquivo": "cyt10131a.jpg", "titulo": "Campo adicional 3", "referencia": None,
     "descricao": "Halos perinucleares indicados por setas, exsudato abundante."},
    {"arquivo": "cyt16243.jpg", "titulo": "Campo adicional 4", "referencia": None,
     "descricao": "Agrupamentos densos, alta relação núcleo/citoplasma."},
]

DATASETS = [
    {"sigla": "IARC", "nome": "IARC Digital Atlas of Cervical Cytology",
     "instituicao": "Agência Internacional de Pesquisa em Câncer, vinculada à Organização Mundial da Saúde",
     "url": "https://screening.iarc.fr/atlascyto.php",
     "descricao": ("Atlas digital de referência em citologia cervical, com campos "
                   "anotados por citopatologistas e usado internacionalmente para ensino."),
     "referencia": None},
    {"sigla": "Mendeley Data",
     "nome": "Liquid based cytology Pap smear images for multi-class diagnosis of cervical cancer",
     "instituicao": "Institute of Advanced Study in Science and Technology, Índia",
     "url": "https://data.mendeley.com/datasets/zddtpgzv63/2",
     "descricao": ("Conjunto de imagens de citologia em meio líquido preparadas e "
                   "anotadas para classificação multiclasse de lesões pré-cancerosas e "
                   "de câncer cervical."),
     "referencia": ("HUSSAIN, Elima; MAHANTA, Lipi B.; BORAH, Himakshi; DAS, Chandana Ray. "
                    "Liquid based-cytology Pap smear dataset for automated multi-class "
                    "diagnosis of pre-cancerous and cervical cancer lesions. <em>Data in "
                    "Brief</em>, v. 30, p. 105589, jun. 2020. "
                    "DOI: <a href=\"https://doi.org/10.1016/j.dib.2020.105589\" target=\"_blank\" rel=\"noopener\">"
                    "https://doi.org/10.1016/j.dib.2020.105589</a>")},
]

FERRAMENTAS_APOIO = (
    "O código desta plataforma e do aplicativo Android foi desenvolvido com "
    "apoio do modelo de IA generativa <strong>Claude Opus 5</strong>, da Anthropic, usado "
    "como assistente de programação. A concepção científica, a escolha da "
    "arquitetura, o treinamento, a curadoria dos dados, a validação e a revisão "
    "de todo o código são de responsabilidade da equipe do projeto."
)

SEO_TITULO = "IA Citologia UFAM & SEMSA | Citopatologia com Inteligência Artificial"
SEO_DESCRICAO = (
    "Plataforma de apoio ao diagnóstico citológico em meio líquido pelo Sistema "
    "Bethesda com inteligência artificial. Projeto da Faculdade de Ciências "
    "Farmacêuticas e do Instituto de Computação da UFAM em parceria com a SEMSA "
    "Manaus. Classificação de NILM, LSIL, HSIL e carcinoma escamoso com ConvNeXt "
    "e mapa de atenção Grad-CAM."
)
SEO_PALAVRAS_CHAVE = [
    "IA Citologia UFAM & SEMSA", "IA Citologia UFAM", "Citopatologia SEMSA", "Citopatologia IA",
    "Citopatologia UFAM", "Citopatologia FCF", "Citopatologia ICOMP", "citologia em meio líquido",
    "citologia em meio líquido inteligência artificial", "Sistema Bethesda",
    "rastreio câncer do colo do útero", "câncer cervical Amazonas", "Papanicolaou inteligência artificial",
    "ConvNeXt citologia", "Grad-CAM citopatologia", "IARC Digital Atlas", "deep learning citopatologia",
    "UFAM Faculdade de Ciências Farmacêuticas", "Instituto de Computação UFAM", "SEMSA Manaus",
    "Laboratório Sebastião Marinho", "saúde digital Amazonas", "CitoPred",
]

MENUS = [  # (id, rótulo)
    ("analise", "Análise"),
    ("bethesda", "Sistema Bethesda"),
    ("validacao", "Validação do algoritmo"),
    ("exemplos", "Imagens de teste"),
    ("equipe", "Equipe e parcerias"),
    ("dados", "Dados e créditos"),
]
