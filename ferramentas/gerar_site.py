#!/usr/bin/env python3
"""Gera docs/index.html, docs/catalogo.html, docs/predicao.html (redirecionamento) e
docs/js/dados.js a partir de conteudo_site.py.

    python ferramentas/gerar_site.py

O HTML das seções de leitura (Bethesda, Validação, Equipe, Dados) é estático, para
funcionar sem JavaScript e ser indexado por buscadores. A análise e o catálogo são
feitos por JavaScript (docs/js).
"""
import html
import json
import sys
from pathlib import Path

import matplotlib

sys.path.insert(0, str(Path(__file__).parent))
from conteudo_site import *  # noqa: F401,F403,E402

DOCS = Path(__file__).resolve().parent.parent / "docs"
e = html.escape


def pct(v, casas=1):
    return f"{v * 100:.{casas}f}".replace(".", ",") + "%"


def milhar(n):
    return f"{n:,}".replace(",", ".")


# ----------------------------------------------------------------- estrutura

def cabecalho():
    return f"""
<header class="cabecalho">
  <div class="cab-linha">
    <a class="cab-logo cab-logo--ufam" href="index.html" aria-label="Início"><img src="img/logo_ufam.png" alt="Universidade Federal do Amazonas" width="86"></a>
    <div class="cab-titulo">
      <div class="sobrancelha">UFAM · IComp · SEMSA Manaus</div>
      <h1>Sistema de Apoio ao Diagnóstico Citológico</h1>
      <p class="subtitulo">Citologia em meio líquido classificada pelo Sistema Bethesda, com ConvNeXt Large e mapa de atenção Grad-CAM.</p>
    </div>
    <div class="cab-logos">
      <img src="img/logo-icomp.png" alt="Instituto de Computação, UFAM" height="52">
      <img src="img/semsa.png" alt="Secretaria Municipal de Saúde de Manaus" height="52">
    </div>
  </div>
  <div class="creditos">
    <div class="credito-bloco"><div class="credito-titulo">Desenvolvimento</div>
      <div class="credito-nome">Micael Davi Lima de Oliveira</div><div class="credito-sub">Iniciação Científica · FCF/UFAM</div></div>
    <div class="credito-bloco"><div class="credito-titulo">Coordenação</div>
      <div class="credito-nome">Prof. Dr. Toni Ricardo Martins</div><div class="credito-sub">Faculdade de Ciências Farmacêuticas · UFAM</div></div>
    <div class="credito-bloco"><div class="credito-titulo">Pesquisadores parceiros · IComp/UFAM</div>
      <div class="credito-nome">Profa. Dra. Fabíola Nakamura<br>Prof. Dr. Felipe Gomes de Oliveira</div></div>
    <div class="credito-bloco"><div class="credito-titulo">Validação clínica · SEMSA Manaus</div>
      <div class="credito-nome">Dra. Ivanete · Dra. Carol</div><div class="credito-sub">Especialistas clínicas e citopatologistas</div></div>
  </div>
  <div class="faixa-etica" role="note">{AVISO_ETICO_HTML}</div>
</header>"""


def navegacao(ativo, na_index):
    itens = []
    for sid, rotulo in MENUS:
        href = f"#{sid}" if na_index else f"index.html#{sid}"
        cur = ' aria-current="page"' if (ativo == sid) else ""
        itens.append(f'<a href="{href}" data-secao="{sid}"{cur}>{e(rotulo)}</a>')
    cur = ' aria-current="page"' if ativo == "catalogo" else ""
    itens.append(f'<a href="catalogo.html" data-secao="catalogo"{cur}>Catálogo de lâminas</a>')
    return f'<nav class="abas" aria-label="Seções do site">{"".join(itens)}</nav>'


def rodape():
    return f"""
<footer class="rodape">
  <div>
    <strong>Sistema de Apoio ao Diagnóstico Citológico</strong><br>
    Faculdade de Ciências Farmacêuticas e Instituto de Computação · UFAM<br>
    Secretaria Municipal de Saúde de Manaus · Laboratório Sebastião Marinho<br>
    <a href="{URL_REPO}" target="_blank" rel="noopener">Código-fonte no GitHub</a> ·
    <a href="{URL_REPO}/releases" target="_blank" rel="noopener">Aplicativo Android/iOS</a> ·
    <a href="{URL_STREAMLIT}" target="_blank" rel="noopener">Versão anterior (Streamlit)</a>
  </div>
  <div class="rodape-dir">
    <span class="ano">© {ANO_DESENVOLVIMENTO}</span> · Projeto de pesquisa acadêmica<br>
    Modelo <code>ConvNeXt Large</code> · IARC Digital Atlas e Mendeley Data<br>
    Sem aprovação de CEP · Não emite laudo<br>
    <span class="rodape-mini">Código desenvolvido com apoio de Claude Opus 5 (Anthropic)</span>
  </div>
</footer>"""


def meta_seo():
    palavras = ", ".join(SEO_PALAVRAS_CHAVE)
    ld = {
        "@context": "https://schema.org", "@type": "WebApplication", "name": SEO_TITULO,
        "alternateName": ["IA Citologia UFAM", "CitoPred"], "description": SEO_DESCRICAO,
        "url": URL_SITE, "applicationCategory": "HealthApplication", "inLanguage": "pt-BR",
        "keywords": palavras, "isAccessibleForFree": True,
        "creator": [{"@type": "Person", "name": p["nome"], "affiliation": p["vinculo"]} for p in EQUIPE],
        "publisher": {"@type": "CollegeOrUniversity", "name": "Universidade Federal do Amazonas",
                      "department": ["Faculdade de Ciências Farmacêuticas", "Instituto de Computação"]},
        "about": ["Citopatologia", "Citologia em meio líquido", "Sistema Bethesda",
                  "Rastreio do câncer do colo do útero", "Inteligência artificial"],
    }
    return f"""
  <meta name="description" content="{e(SEO_DESCRICAO)}">
  <meta name="keywords" content="{e(palavras)}">
  <meta name="author" content="Universidade Federal do Amazonas">
  <meta name="robots" content="index, follow">
  <link rel="canonical" href="{URL_SITE}">
  <meta property="og:title" content="{e(SEO_TITULO)}">
  <meta property="og:description" content="{e(SEO_DESCRICAO)}">
  <meta property="og:type" content="website">
  <meta property="og:locale" content="pt_BR">
  <meta property="og:url" content="{URL_SITE}">
  <meta name="twitter:card" content="summary">
  <meta name="twitter:title" content="{e(SEO_TITULO)}">
  <meta name="twitter:description" content="{e(SEO_DESCRICAO)}">
  <script type="application/ld+json">{json.dumps(ld, ensure_ascii=False)}</script>"""


ICONE = ("data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 32 32%27%3E%3Crect width=%2732%27 "
         "height=%2732%27 rx=%277%27 fill=%27%230e6b7b%27/%3E%3Ctext x=%2716%27 y=%2722%27 font-size=%2714%27 "
         "font-family=%27sans-serif%27 font-weight=%27700%27 text-anchor=%27middle%27 fill=%27white%27%3ECU%3C/text%3E%3C/svg%3E")


def pagina(titulo, corpo, ativo, na_index, scripts, extra_head="", descricao=None):
    return f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{e(titulo)}</title>
  <script>document.documentElement.classList.add("js")</script>{extra_head}
  <meta name="theme-color" content="#0e6b7b">
  <link rel="icon" href="{ICONE}">
  <link rel="preload" href="fontes/inter-latin-400-normal.woff2" as="font" type="font/woff2" crossorigin>
  <link rel="stylesheet" href="css/site.css?v=3">
</head>
<body>
  <a class="pular" href="#conteudo">Pular para o conteúdo</a>
  <div class="pagina">
{cabecalho()}
{navegacao(ativo, na_index)}
    <main id="conteudo" tabindex="-1">
{corpo}
    </main>
{rodape()}
  </div>
{scripts}
</body>
</html>
"""


# ------------------------------------------------------------------ componentes

def regua(ativa=None, id_=None):
    graus = []
    for s in ORDEM_GRAVIDADE:
        i = CLASS_INFO[s]
        a = " ativo" if s == ativa else ""
        graus.append(f'<div class="degrau{a}" data-sigla="{s}" style="border-color:{i["cor"]}">'
                     f'<div class="sigla" style="color:{i["cor"]}">{s}</div><div class="desc">{e(i["nome"])}</div></div>')
    ident = f' id="{id_}"' if id_ else ""
    return f'<div class="regua"{ident} role="img" aria-label="Escala de gravidade do Sistema Bethesda, de NILM a SCC">{"".join(graus)}</div>' \
           '<p class="legenda">Gravidade crescente da esquerda para a direita.</p>'


def cartao_numero(rotulo, texto, nota=None, cor=None):
    estilo = f' style="color:{cor}"' if cor else ""
    n = f'<div class="nota-numero">{nota}</div>' if nota else ""
    return f'<div class="cartao"><div class="rotulo-numero">{rotulo}</div><div class="numero"{estilo}>{texto}</div>{n}</div>'


def tabela_matriz():
    blues = matplotlib.colormaps["Blues"]
    ordem = ORDEM_GRAVIDADE
    linhas = []
    for i, s in enumerate(ordem):
        total = max(sum(MATRIZ_CONFUSAO[i]), 1)
        cels = []
        for j, _ in enumerate(ordem):
            n = MATRIZ_CONFUSAO[i][j]
            v = n / total
            r, g, b, _a = blues(v)
            cor = f"rgb({round(r*255)},{round(g*255)},{round(b*255)})"
            tinta = "#fff" if v >= 0.55 else "#14232F"
            cels.append(f'<td style="background:{cor};color:{tinta}"><b>{n}</b><span>{round(v*100)}%</span></td>')
        linhas.append(f'<tr><th scope="row">{s}</th>{"".join(cels)}</tr>')
    cab = "".join(f'<th scope="col">{s}</th>' for s in ordem)
    return f"""<table class="matriz">
<caption class="so-leitor">Matriz de confusão global: linhas são a referência do citopatologista, colunas a predição da IA.</caption>
<thead><tr><td class="canto"></td><th colspan="4" class="eixo">Predito pela IA</th></tr><tr><td class="canto"></td>{cab}</tr></thead>
<tbody>{"".join(linhas)}</tbody></table>
<p class="eixo-y">Referência (citopatologista) ↑ linhas</p>"""


# ------------------------------------------------------------------- seções

def sec_inicio():
    fontes = [
        ("INCA, Diretrizes Brasileiras para o Rastreamento do Câncer do Colo do Útero (3ª ed., 2025)",
         "https://ninho.inca.gov.br/jspui/bitstream/123456789/17858/3/Diretrizes%20para%20o%20Rastreamento%20do%20c%c3%a2ncer%20do%20colo%20do%20%c3%batero_2025_completo.pdf"),
        ("INCA, Nota Técnica nº 1/2024 sobre o teste DNA-HPV",
         "https://www.inca.gov.br/sites/ufu.sti.inca.local/files/media/document/nota_informativa_sobre_dna-hpv.pdf"),
        ("OMS, Cervical cancer (fact sheet)", "https://www.who.int/news-room/fact-sheets/detail/cervical-cancer"),
        ("Agência Brasil, 26/03/2025: SUS vai substituir o Papanicolau por exame mais sensível",
         "https://agenciabrasil.ebc.com.br/saude/noticia/2025-03/sus-vai-substituir-papanicolau-por-exame-mais-sensivel-ainda-este-ano"),
        ("Canaltech, 02/04/2024: vacina do HPV em dose única no SUS",
         "https://canaltech.com.br/saude/vacina-do-hpv-vai-ser-aplicada-em-dose-unica-no-sus-284271/"),
    ]
    lista_fontes = "".join(f'<li><a href="{u}" target="_blank" rel="noopener">{e(t)}</a></li>' for t, u in fontes)
    mitos = [
        ("Já tomei a vacina, então não preciso fazer o exame.",
         "Falso. A vacina não protege contra todos os tipos de HPV. O rastreamento continua necessário, mesmo para quem se vacinou."),
        ("Só preciso fazer o exame se eu sentir alguma coisa.",
         "Falso. O exame serve justamente para achar alterações antes de qualquer sintoma. Com sintomas, procure atendimento sem esperar a data do exame."),
        ("Resultado alterado ou HPV positivo quer dizer que tenho câncer.",
         "Falso. Na maioria das vezes significa apenas que é preciso acompanhar ou investigar melhor. Lesões iniciais têm tratamento simples."),
        ("Tenho vergonha ou medo do exame.",
         "É comum sentir isso. A coleta é rápida e você pode pedir à profissional que explique cada etapa. Em algumas situações de difícil acesso, a autocoleta pode ser oferecida pelo serviço."),
    ]
    mitos_html = "".join(f"<details class='mito'><summary>{e(a)}</summary><p>{e(b)}</p></details>" for a, b in mitos)
    return f"""
<section id="inicio" class="secao" data-secao-conteudo aria-labelledby="h-inicio">
  <h2 id="h-inicio" class="titulo-secao">Rastreio do câncer do colo do útero: o exame que evita o câncer</h2>
  <div class="colunas colunas--texto">
    <div class="prosa">
      <p>O câncer do colo do útero é, em quase todos os casos, consequência de uma infecção persistente por tipos de HPV de alto risco. Ele evolui devagar: antes de virar câncer, passa por alterações que o exame preventivo consegue encontrar. Por isso é um dos poucos cânceres que podem ser evitados com vacina e rastreamento.</p>
      <p>Segundo a Estimativa 2026–2028 do INCA, ele está entre os cinco cânceres mais frequentes em mulheres no Brasil e permanece entre os mais incidentes nas regiões Norte e Nordeste, onde o acesso à vacinação e ao rastreamento ainda é desigual.</p>
    </div>
    <div class="pilha">
      <div class="cartao"><div class="sobrancelha">Em uma frase</div>
        <p class="fraco corpo-cartao">Vacinar aos 9–14 anos e fazer o exame de rastreio no tempo certo evita a maior parte dos casos e permite tratar cedo os que surgem.</p></div>
      <div class="cartao"><div class="sobrancelha">Onde fazer</div>
        <p class="fraco corpo-cartao">Na Unidade Básica de Saúde (UBS) mais próxima de você. O exame e a vacina são gratuitos no SUS.</p></div>
    </div>
  </div>

  <h2 class="titulo-secao espaco">Três formas de se proteger</h2>
  <div class="grade-3 grade-3--texto">
    <div class="cartao"><h4>1. Vacina contra o HPV</h4>
      <p class="fraco corpo-cartao">No SUS, meninas e meninos de 9 a 14 anos recebem dose única, de preferência antes do início da vida sexual. Quem tem até 19 anos e ainda não se vacinou pode receber conforme o estoque da região: pergunte na UBS. Pessoas imunocomprometidas seguem esquema com mais doses.</p></div>
    <div class="cartao"><h4>2. Exame de rastreio</h4>
      <p class="fraco corpo-cartao">As diretrizes do INCA (2025) indicam, para mulheres de 25 a 64 anos, o teste de DNA-HPV, repetido a cada 5 anos quando o resultado é negativo. A troca do Papanicolau pelo DNA-HPV é gradual e varia por cidade: onde o Papanicolau ainda é o exame ofertado, o intervalo usual era de 3 anos após dois exames anuais normais. Confirme na UBS qual exame e qual intervalo valem para você.</p></div>
    <div class="cartao"><h4>3. Hábitos e cuidados</h4>
      <p class="fraco corpo-cartao">O preservativo reduz o risco de infecção pelo HPV, embora não elimine totalmente. Não fumar diminui o risco entre mulheres com HPV. Infecções sexualmente transmissíveis devem ser investigadas e tratadas.</p></div>
  </div>

  <h2 class="titulo-secao espaco">Quem deve fazer o exame</h2>
  <div class="prosa">
    <ul>
      <li><strong>Mulheres de 25 a 64 anos</strong> que já tiveram vida sexual. Antes dos 25 anos o rastreamento não é recomendado. Quem nunca fez o exame ou está atrasada deve ser a prioridade.</li>
      <li><strong>Mulheres vacinadas</strong> também: a vacinação não dispensa o rastreamento.</li>
      <li><strong>Acima de 60 anos:</strong> se o último teste foi negativo, o rastreamento pode ser encerrado; quem nunca fez deve fazer.</li>
      <li><strong>HIV/aids ou imunossupressão:</strong> acompanhamento diferenciado, com intervalo de 3 anos após teste negativo.</li>
      <li><strong>Gestantes</strong> podem fazer o exame no pré-natal. <strong>Após histerectomia</strong>, depende do motivo da cirurgia: converse com a equipe de saúde.</li>
    </ul>
  </div>

  <h2 class="titulo-secao espaco">Recebi um resultado alterado. E agora?</h2>
  <div class="prosa">
    <p>Alteração não significa câncer. Nas diretrizes de 2025, quando o teste de DNA-HPV é positivo para os tipos 16 ou 18, a mulher é encaminhada direto para a colposcopia. Para os outros tipos de alto risco, faz-se uma citologia na mesma amostra para decidir entre colposcopia e novo teste em 12 meses. Siga a orientação da equipe e não deixe de voltar nas datas marcadas: é o seguimento que impede a progressão.</p>
  </div>
  <div class="aviso aviso--atencao"><strong>Sinais que merecem avaliação, mesmo com exame em dia:</strong> sangramento após a relação sexual, entre as menstruações ou depois da menopausa; corrimento diferente do habitual ou com mau cheiro; dor na pelve ou durante a relação. Têm várias causas possíveis, mas devem ser avaliados por um profissional.</div>

  <h2 class="titulo-secao espaco">Mitos e verdades</h2>
  <div class="mitos">{mitos_html}</div>

  <h2 class="titulo-secao espaco">Como esta plataforma se encaixa</h2>
  <div class="prosa">
    <p>Depois que o exame é coletado, as células são examinadas ao microscópio por profissionais de citopatologia e classificadas pelo Sistema Bethesda. Esta plataforma é um projeto de pesquisa da UFAM e da SEMSA Manaus que explora o uso de inteligência artificial como apoio a essa leitura. Ela <strong>não emite laudo</strong> e não substitui o exame nem a avaliação profissional.</p>
    <p class="botoes--linha acoes-inicio">
      <a class="btn btn--grande" href="#analise">Experimentar a análise de imagem</a>
      <a class="btn btn--sec" href="#bethesda">Entender o Sistema Bethesda</a>
    </p>
  </div>
  <div class="aviso aviso--info">Conteúdo informativo, sem caráter de consulta médica. As orientações seguem as diretrizes do INCA, mas a implantação do teste de DNA-HPV varia por município: em caso de dúvida, procure a UBS.</div>
  <p class="legenda">Fontes:</p>
  <ul class="legenda lista-fontes">{lista_fontes}</ul>
</section>
"""


def sec_analise():
    passos = [("Passo 1", "Carregue um campo", "Use o seletor acima ou abra uma das imagens de teste."),
              ("Passo 2", "Execute a análise", "A rede classifica a imagem nas quatro categorias do Sistema Bethesda."),
              ("Passo 3", "Leia o mapa de atenção", "O Grad-CAM mostra quais regiões pesaram na decisão.")]
    cartoes = "".join(f'<div class="cartao"><div class="sobrancelha">{a}</div><b>{b}</b><br><span class="fraco">{c}</span></div>'
                      for a, b, c in passos)
    aug = "".join(f"<tr><td><code>{e(a)}</code></td><td>{e(b)}</td></tr>" for a, b in AUMENTOS_TREINO)
    return f"""
<section id="analise" class="secao" data-secao-conteudo aria-labelledby="h-analise">
  <h2 id="h-analise" class="titulo-secao">Análise de campo citológico</h2>
  <div class="colunas colunas--entrada">
    <div>
      <div class="zona" id="zona">
        <img class="zona__previa" id="previa-mini" alt="" hidden>
        <p class="zona__rotulo"><strong>Carregue a lâmina digitalizada</strong></p>
        <p class="pequeno">Arraste uma imagem para cá ou escolha abaixo. Um campo digitalizado por vez. <strong>Não envie imagens de pacientes.</strong></p>
        <div class="botoes">
          <label class="btn" for="arquivo">Escolher imagem</label>
          <label class="btn btn--sec" for="camera">Usar a câmera</label>
        </div>
        <input class="so-leitor" type="file" id="arquivo" accept="image/*">
        <input class="so-leitor" type="file" id="camera" accept="image/*" capture="environment">
      </div>
    </div>
    <div class="pilha">
      <label class="caixa-marcar">
        <input type="checkbox" id="robustez" checked>
        <span><strong>Verificar robustez nas oito orientações</strong>
          <span class="pequeno bloco">Analisa o campo girado e espelhado. Revela se a rede muda de opinião só por causa da orientação. Deixa a análise mais lenta.</span></span>
      </label>
      <p class="pequeno">Sem uma imagem própria? A seção <a href="#exemplos" data-ir="exemplos"><strong>Imagens de teste</strong></a> traz campos do IARC prontos para analisar.</p>
      <div id="modelo-estado" class="modelo-estado">
        <p id="modelo-msg" class="pequeno" role="status" aria-live="polite">Consultando o modelo publicado…</p>
        <progress id="modelo-prog" max="100" hidden aria-label="Progresso do download do modelo"></progress>
        <button class="btn btn--sec btn--mini" id="modelo-remover" type="button" hidden>Remover o modelo deste navegador</button>
      </div>
    </div>
  </div>
  <hr>
  <div id="painel-imagem" class="colunas colunas--imagem" hidden>
    <figure class="figura"><img id="previa" alt=""><figcaption id="previa-legenda"></figcaption></figure>
    <div class="pilha">
      <h3>Lâmina carregada</h3>
      <p id="previa-dim"></p>
      <p class="pequeno" id="previa-desc" hidden></p>
      <p class="pequeno" id="previa-ref" hidden></p>
      <div class="botoes botoes--linha">
        <button class="btn btn--grande" id="analisar" type="button">Executar análise citológica</button>
        <button class="btn btn--sec" id="limpar" type="button">Limpar</button>
      </div>
      <progress id="analise-prog" max="8" hidden aria-label="Progresso da análise"></progress>
      <p class="pequeno" id="analise-msg" role="status" aria-live="polite" hidden></p>
    </div>
  </div>
  <div id="resultado" aria-live="polite" hidden></div>
  <div id="boas-vindas">
    <hr>
    <h2 class="titulo-secao">Como funciona</h2>
    <div class="grade-3">{cartoes}</div>
    <h2 class="titulo-secao espaco">Escala de gravidade do Sistema Bethesda</h2>
    {regua()}
    <p class="legenda">A seção <a href="#bethesda" data-ir="bethesda"><strong>Sistema Bethesda</strong></a> explica cada categoria em detalhe: o que significa, o que se vê ao microscópio e que conduta costuma decorrer dela.</p>
    <div class="colunas colunas--duas espaco">
      <div>
        <h3>Por que rastrear</h3>
        <p>O câncer do colo do útero é dos mais evitáveis, e ainda assim segue entre as principais causas de morte por câncer em mulheres no Amazonas. A citologia em meio líquido detecta lesões precursoras anos antes da malignização. O gargalo está no volume de lâminas por citopatologista. Uma triagem automatizada pode ordenar a fila, colocando os casos suspeitos na frente.</p>
      </div>
      <div>
        <h3>Como o modelo foi treinado</h3>
        <p>ConvNeXt Large com <em>transfer learning</em> a partir do ImageNet, ajustado em dois conjuntos públicos de citologia em meio líquido. Aumentos aplicados no treino:</p>
        <div class="tabela-rolagem"><table class="tabela"><thead><tr><th>Transformação</th><th>Motivo</th></tr></thead><tbody>{aug}</tbody></table></div>
      </div>
    </div>
  </div>
</section>"""


def sec_bethesda():
    intro = "".join(f"<p>{e(p)}</p>" for p in BETHESDA_INTRO)
    blocos = []
    for s in ORDEM_GRAVIDADE:
        i, d = CLASS_INFO[s], BETHESDA_DETALHE[s]
        blocos.append(f"""
    <article class="categoria">
      <div class="cartao" style="border-left:5px solid {i['cor']}">
        <div class="cat-sigla" style="color:{i['cor']}">{s}</div>
        <div class="cat-nome">{e(i['nome'])}</div>
        <div class="cat-trad">{e(d['traducao'])}</div>
      </div>
      <div class="grade-3 grade-3--texto">
        <div><h4>O que significa</h4><p class="fraco">{e(d['o_que_e'])}</p></div>
        <div><h4>O que se vê ao microscópio</h4><p class="fraco">{e(d['morfologia'])}</p></div>
        <div><h4>Conduta habitual</h4><p class="fraco">{e(d['conduta'])}</p><p class="legenda">{e(d['frequencia'])}</p></div>
      </div>
      <hr>
    </article>""")
    alem = "".join(f"<li><strong>{e(a)}</strong>: {e(b)}</li>" for a, b in BETHESDA_ALEM_ITENS)
    return f"""
<section id="bethesda" class="secao" data-secao-conteudo aria-labelledby="h-bethesda">
  <h2 id="h-bethesda" class="titulo-secao">O que é o Sistema Bethesda</h2>
  <div class="colunas colunas--texto">
    <div class="prosa">{intro}</div>
    <div class="pilha">
      <div class="cartao"><div class="sobrancelha">Em uma frase</div>
        <p class="fraco corpo-cartao">Um vocabulário comum que transforma o que se vê ao microscópio numa categoria que orienta conduta clínica, e permite comparar resultados entre laboratórios, cidades e países.</p></div>
      <div class="cartao"><div class="sobrancelha">Por que a ordem importa</div>
        <p class="fraco corpo-cartao">As categorias formam uma escala. Um erro entre vizinhas costuma adiar um seguimento; um erro entre extremos pode perder um câncer. É por isso que a matriz de confusão diz mais que a acurácia isolada.</p></div>
    </div>
  </div>
  {regua()}
  <hr>
  <h2 class="titulo-secao">As quatro categorias em detalhe</h2>
  <p class="legenda">Cada bloco reúne o que a categoria significa, o que se vê ao microscópio e o que costuma decorrer dela em termos de conduta.</p>
  {"".join(blocos)}
  <h2 class="titulo-secao">O que este modelo não classifica</h2>
  <div class="prosa"><p>{e(BETHESDA_ALEM_INTRO)}</p><ul>{alem}</ul><p>{BETHESDA_ALEM_FIM}</p></div>
  <div class="aviso aviso--info">As descrições acima são material didático desta plataforma. Não substituem as definições oficiais do Sistema Bethesda nem as diretrizes do Ministério da Saúde e do INCA para conduta clínica.</div>
</section>"""


def sec_validacao():
    m = METRICAS
    ic = m["ic95"]
    linhas = []
    for s in ORDEM_GRAVIDADE:
        d = m["por_classe"][s]
        linhas.append(f'<tr><td><b style="color:{CLASS_INFO[s]["cor"]}">{s}</b></td><td>{e(CLASS_INFO[s]["nome"])}</td>'
                      f'<td>{pct(d["precisao"])}</td><td>{pct(d["recall"])}</td><td>{pct(d["f1"])}</td><td>{d["suporte"]}</td></tr>')
    erros = m["n_imagens"] - m["acertos"]
    k = f'{m["kappa"]:.4f}'.replace(".", ",")
    return f"""
<section id="validacao" class="secao" data-secao-conteudo aria-labelledby="h-validacao">
  <h2 id="h-validacao" class="titulo-secao">Validação do algoritmo</h2>
  <p class="legenda">{e(m['protocolo'])} · {e(m['conjunto'])} · arquitetura: <code>ConvNeXt Large</code></p>
  <p>O modelo foi avaliado por validação cruzada estratificada em cinco partições, somando <strong>{milhar(m['n_imagens'])} imagens</strong>. Cada imagem foi classificada exatamente uma vez, por um modelo que não a viu no treino. É um protocolo mais exigente que uma única divisão treino/validação, porque toda a base entra na avaliação e o resultado não depende do sorteio de uma partição específica.</p>
  <div class="grade-4 espaco">
    {cartao_numero("Acurácia global", pct(m['acuracia'], 2), f"IC 95%: {pct(ic[0])} – {pct(ic[1])}", "#0E6B7B")}
    {cartao_numero("F1 macro", pct(m['f1_macro'], 2))}
    {cartao_numero("Precisão macro", pct(m['precisao_macro'], 2))}
    {cartao_numero("Recall macro", pct(m['recall_macro'], 2))}
  </div>
  <div class="grade-3 espaco">
    {cartao_numero("Kappa de Cohen", k, "Concordância além do acaso")}
    {cartao_numero("Imagens avaliadas", milhar(m['n_imagens']), f"{milhar(m['acertos'])} classificações corretas")}
    {cartao_numero("Erros", str(erros), "Nenhum deles subestimou um caso grave")}
  </div>
  <hr>
  <h3>Matriz de confusão global</h3>
  <div class="colunas colunas--texto">
    <div class="matriz-caixa">{tabela_matriz()}</div>
    <div class="prosa">
      <p>Cada linha é a classe de referência e cada coluna, a decisão da rede. A diagonal são os acertos.</p>
      <p><strong>O que mais importa não é o total de erros, é a direção deles.</strong> Numa ferramenta de triagem, confundir LSIL com NILM adia um seguimento; confundir carcinoma com NILM perde um caso.</p>
      <p>Nesta validação, <strong>nenhuma das 312 imagens de HSIL ou carcinoma foi classificada como normal</strong>. O erro mais frequente foi o oposto, sete carcinomas lidos como HSIL, o que mantém a paciente na via de investigação. Seis campos de LSIL foram para NILM: é o achado que merece mais atenção, porque adia o seguimento.</p>
    </div>
  </div>
  <details class="expansor"><summary>Figura original do experimento</summary>
    <img src="img/matriz_confusao.png" alt="Matriz de confusão original do experimento, com rótulos Carcinoma, HSIL, LSIL e Normal" loading="lazy">
    <p class="legenda">Rótulos como saíram do treinamento: Carcinoma, HSIL, LSIL, Normal. Ordem alfabética das pastas, que é a mesma dos índices do modelo.</p></details>
  <hr>
  <h3>Desempenho por classe</h3>
  <div class="tabela-rolagem"><table class="tabela"><thead><tr><th>Classe</th><th>Descrição</th><th>Precisão</th><th>Recall (sensibilidade)</th><th>F1</th><th>Imagens</th></tr></thead><tbody>{"".join(linhas)}</tbody></table></div>
  <p class="legenda">Em rastreio, o recall de HSIL e SCC é a métrica decisiva: mede quantos casos graves escapariam. HSIL alcançou 99,1% e carcinoma 92,7%, e as sete imagens de carcinoma não reconhecidas foram classificadas como HSIL, ou seja, permaneceram sinalizadas como alteração de alto grau.</p>
  <hr>
  <h3>Curvas de treinamento</h3>
  <img class="figura-larga" src="img/historico_treinamento.png" alt="Curvas de acurácia e perda de treino e validação por época" loading="lazy">
  <p class="legenda">A acurácia de treino satura cedo e a de validação oscila, comportamento esperado num conjunto curado. O checkpoint publicado corresponde ao melhor ponto de validação, não à última época.</p>
  <hr>
  <h3>O que estes números ainda não dizem</h3>
  <ul class="prosa">
    <li>A validação é <strong>interna aos dois conjuntos públicos usados no treino</strong>. Desempenho em lâminas do Laboratório Sebastião Marinho, com outro scanner, outra coloração e outra prevalência, ainda não foi medido. É justamente o objeto da parceria com a SEMSA Manaus.</li>
    <li>A prevalência aqui não é a de um programa de rastreio. No acervo, 42% dos campos são alterados; na rotina, a proporção de exames alterados é bem menor. O valor preditivo positivo em campo será menor que a precisão medida.</li>
    <li>Métricas por imagem não equivalem a métricas por lâmina ou por paciente. Uma lâmina tem centenas de campos, e a decisão clínica é sobre a lâmina.</li>
    <li>O modelo só conhece quatro categorias. Atipias de significado indeterminado (ASC-US, ASC-H), lesões glandulares (AGC, AIS) e amostras insatisfatórias não têm classe própria e serão forçadas para uma das quatro.</li>
    <li>Não há comparação com dupla leitura humana, que é o padrão de referência real.</li>
    <li>Sem aprovação de CEP, nenhum dado de paciente pode ser processado nesta ferramenta.</li>
  </ul>
</section>"""


def sec_exemplos():
    cards = []
    for ex in EXEMPLOS:
        etq = ""
        if ex["referencia"]:
            i = CLASS_INFO[ex["referencia"]]
            etq = f'<span class="etiqueta" style="background:{i["cor"]}1A;color:{i["cor"]}">{ex["referencia"]} · referência do atlas</span>'
        arq = ex["arquivo"]
        cards.append(f"""
    <article class="exemplo">
      <img src="exemplos/{arq}" alt="{e(ex['titulo'])}: {e(ex['descricao'])}" loading="lazy">
      {etq}
      <p class="exemplo-titulo"><strong>{e(ex['titulo'])}</strong> · <code>{arq}</code></p>
      <p class="legenda">{e(ex['descricao'])}</p>
      <div class="botoes botoes--linha">
        <button class="btn" type="button" data-analisar="{arq}">Analisar</button>
        <a class="btn btn--sec" href="exemplos/{arq}" download="{arq}">Baixar</a>
      </div>
    </article>""")
    return f"""
<section id="exemplos" class="secao" data-secao-conteudo aria-labelledby="h-exemplos">
  <h2 id="h-exemplos" class="titulo-secao">Imagens de teste</h2>
  <p>Campos do IARC Digital Atlas que acompanham a plataforma. Clique em <strong>Analisar</strong> para enviá-los direto ao modelo: a navegação salta para a seção Análise e o resultado aparece assim que o modelo estiver pronto.</p>
  <div class="grade-2 espaco">{"".join(cards)}</div>
</section>"""


def sec_equipe():
    pessoas = "".join(f'<div class="pessoa"><div class="nome">{e(p["nome"])}</div><div class="papel">{e(p["papel"])}</div><div class="vinculo">{e(p["vinculo"])}</div></div>' for p in EQUIPE)
    insts = "".join(f"""
    <div class="inst">
      <img src="{i['logo']}" alt="Logotipo {e(i['sigla'])}" loading="lazy">
      <div class="cartao"><span class="etiqueta">{e(i['sigla'])}</span>
        <div class="inst-nome">{e(i['nome'])}</div><div class="fraco">{e(i['papel'])}</div></div>
    </div>""" for i in INSTITUICOES)
    return f"""
<section id="equipe" class="secao" data-secao-conteudo aria-labelledby="h-equipe">
  <h2 id="h-equipe" class="titulo-secao">Equipe</h2>
  <div class="grade-2">{pessoas}</div>
  <hr>
  <h2 class="titulo-secao">Instituições parceiras</h2>
  {insts}
  <hr>
  <div class="colunas colunas--duas">
    <div><h3>Ficha técnica</h3>
      <p><strong>Arquitetura</strong> · <code>ConvNeXt Large</code>, <em>transfer learning</em> do ImageNet<br>
      <strong>Interpretabilidade</strong> · Grad-CAM sobre o último bloco convolucional<br>
      <strong>Dados</strong> · IARC Digital Atlas e Liquid Based Cytology Pap Smear (Mendeley Data). Detalhes na seção Dados e créditos<br>
      <strong>Entrada</strong> · campo digitalizado de citologia em meio líquido, RGB<br>
      <strong>Validação</strong> · {e(METRICAS['protocolo'])}, {milhar(METRICAS['n_imagens'])} imagens</p></div>
    <div><h3>Situação regulatória</h3>
      <p>Todo o desenvolvimento usou <strong>apenas o acervo público IARC</strong>. Nenhuma lâmina, imagem ou dado de paciente atendido pela rede municipal foi acessado.</p>
      <ul>
        <li>O projeto <strong>ainda não foi aprovado por Comitê de Ética em Pesquisa</strong>. A submissão ao CEP é pré-requisito da fase seguinte.</li>
        <li>Não há registro na ANVISA como software como dispositivo médico (SaMD), exigido para uso assistencial.</li>
        <li>A responsabilidade diagnóstica permanece integralmente com o citopatologista.</li>
      </ul></div>
  </div>
  <hr>
  <h3>Próximos passos</h3>
  <ol class="prosa">
    <li>Submissão do protocolo ao CEP/UFAM.</li>
    <li>Validação externa independente com lâminas do Laboratório Sebastião Marinho, sob supervisão das citopatologistas da SEMSA Manaus, medindo desempenho por lâmina e por paciente.</li>
    <li>Concordância entre a IA e a dupla leitura humana.</li>
    <li>Estudo de impacto no tempo de fila e na taxa de detecção de HSIL ou pior.</li>
  </ol>
</section>"""


def sec_dados():
    ds = []
    for d in DATASETS:
        ref = f'<p class="citacao-ds"><strong>Como citar:</strong> {d["referencia"]}</p>' if d["referencia"] else ""
        ds.append(f"""
    <div class="cartao"><span class="etiqueta">{e(d['sigla'])}</span>
      <div class="ds-nome">{e(d['nome'])}</div><div class="fraco pequeno">{e(d['instituicao'])}</div>
      <p class="fraco">{e(d['descricao'])}</p>
      <p class="pequeno"><a href="{d['url']}" target="_blank" rel="noopener">{d['url']}</a></p></div>{ref}""")
    citacao = ("OLIVEIRA, M. D. L.; MARTINS, T. R.; NAKAMURA, F.; OLIVEIRA, F. G. "
               "Sistema de apoio ao diagnóstico citológico em meio líquido assistido por "
               "inteligência artificial. Faculdade de Ciências Farmacêuticas e Instituto "
               "de Computação, Universidade Federal do Amazonas, em parceria com a "
               f"Secretaria Municipal de Saúde de Manaus, {ANO_DESENVOLVIMENTO}.")
    return f"""
<section id="dados" class="secao" data-secao-conteudo aria-labelledby="h-dados">
  <h2 id="h-dados" class="titulo-secao">Conjuntos de dados de treinamento</h2>
  <p>O modelo foi treinado e validado sobre dois acervos públicos de citologia em meio líquido. Nenhuma imagem de paciente atendido pela rede municipal foi utilizada em qualquer etapa.</p>
  <div class="pilha espaco">{"".join(ds)}</div>
  <div class="aviso aviso--info">A combinação dos dois acervos foi o que permitiu chegar às 1096 imagens da validação cruzada. Ela também amplia a variedade de preparo e coloração vista no treino, o que tende a ajudar na generalização. Ainda assim, continuam sendo acervos curados para ensino e pesquisa, com prevalência de lesões bem diferente da rotina de um programa de rastreio.</div>
  <hr>
  <h2 class="titulo-secao">Ferramentas de apoio ao desenvolvimento</h2>
  <p>{FERRAMENTAS_APOIO}</p>
  <p class="legenda">Registramos esta informação por transparência científica, do mesmo modo que se declara o uso de qualquer outra ferramenta computacional.</p>
  <hr>
  <h2 class="titulo-secao">Bibliotecas e tecnologias</h2>
  <div class="colunas colunas--duas">
    <div><p><strong>Modelo e treinamento</strong><br>PyTorch, torchvision (ConvNeXt Large), Albumentations, scikit-learn (validação cruzada estratificada), NumPy</p>
      <p><strong>Interpretabilidade</strong><br>Grad-CAM implementado sobre o último bloco convolucional; no navegador e no aplicativo, calculado em forma fechada dentro do próprio grafo ONNX</p></div>
    <div><p><strong>Este site</strong><br>HTML, CSS e JavaScript sem frameworks; ONNX Runtime Web (WebAssembly) para a inferência no navegador, com o modelo quantizado em 8 bits; fontes Inter e IBM Plex Mono (licença OFL), servidas pelo próprio site. Catálogo: Firebase Authentication e Firestore</p>
      <p><strong>Versão anterior e aplicativo</strong><br>Streamlit, OpenCV, Matplotlib, pandas, Pillow. Aplicativo Android/iOS: Flutter, ONNX Runtime para inferência local no aparelho</p></div>
  </div>
  <hr>
  <h2 class="titulo-secao">Como citar esta plataforma</h2>
  <pre class="citacao">{e(citacao)}</pre>
  <p class="legenda">Trabalho em andamento, ainda sem publicação revisada por pares e sem aprovação de Comitê de Ética em Pesquisa.</p>
</section>"""


# -------------------------------------------------------------------- escrita

def dados_js():
    info = {s: {"nome": CLASS_INFO[s]["nome"], "cor": CLASS_INFO[s]["cor"], "gravidade": CLASS_INFO[s]["gravidade"],
                "resumo": CLASS_INFO[s]["resumo"]} for s in CLASS_INFO}
    ex = [{"arquivo": f"exemplos/{x['arquivo']}", "titulo": x["titulo"], "referencia": x["referencia"],
           "descricao": x["descricao"]} for x in EXEMPLOS]
    return ("// GERADO por ferramentas/gerar_site.py a partir de ferramentas/conteudo_site.py. Não edite à mão.\n"
            f"export const ORDEM_DO_MODELO = {json.dumps(CLASSES)};\n"
            f"export const ORDEM_GRAVIDADE = {json.dumps(ORDEM_GRAVIDADE)};\n"
            f"export const LIMIAR_CONFIANCA = {LIMIAR_CONFIANCA};\n"
            f"export const LIMIAR_CONCORDANCIA = {LIMIAR_CONCORDANCIA};\n"
            f"export const CLASS_INFO = {json.dumps(info, ensure_ascii=False, indent=2)};\n"
            f"export const EXEMPLOS = {json.dumps(ex, ensure_ascii=False, indent=2)};\n")


def main():
    corpo_index = "\n".join([sec_inicio(), sec_analise(), sec_bethesda(), sec_validacao(), sec_exemplos(), sec_equipe(), sec_dados()])
    index = pagina(SEO_TITULO, corpo_index, "inicio", True,
                   '  <script type="module" src="js/app.js?v=3"></script>', extra_head=meta_seo())
    (DOCS / "index.html").write_text(index, encoding="utf-8")

    cat = (Path(__file__).parent / "catalogo_corpo.html").read_text(encoding="utf-8")
    catalogo = pagina("Catálogo de lâminas | " + SEO_TITULO, cat, "catalogo", False,
                      '  <script type="module" src="js/catalogo.js?v=5"></script>',
                      extra_head=f'\n  <meta name="description" content="Catálogo eletrônico de lâminas de citopatologia cervical do Laboratório Prof. Sebastião Marinho (UFAM / SEMSA). Cadastro, consulta e avaliação cega, sem necessidade de conta Google.">')
    (DOCS / "catalogo.html").write_text(catalogo, encoding="utf-8")

    (DOCS / "predicao.html").write_text(
        '<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><title>Análise de citologia</title>'
        '<meta http-equiv="refresh" content="0; url=index.html#analise"><link rel="canonical" href="index.html">'
        '<meta name="robots" content="noindex"></head><body><p>Esta página mudou: '
        '<a href="index.html#analise">abrir a análise de citologia</a>.</p>'
        '<script>location.replace("index.html#analise")</script></body></html>', encoding="utf-8")
    (DOCS / "js" / "dados.js").write_text(dados_js(), encoding="utf-8")
    print("gerado:", [p.name for p in (DOCS / "index.html", DOCS / "catalogo.html", DOCS / "predicao.html")])


if __name__ == "__main__":
    main()
