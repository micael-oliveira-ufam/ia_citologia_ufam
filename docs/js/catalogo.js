// Módulo "Catálogo de lâminas" — acervo do CitoAtlas com identidade anônima do Firebase.
//
// Protocolo cego preservado: a lista e o visor nunca mostram classificação,
// descrição, observação nem avaliador de lâminas de OUTRAS pessoas. O criador só vê
// o que ele mesmo cadastrou (no formulário de edição) e cada pessoa só vê a
// própria avaliação. Todo texto vindo do banco entra na página por textContent.

import { CATALOGO, VALIDADOR_EXTERNO } from './config.js?v=5';
import { $, el, limpar, formatarData, idPublico, urlImagemSegura, iniciarPagina } from './comum.js';
import {
  listarLaminas, contarVotos, minhasAvaliacoes, proximoId, criarLamina, atualizarLamina,
  excluirLamina, salvarAvaliacao, salvarComplementoAvaliacao, salvarPredicao, resumoPredicoes, entrar, uidAtual, ErroFirestore,
  resumoAvaliacoes, minhasPredicoes,
} from './firestore.js?v=7';
import { comprimirImagem } from './imagem.js?v=5';
import {
  situacaoModelo, carregarModelo, modeloPronto, classificarLote, blobDeDataUrl, csv, baixarTexto,
} from './ia.js?v=5';
import { CATEGORIAS } from './bethesda.js';
import { CLASS_INFO } from './dados.js';

const CLASSES = [
  ['NILM', 'Negativo para Lesão ou Malignidade'],
  ['ASC-US', 'Células Escamosas Atípicas de Significado Indeterminado'],
  ['ASC-H', 'Células Escamosas Atípicas (Não se pode excluir HSIL)'],
  ['LSIL', 'Lesão Intraepitelial Escamosa de Baixo Grau'],
  ['HSIL', 'Lesão Intraepitelial Escamosa de Alto Grau'],
  ['SCC', 'Carcinoma de Células Escamosas'],
  ['Adenocarcinoma', 'Adenocarcinoma'],
  ['Outro', 'Outro preenchimento / Outras alterações'],
];
const SIGLAS = new Set(CLASSES.map((c) => c[0]));

iniciarPagina('catalogo');

const PREF_CHAVE = 'citoilumina.exibir';
const estado = {
  laminas: [],
  proximo: null,
  fim: false,
  carregando: false,
  votos: null, // Map(slideId -> n) ou null se indisponível
  votosCompleto: true,
  erro: null,
  minhas: new Map(), // slideId -> minha avaliação
  minhasIA: new Map(), // slideId -> minha predição da IA salva no banco
  selecionadas: new Set(),
  revelarIA: false, // o usuário aceitou ver a IA antes de avaliar (pode influenciar a avaliação cega)
  mostrarIA: lerPrefs().ia === true, // opt-in: classificações humanas e predições da IA nos cartões
  mostrarVotos: lerPrefs().votos === true, // opt-in: contagem de votos nos cartões
  resumo: null, // Map(slideId -> avaliações) quando já carregado
  resumoCarregando: false,
  resumoErro: false,
};

function lerPrefs() {
  try { return JSON.parse(localStorage.getItem(PREF_CHAVE) || '{}') || {}; } catch (e) { return {}; }
}
function salvarPrefs() {
  try { localStorage.setItem(PREF_CHAVE, JSON.stringify({ ia: estado.mostrarIA, votos: estado.mostrarVotos })); } catch (e) { /* navegador sem armazenamento */ }
}

/** Consenso humano: classe com maioria das avaliações, com no mínimo `votosParaConsenso`. */
function consensoHumano(lista) {
  const votos = (lista || []).map((x) => x.classificacao).filter((c) => SIGLAS.has(c));
  if (votos.length < CATALOGO.votosParaConsenso) return null;
  const cont = {};
  for (const v of votos) cont[v] = (cont[v] || 0) + 1;
  const [sigla, n] = Object.entries(cont).sort((a, b) => b[1] - a[1])[0];
  return n > votos.length / 2 ? { sigla, n, total: votos.length } : null;
}

/** Classe da IA para esta lâmina: a mais frequente entre as predições registradas (ou null). */
function siglaIA(lista) {
  const cont = new Map();
  for (const x of lista || []) if (x.ia) cont.set(x.ia, (cont.get(x.ia) || 0) + 1);
  if (!cont.size) return null;
  return [...cont.entries()].sort((a, b) => b[1] - a[1])[0][0];
}

/** Texto da predição da IA para o cartão: classe majoritária, confiança e nº de análises. */
function iaRegistrada(lista) {
  const sigla = siglaIA(lista);
  if (!sigla) return null;
  const doGrupo = (lista || []).filter((x) => x.ia);
  const r = doGrupo.find((x) => x.ia === sigla);
  const conf = typeof r?.confianca === 'number' ? ` (${Math.round(r.confianca * 100)}%)` : '';
  const n = doGrupo.filter((x) => x.ia === sigla).length;
  const analises = doGrupo.length > 1 ? ` · ${n} de ${doGrupo.length} análises` : '';
  return `${sigla}${conf}${analises}`;
}

/** Acrescenta uma predição recém-gravada ao resumo já carregado, sem recarregar a página. */
function adicionarIAResumo(slideId, sigla, confianca) {
  if (!estado.resumo) return;
  const lista = estado.resumo.get(slideId) || [];
  lista.push({ classificacao: null, ia: sigla, confianca });
  estado.resumo.set(slideId, lista);
}

async function carregarResumo() {
  if (estado.resumoCarregando || estado.resumo) return;
  estado.resumoCarregando = true;
  estado.resumoErro = false;
  renderizar();
  try {
    const [{ mapa }, pred] = await Promise.all([
      resumoAvaliacoes(),
      resumoPredicoes().then((r) => { avisarBanco(''); return r.mapa; }).catch((e) => {
        console.warn('Predições indisponíveis:', e);
        avisarBanco(`Não foi possível ler as predições da IA no banco. ${e.message || ''}`.trim());
        return new Map();
      }),
    ]);
    for (const [id, lista] of pred) {
      const atual = mapa.get(id) || [];
      mapa.set(id, [...atual, ...lista.map((x) => ({ classificacao: null, ia: x.ia, confianca: x.confianca }))]);
    }
    estado.resumo = mapa;
  } catch (e) {
    estado.resumoErro = true;
    console.warn('Classificações indisponíveis:', e);
  } finally {
    estado.resumoCarregando = false;
    renderizar();
    if (visor.open) desenharResumoVisor();
  }
}

const grade = $('#grade');
const status = $('#status');
const maisBtn = $('#mais');
const todasBtn = $('#todas');
const notaParcial = $('#nota-parcial');
const contagemEl = $('#contagem');
const buscaEl = $('#busca');
const filtroVotosEl = $('#filtro-votos');
const filtroPastaEl = $('#filtro-pasta');
const filtroLesaoEl = $('#filtro-lesao');
const notaLesao = $('#nota-lesao');
const avisoBanco = $('#aviso-banco');
function avisarBanco(msg) { avisoBanco.textContent = msg || ''; avisoBanco.hidden = !msg; }
const notaVotos = $('#nota-votos');

// ---------------------------------------------------------------- dados

async function carregarVotos() {
  try {
    const { contagem, completo } = await contarVotos();
    estado.votos = contagem;
    estado.votosCompleto = completo;
  } catch (e) {
    estado.votos = null;
    console.warn('Contagem de votos indisponível:', e);
  }
  atualizarNotaVotos();
  renderizar();
}

async function carregarMais() {
  if (estado.carregando || estado.fim) return;
  estado.carregando = true;
  estado.erro = null;
  mostrarStatus('carregando');
  maisBtn.hidden = true;
  try {
    const { itens, proximo } = await listarLaminas({
      tamanho: CATALOGO.tamanhoPagina,
      token: estado.proximo,
    });
    for (const it of itens) {
      if (!urlImagemSegura(it.imageUrl)) continue; // documento sem imagem utilizável
      estado.laminas.push(it);
    }
    estado.proximo = proximo;
    estado.fim = !proximo;
  } catch (e) {
    estado.erro = e instanceof ErroFirestore ? e : new ErroFirestore('inesperado', String(e));
  } finally {
    estado.carregando = false;
  }
  renderizar();
}

// ------------------------------------------------------------- exibição

function mostrarStatus(tipo) {
  limpar(status);
  status.hidden = false;
  if (tipo === 'carregando') {
    status.append(el('div', { class: 'estado', role: 'status' },
      el('div', { class: 'girando', 'aria-hidden': 'true' }),
      'Carregando lâminas…'));
  } else if (tipo === 'erro') {
    const e = estado.erro;
    status.append(el('div', { class: 'aviso aviso--erro', role: 'alert' },
      el('strong', {}, 'Não foi possível carregar o catálogo. '),
      e.message, ' ',
      el('button', { class: 'btn btn--sec', type: 'button', onclick: carregarMais }, 'Tentar de novo')));
  } else if (tipo === 'vazio') {
    status.append(el('div', { class: 'estado' }, 'Nenhuma lâmina encontrada com esses filtros.'));
  } else {
    status.hidden = true;
  }
}

function votosDe(l) {
  if (!estado.votos) return null;
  return estado.votos.get(l.id) || 0;
}

function seloVotos(n) {
  if (n == null) return { texto: 'votos indisponíveis', classe: 'votos--0' };
  if (n === 0) return { texto: 'Sem votos ainda', classe: 'votos--0' };
  if (n >= CATALOGO.votosParaConsenso) {
    return { texto: `${n} votos · quórum atingido`, classe: 'votos--completo' };
  }
  return { texto: `${n}/${CATALOGO.votosParaConsenso} avaliações`, classe: 'votos--parcial' };
}

/** Atualiza as opções de pasta com as pastas já carregadas, mantendo a escolha atual. */
function atualizarPastas() {
  const atual = filtroPastaEl.value;
  const pastas = [...new Set(estado.laminas.map((l) => l.pasta).filter(Boolean))].sort((a, b) => a.localeCompare(b, 'pt-BR'));
  limpar(filtroPastaEl);
  filtroPastaEl.append(el('option', { value: 'todas' }, 'Todas as pastas'));
  for (const p of pastas) filtroPastaEl.append(el('option', { value: p }, p));
  filtroPastaEl.append(el('option', { value: '__sem__' }, 'Sem pasta'));
  filtroPastaEl.value = [...filtroPastaEl.options].some((o) => o.value === atual) ? atual : 'todas';
}

/**
 * Filtro por tipo de lesão: "h:<classe>" usa o consenso humano, "i:<classe>" a predição da IA;
 * "__sem__" é a lâmina sem consenso (ou sem análise da IA). Só funciona com as classificações
 * visíveis (protocolo cego): sem a opção marcada o filtro fica desativado.
 */
function passaFiltroLesao(l, valor) {
  if (valor === 'todas') return true;
  if (!estado.resumo) return false; // classificações ainda carregando
  const [fonte, sigla] = valor.split(':');
  const lista = estado.resumo.get(l.id) || [];
  const atual = fonte === 'h' ? (consensoHumano(lista)?.sigla || null) : siglaIA(lista);
  return sigla === '__sem__' ? atual === null : atual === sigla;
}

function atualizarFiltroLesao() {
  const atual = filtroLesaoEl.value;
  limpar(filtroLesaoEl);
  filtroLesaoEl.append(el('option', { value: 'todas' }, 'Todos os tipos'));
  const humano = el('optgroup', { label: 'Pela avaliação humana (consenso)' });
  const ia = el('optgroup', { label: 'Pela predição da IA' });
  for (const [sigla] of CLASSES) {
    const nome = CATEGORIAS[sigla]?.nome;
    const texto = nome ? `${sigla} · ${nome}` : sigla;
    humano.append(el('option', { value: `h:${sigla}` }, texto));
    ia.append(el('option', { value: `i:${sigla}` }, texto));
  }
  humano.append(el('option', { value: 'h:__sem__' }, 'Sem consenso (menos de 3 avaliações ou sem maioria)'));
  ia.append(el('option', { value: 'i:__sem__' }, 'Sem análise da IA'));
  filtroLesaoEl.append(humano, ia);
  filtroLesaoEl.value = [...filtroLesaoEl.options].some((o) => o.value === atual) ? atual : 'todas';
  filtroLesaoEl.disabled = !estado.mostrarIA;
  notaLesao.hidden = estado.mostrarIA;
  if (!estado.mostrarIA) filtroLesaoEl.value = 'todas';
}

/** Carrega o catálogo inteiro. Só é chamada por clique do usuário, nunca automaticamente. */
async function carregarTodas() {
  todasBtn.disabled = true;
  try {
    if (!estado.resumo && !estado.resumoCarregando) await carregarResumo();
    while (!estado.fim && !estado.erro) {
      if (estado.carregando) { await new Promise((r) => setTimeout(r, 150)); continue; }
      await carregarMais();
    }
  } catch (e) { console.warn(e); }
  finally { todasBtn.disabled = false; }
  renderizar();
}

function filtradas() {
  const termo = buscaEl.value.trim().toLowerCase();
  const f = filtroVotosEl.value;
  const pasta = filtroPastaEl.value;
  const lesao = estado.mostrarIA ? filtroLesaoEl.value : 'todas';
  return estado.laminas.filter((l) => {
    if (termo && !idPublico(l.patientId).texto.toLowerCase().includes(termo)) return false;
    if (pasta === '__sem__' && l.pasta) return false;
    if (pasta !== 'todas' && pasta !== '__sem__' && l.pasta !== pasta) return false;
    if (!passaFiltroLesao(l, lesao)) return false;
    const n = votosDe(l);
    if (f !== 'todas' && n != null) {
      if (f === 'sem' && n !== 0) return false;
      if (f === 'andamento' && !(n >= 1 && n < CATALOGO.votosParaConsenso)) return false;
      if (f === 'quorum' && n < CATALOGO.votosParaConsenso) return false;
    }
    return true;
  });
}

/** Cor da classe (pela gravidade, em dados.js) e cor do texto que fica legível sobre ela. */
function corDaClasse(sigla) { return CLASS_INFO[sigla]?.cor || '#4a5a66'; }
/** Texto branco ou escuro, o que tiver maior contraste sobre a cor (WCAG). */
function corSobre(hex) {
  const linear = (c) => { c /= 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
  const n = parseInt(hex.slice(1), 16);
  const luz = 0.2126 * linear((n >> 16) & 255) + 0.7152 * linear((n >> 8) & 255) + 0.0722 * linear(n & 255);
  const escuro = 0.2126 * linear(0x14) + 0.7152 * linear(0x23) + 0.0722 * linear(0x2f); // #14232f
  const contraste = (l2) => (Math.max(luz, l2) + 0.05) / (Math.min(luz, l2) + 0.05);
  return contraste(1) >= contraste(escuro) ? '#ffffff' : '#14232f';
}

/** Selos sobre a foto: classificação humana (consenso) e da IA, na cor da gravidade. */
function chipsClassificacao(l) {
  if (!estado.mostrarIA || !estado.resumo) return [];
  const lista = estado.resumo.get(l.id) || [];
  const cons = consensoHumano(lista);
  const ia = siglaIA(lista);
  const itens = [];
  if (cons) itens.push(['Humano', cons.sigla]);
  if (ia) itens.push(['IA', ia]);
  return itens.map(([rotulo, sigla]) => {
    const cor = corDaClasse(sigla);
    return el('span', {
      class: 'chip-classe',
      style: `background:${cor};color:${corSobre(cor)}`,
      title: `${rotulo}: ${CATEGORIAS[sigla]?.nome || sigla}`,
    }, `${rotulo} · ${sigla}`);
  });
}

/** Mantém os selos do visor atualizados com a lâmina aberta e o resumo carregado. */
function desenharChipsVisor() {
  const caixa = $('#visor-chips');
  limpar(caixa);
  if (atual) caixa.append(...chipsClassificacao(atual));
}

function cartao(l) {
  const id = idPublico(l.patientId);
  const n = votosDe(l);
  const selo = seloVotos(n);
  const url = urlImagemSegura(l.imageUrl);
  const botao = el('button', {
    class: 'lamina', type: 'button',
    'aria-label': `Abrir ${id.texto}`,
    onclick: () => abrirVisor(l),
  },
  el('div', { class: 'lamina__foto' },
    el('img', { src: url, alt: `Campo microscópico da ${id.texto}`, loading: 'lazy', decoding: 'async' }),
    (souCriador(l) || !estado.mostrarIA)
      ? el('span', { class: 'lamina__selo' }, souCriador(l) ? 'Sua lâmina' : 'Classificação oculta')
      : null,
    estado.minhas.has(l.id) ? el('span', { class: 'lamina__selo lamina__selo--esq' }, 'Avaliada por você') : null,
    el('div', { class: 'lamina__chips' }, ...chipsClassificacao(l))),
  el('div', { class: 'lamina__corpo' },
    el('div', { class: 'lamina__linha' },
      el('span', { class: 'lamina__id', title: id.oculto ? 'ID original ocultado por privacidade' : null }, id.texto),
      el('span', { class: 'lamina__data' }, formatarData(l.date || l.createdAt))),
    l.pasta ? el('span', { class: 'lamina__pasta' }, `Pasta: ${l.pasta}`) : null,
    estado.mostrarIA ? blocoClassificacoes(l) : null,
    estado.mostrarVotos
      ? el('span', { class: `votos ${selo.classe}` }, selo.texto)
      : el('span', { class: 'votos votos--oculto' }, 'Votos ocultos')));
  const caixa = el('input', {
    type: 'checkbox', checked: estado.selecionadas.has(l.id), 'aria-label': `Selecionar ${id.texto} para análise em lote`,
    onchange: (ev) => {
      if (ev.target.checked) estado.selecionadas.add(l.id); else estado.selecionadas.delete(l.id);
      atualizarBarraSelecao();
    },
  });
  return el('div', { class: 'lamina-wrap' }, botao, el('label', { class: 'sel', title: 'Selecionar para análise em lote' }, caixa, 'Selecionar'));
}

/** Bloco do cartão com a classificação humana (consenso) e a predição da IA registrada. */
function blocoClassificacoes(l) {
  if (estado.resumoCarregando || (!estado.resumo && !estado.resumoErro)) {
    return el('span', { class: 'lamina__resumo pequeno' }, 'Carregando classificações…');
  }
  if (estado.resumoErro || !estado.resumo) {
    return el('span', { class: 'lamina__resumo pequeno' }, 'Classificações indisponíveis.');
  }
  const lista = estado.resumo.get(l.id) || [];
  const cons = consensoHumano(lista);
  const ia = iaRegistrada(lista);
  return el('span', { class: 'lamina__resumo' },
    el('span', { class: 'lamina__res-linha' }, el('strong', {}, 'Humano: '),
      cons ? `${cons.sigla} (${cons.n} de ${cons.total})` : (lista.length ? 'sem consenso' : 'sem avaliações')),
    el('span', { class: 'lamina__res-linha' }, el('strong', {}, 'IA: '),
      ia || 'sem análise registrada'));
}

function renderizar() {
  limpar(grade);
  atualizarPastas();
  if (estado.mostrarIA && !estado.resumo && !estado.resumoCarregando && !estado.resumoErro) carregarResumo();
  const lista = filtradas();
  for (const l of lista) grade.append(cartao(l));
  const esperandoLesao = estado.mostrarIA && filtroLesaoEl.value !== 'todas' && !estado.resumo;
  contagemEl.textContent = esperandoLesao
    ? 'Carregando classificações para filtrar…'
    : `${lista.length} de ${estado.laminas.length} carregadas`;

  if (estado.erro) mostrarStatus('erro');
  else if (estado.carregando) mostrarStatus('carregando');
  else if (lista.length === 0) mostrarStatus('vazio');
  else mostrarStatus(null);

  maisBtn.hidden = estado.fim || estado.carregando || !!estado.erro;
  todasBtn.hidden = estado.fim || estado.carregando || !!estado.erro;
  notaParcial.hidden = estado.fim || !estado.mostrarIA || filtroLesaoEl.value === 'todas';
  atualizarBarraSelecao();
}

function atualizarNotaVotos() {
  const semVotos = estado.votos == null;
  filtroVotosEl.disabled = semVotos;
  notaVotos.hidden = !semVotos && estado.votosCompleto;
  notaVotos.textContent = semVotos
    ? 'A contagem de votos não está disponível agora; o filtro por votos foi desativado.'
    : 'Há mais avaliações do que o limite lido; as contagens podem estar incompletas.';
}

// ---------------------------------------------------------------- visor

const visor = $('#visor');
const palco = $('#palco');
const visorImg = $('#visor-img');
const visorTitulo = $('#visor-titulo');
const visorInfo = $('#visor-info');
const zoomEl = $('#zoom-valor');

const vista = { escala: 1, x: 0, y: 0, ajuste: 1 };

function aplicarVista() {
  visorImg.style.transform = `translate(${vista.x}px, ${vista.y}px) scale(${vista.escala})`;
  zoomEl.textContent = `${Math.round((vista.escala / vista.ajuste) * 100)}%`;
}

function ajustarVista() {
  const w = visorImg.naturalWidth || 1;
  const h = visorImg.naturalHeight || 1;
  const pw = palco.clientWidth;
  const ph = palco.clientHeight;
  if (!pw || !ph) return; // visor ainda fechado: a área tem tamanho zero (evita zoom NaN)
  vista.ajuste = Math.min(pw / w, ph / h);
  vista.escala = vista.ajuste;
  vista.x = (pw - w * vista.escala) / 2;
  vista.y = (ph - h * vista.escala) / 2;
  aplicarVista();
}

function zoomPara(fator, cx, cy) {
  const nova = Math.min(Math.max(vista.escala * fator, vista.ajuste * 0.5), vista.ajuste * 12);
  const razao = nova / vista.escala;
  if (cx == null) { cx = palco.clientWidth / 2; cy = palco.clientHeight / 2; }
  vista.x = cx - (cx - vista.x) * razao;
  vista.y = cy - (cy - vista.y) * razao;
  vista.escala = nova;
  aplicarVista();
}

let atual = null; // lâmina aberta no visor

function abrirVisor(l) {
  atual = l;
  const id = idPublico(l.patientId);
  const n = votosDe(l);
  visorTitulo.textContent = id.texto;
  visorInfo.textContent = textoInfoVisor(l, n);
  visorImg.alt = `Campo microscópico da ${id.texto}`;
  visorImg.onload = ajustarVista;
  visorImg.src = urlImagemSegura(l.imageUrl);
  if (visorImg.complete && visorImg.naturalWidth) ajustarVista();
  desenharPainel();
  desenharResumoVisor();
  desenharChipsVisor();
  visor.showModal();
  requestAnimationFrame(() => { if (visorImg.naturalWidth) ajustarVista(); });
  palco.focus();
}

function textoInfoVisor(l, n) {
  const votos = estado.mostrarVotos ? seloVotos(n).texto : 'votos ocultos';
  return `Data: ${formatarData(l.date || l.createdAt)} · ${votos}`;
}

/** Resumo no topo do painel: aparece só com as opções de exibição ligadas. */
function desenharResumoVisor() {
  desenharChipsVisor();
  const caixa = $('#resumo-visor');
  limpar(caixa);
  if (!atual || (!estado.mostrarIA && !estado.mostrarVotos)) return;
  const linhas = [];
  if (estado.mostrarVotos) linhas.push(el('p', {}, el('strong', {}, 'Votos: '), seloVotos(votosDe(atual)).texto));
  if (estado.mostrarIA) {
    if (estado.resumoCarregando || (!estado.resumo && !estado.resumoErro)) {
      linhas.push(el('p', { class: 'pequeno' }, 'Carregando classificações…'));
    } else if (estado.resumoErro || !estado.resumo) {
      linhas.push(el('p', { class: 'pequeno' }, 'Classificações indisponíveis no momento.'));
    } else {
      const lista = estado.resumo.get(atual.id) || [];
      const cons = consensoHumano(lista);
      linhas.push(el('p', {}, el('strong', {}, 'Classificação humana: '),
        cons ? `${cons.sigla} (${cons.n} de ${cons.total} avaliações)` : (lista.length ? 'sem consenso ainda' : 'sem avaliações')));
      linhas.push(el('p', {}, el('strong', {}, 'Predição da IA registrada: '), iaRegistrada(lista) || 'nenhuma análise registrada'));
    }
  }
  caixa.append(el('div', { class: 'resumo-visor', role: 'status' }, ...linhas));
}

$('#zoom-mais').addEventListener('click', () => zoomPara(1.4));
$('#zoom-menos').addEventListener('click', () => zoomPara(1 / 1.4));
$('#zoom-ajustar').addEventListener('click', ajustarVista);
$('#visor-fechar').addEventListener('click', () => visor.close());
visor.addEventListener('close', () => { visorImg.removeAttribute('src'); });
visor.addEventListener('click', (e) => { if (e.target === visor) visor.close(); });
window.addEventListener('resize', () => { if (visor.open) ajustarVista(); });

palco.addEventListener('wheel', (e) => {
  e.preventDefault();
  const r = palco.getBoundingClientRect();
  zoomPara(e.deltaY < 0 ? 1.15 : 1 / 1.15, e.clientX - r.left, e.clientY - r.top);
}, { passive: false });

// Arrastar (1 ponteiro) e pinça (2 ponteiros).
const ponteiros = new Map();
let distAnterior = 0;
palco.addEventListener('pointerdown', (e) => {
  palco.setPointerCapture(e.pointerId);
  ponteiros.set(e.pointerId, { x: e.clientX, y: e.clientY });
  palco.classList.add('arrastando');
  if (ponteiros.size === 2) {
    const [a, b] = [...ponteiros.values()];
    distAnterior = Math.hypot(a.x - b.x, a.y - b.y);
  }
});
palco.addEventListener('pointermove', (e) => {
  const p = ponteiros.get(e.pointerId);
  if (!p) return;
  const dx = e.clientX - p.x;
  const dy = e.clientY - p.y;
  p.x = e.clientX; p.y = e.clientY;
  if (ponteiros.size === 1) {
    vista.x += dx; vista.y += dy; aplicarVista();
  } else if (ponteiros.size === 2) {
    const [a, b] = [...ponteiros.values()];
    const d = Math.hypot(a.x - b.x, a.y - b.y);
    if (distAnterior > 0) {
      const r = palco.getBoundingClientRect();
      zoomPara(d / distAnterior, (a.x + b.x) / 2 - r.left, (a.y + b.y) / 2 - r.top);
    }
    distAnterior = d;
  }
});
function soltar(e) {
  ponteiros.delete(e.pointerId);
  if (ponteiros.size === 0) palco.classList.remove('arrastando');
  distAnterior = 0;
}
palco.addEventListener('pointerup', soltar);
palco.addEventListener('pointercancel', soltar);

palco.addEventListener('keydown', (e) => {
  const passo = 40;
  if (e.key === '+' || e.key === '=') zoomPara(1.25);
  else if (e.key === '-') zoomPara(1 / 1.25);
  else if (e.key === '0') ajustarVista();
  else if (e.key === 'ArrowLeft') { vista.x += passo; aplicarVista(); }
  else if (e.key === 'ArrowRight') { vista.x -= passo; aplicarVista(); }
  else if (e.key === 'ArrowUp') { vista.y += passo; aplicarVista(); }
  else if (e.key === 'ArrowDown') { vista.y -= passo; aplicarVista(); }
  else return;
  e.preventDefault();
});

// ------------------------------------------------------ avaliação / edição

const painelEl = $('#painel-conteudo');

function souCriador(l) {
  const u = uidAtual();
  return !!u && !!l.creatorId && l.creatorId === u;
}

function listaClasses(nome, marcada) {
  return CLASSES.map(([sigla, descricao]) => el('label', { class: 'opcao' },
    el('input', { type: 'radio', name: nome, value: sigla, checked: sigla === marcada }),
    el('strong', {}, sigla), ` — ${descricao}`));
}

/** Botões das 8 classes de Bethesda: um toque escolhe (aria-pressed). */
function gradeBethesda(marcada, aoEscolher) {
  let escolhida = SIGLAS.has(marcada) ? marcada : null;
  const botoes = CLASSES.map(([sigla, descricao]) => {
    const b = el('button', {
      type: 'button', class: 'bethesda-btn', 'data-sigla': sigla,
      'aria-pressed': String(sigla === escolhida), title: descricao,
    }, el('strong', {}, sigla), el('span', {}, descricao));
    b.addEventListener('click', () => {
      escolhida = sigla;
      for (const x of botoes) x.setAttribute('aria-pressed', String(x.dataset.sigla === sigla));
      aoEscolher?.(sigla);
    });
    return b;
  });
  return {
    grade: el('div', { class: 'bethesda-grade', role: 'group', 'aria-label': 'Classificação Bethesda' }, ...botoes),
    valor: () => escolhida,
  };
}

function desenharPainel(refazer = false) {
  limpar(painelEl);
  const l = atual;
  if (!l) return;
  const minha = estado.minhas.get(l.id);
  const caixa = el('div', { class: 'aviso aviso--erro', role: 'alert', hidden: true });
  const falha = (e) => { caixa.hidden = false; caixa.textContent = e.message || String(e); };

  if (minha && !refazer) {
    painelEl.append(el('div', { class: 'aviso aviso--ok' },
      el('strong', {}, 'Sua classificação: '), minha.classification || '—', ' ',
      el('button', { class: 'btn btn--sec btn--mini', type: 'button', onclick: () => desenharPainel(true) },
        'Refazer minha avaliação')));
  } else {
    const avaliacao = gradeBethesda(minha?.classification);
    const desc = el('textarea', { rows: 2, maxlength: 2000, 'aria-label': 'Descrição morfológica (opcional)', placeholder: 'Descrição morfológica (opcional)' });
    const obs = el('textarea', { rows: 2, maxlength: 2000, 'aria-label': 'Observações (opcional)', placeholder: 'Observações (opcional)' });
    if (minha) { desc.value = minha.morphologyDescription || ''; obs.value = minha.notes || ''; }
    const salvar = el('button', { class: 'btn btn--bloco', type: 'submit' }, minha ? 'Salvar nova avaliação' : 'Salvar minha avaliação');
    const form = el('form', { class: 'aval-form', novalidate: true,
      onsubmit: async (ev) => {
        ev.preventDefault();
        caixa.hidden = true;
        const marcada = avaliacao.valor();
        if (!marcada || !SIGLAS.has(marcada)) { falha(new Error('Escolha uma classificação de Bethesda.')); return; }
        salvar.disabled = true;
        try {
          const a = await salvarAvaliacao({
            slideId: l.id, classificacao: marcada, descricao: desc.value.trim(), observacoes: obs.value.trim(),
          });
          estado.minhas.set(l.id, a);
          const antes = estado.votos?.get(l.id) || 0;
          if (estado.votos && !minha) estado.votos.set(l.id, antes + 1);
          if (estado.resumo) {
            const lista = estado.resumo.get(l.id) || [];
            lista.push({ classificacao: marcada, ia: null, confianca: null });
            estado.resumo.set(l.id, lista);
          }
          desenharPainel();
          renderizar();
          atualizarInfoVisor();
        } catch (e) { falha(e); salvar.disabled = false; }
      } },
    el('h3', { class: 'painel__titulo' }, minha ? 'Refazer sua avaliação' : 'Sua classificação (Bethesda)'),
    el('p', { class: 'pequeno' }, 'Toque na classe. A avaliação é cega: as classificações das outras pessoas só aparecem se você ligar a opção no topo da página.'),
    avaliacao.grade,
    el('details', { class: 'detalhes-texto' },
      el('summary', {}, 'Descrição e observações (opcional)'),
      desc, obs),
    salvar);
    painelEl.append(form);
  }

  if (souCriador(l)) {
    painelEl.append(el('div', { class: 'botoes--linha acoes-dono' },
      el('button', { class: 'btn btn--sec btn--mini', type: 'button', onclick: () => abrirFormulario(l) }, 'Editar minha lâmina'),
      el('button', { class: 'btn btn--sec btn--mini', type: 'button', onclick: (ev) => confirmarExclusao(ev.currentTarget, l, falha) }, 'Excluir minha lâmina')));
  }
  painelEl.append(el('details', { class: 'detalhes-ia' },
    el('summary', {}, 'Análise por IA e validação externa'),
    painelAnalises(l, minha, falha)));
  painelEl.append(caixa);
}

function confirmarExclusao(botao, l, falha) {
  const dono = botao.parentElement;
  limpar(dono);
  dono.append(
    el('span', { class: 'pequeno' }, 'Excluir esta lâmina e suas avaliações? Não dá para desfazer. '),
    el('button', { class: 'btn btn--mini', type: 'button', onclick: async (ev) => {
      ev.currentTarget.disabled = true;
      try {
        await excluirLamina(l.id);
        estado.laminas = estado.laminas.filter((x) => x.id !== l.id);
        estado.minhas.delete(l.id);
        visor.close();
        renderizar();
      } catch (e) { falha(e); desenharPainel(); }
    } }, 'Sim, excluir'),
    el('button', { class: 'btn btn--sec btn--mini', type: 'button', onclick: () => desenharPainel() }, 'Cancelar'));
}

function atualizarInfoVisor() {
  if (!atual) return;
  visorInfo.textContent = textoInfoVisor(atual, votosDe(atual));
  desenharResumoVisor();
}

// ------------------------------------------------ IA, validação externa, comparação

const CLASSES_MODELO = new Set(['SCC', 'HSIL', 'LSIL', 'NILM']);
const pct = (v) => `${(v * 100).toFixed(1).replace('.', ',')}%`;

/** O modelo só conhece 4 classes; as demais das 8 do CitoAtlas não têm equivalente direto. */
function comparar(a, b) {
  if (!a || !b) return null;
  if (!CLASSES_MODELO.has(a) || !CLASSES_MODELO.has(b)) return 'sem equivalente direto';
  return a === b ? 'concordam' : 'divergem';
}

/** Mostra o estado do carregamento da IA numa caixa de mensagem + barra de progresso. */
async function prepararIA(msgEl, progEl) {
  if (modeloPronto()) return true;
  try {
    const s = await situacaoModelo();
    msgEl.textContent = s.emCache
      ? 'Abrindo a IA neste navegador…'
      : `Baixando a IA (${s.tamanhoTexto}) uma única vez; depois fica guardada neste navegador…`;
    progEl.hidden = false;
    progEl.removeAttribute('value');
    await carregarModelo({
      aoProgredir: (r, t) => {
        progEl.max = t; progEl.value = r;
        msgEl.textContent = `Baixando a IA: ${Math.round((r / t) * 100)}%`;
      },
    });
    msgEl.textContent = 'IA pronta. A análise roda neste aparelho; nenhuma imagem sai do navegador.';
    return true;
  } catch (e) {
    msgEl.textContent = e.message || String(e);
    return false;
  } finally {
    progEl.hidden = true;
  }
}

function resumoIA(r) {
  const info = CATEGORIAS[r.sigla];
  return el('div', { class: 'ia-res', style: `--c:${info?.cor || '#555'}` },
    el('div', {}, el('strong', {}, `${r.sigla}`), ` · ${info?.nome || ''} · confiança `, el('strong', {}, pct(r.confianca))),
    r.instavel ? el('div', { class: 'pequeno' }, 'Predição de baixa confiança: trate como indefinida.') : null);
}

/** Grava a predição da IA no banco (qualquer lâmina). Devolve { ok, erro } com o motivo real. */
async function registrarPredicao(l, r) {
  try {
    await salvarPredicao({
      slideId: l.id, classificacao: r.sigla,
      confianca: Math.round(r.confianca * 1000) / 1000, modelo: 'colo-ia-web',
    });
    adicionarIAResumo(l.id, r.sigla, r.confianca);
    estado.minhasIA.set(l.id, { sigla: r.sigla, confianca: Math.round(r.confianca * 1000) / 1000, quando: Date.now() });
    return { ok: true, erro: null };
  } catch (e) {
    console.warn('Predição não gravada:', e);
    const motivo = e.tipo === 'permissao'
      ? 'O banco recusou a gravação: as regras do Firestore ainda não permitem salvar predições. Publique firebase/firestore.rules.sugerido no console do Firebase.'
      : (e.tipo === 'login' ? 'Sem sessão anônima no banco. Recarregue a página.' : (e.tipo === 'rede' ? 'Sem conexão com o banco agora.' : (e.message || 'Erro desconhecido ao gravar.')));
    return { ok: false, erro: motivo };
  }
}

/** Painel "Análises": humana (já acima), IA interna, validação externa e comparação. */
function painelAnalises(l, minha, falha) {
  const caixa = el('section', { class: 'analises', 'aria-label': 'Análises da lâmina' });
  const podeVerIA = !!minha || souCriador(l) || estado.revelarIA;

  // --- IA interna
  const iaMsg = el('p', { class: 'pequeno', role: 'status' });
  const iaProg = el('progress', { max: 100, hidden: true, 'aria-label': 'Progresso' });
  const iaSaida = el('div', {});
  const salva = estado.minhasIA.get(l.id);
  if (salva?.sigla && podeVerIA) { // só depois da avaliação ou para o criador (avaliação cega)
    iaSaida.append(el('p', { class: 'pequeno' }, 'Sua predição salva no banco:'),
      resumoIA({ sigla: salva.sigla, confianca: salva.confianca, instavel: salva.confianca != null && salva.confianca < 0.6 }));
  }
  const rodar = el('button', { class: 'btn btn--sec btn--mini', type: 'button' },
    podeVerIA ? 'Analisar com a IA' : 'Analisar com a IA (revela a sugestão)');
  rodar.addEventListener('click', async () => {
    if (!podeVerIA) {
      if (rodar.dataset.confirmar !== '1') {
        rodar.dataset.confirmar = '1';
        rodar.textContent = 'Confirmar: ver a IA antes de avaliar pode influenciar a sua avaliação cega';
        return;
      }
      estado.revelarIA = true;
    }
    rodar.disabled = true;
    limpar(iaSaida);
    try {
      if (!(await prepararIA(iaMsg, iaProg))) return;
      iaMsg.textContent = 'Analisando…';
      const blob = await blobDeDataUrl(urlImagemSegura(l.imageUrl));
      const [{ resultado, erro }] = await classificarLote([blob]);
      if (erro) { iaMsg.textContent = erro; return; }
      iaMsg.textContent = '';
      iaSaida.append(resumoIA(resultado));
      const grava = await registrarPredicao(l, resultado);
      iaSaida.append(el('p', { class: 'pequeno' }, grava.ok
        ? 'Predição salva no banco de dados.'
        : `Não foi possível salvar a predição. ${grava.erro} O resultado continua visível aqui.`));
      if (minha) {
        const novo = await salvarComplementoAvaliacao(l.id, {
          aiClassification: resultado.sigla, aiConfidence: Math.round(resultado.confianca * 1000) / 1000,
          aiModelVersion: 'colo-ia-web',
        });
        estado.minhas.set(l.id, { ...minha, ...novo });
        iaSaida.append(el('p', { class: 'pequeno' }, 'Resultado registrado na sua avaliação para comparação.'));
        desenharComparacao();
      }
    } catch (e) { falha(e); } finally { rodar.disabled = false; }
  });
  caixa.append(el('div', { class: 'cartao analise' },
    el('h4', {}, 'Análise por IA (algoritmo interno)'),
    el('p', { class: 'pequeno' }, 'O mesmo modelo da aba Análise (ConvNeXt, 4 classes), executado neste navegador. Apoio à leitura, não é laudo.'),
    el('p', { class: 'pequeno' }, minha || souCriador(l)
      ? 'Na primeira vez a IA (cerca de 188 MB) é baixada e fica guardada neste navegador.'
      : 'Para manter a avaliação cega, avalie a lâmina primeiro: depois a sugestão da IA é liberada. Na primeira vez a IA (cerca de 188 MB) é baixada.'),
    rodar, iaMsg, iaProg, iaSaida));

  // --- validação externa
  const ext = el('div', { class: 'cartao analise' }, el('h4', {}, 'Validação externa (outro algoritmo)'));
  ext.append(el('p', { class: 'pequeno' },
    'Compare com um validador independente: ',
    el('a', { href: VALIDADOR_EXTERNO, target: '_blank', rel: 'noopener noreferrer' }, 'cervical-screening.pythonaisolutions.com'),
    '. Ele é de terceiros: a imagem sai deste site quando você a envia lá, e vale a política daquele serviço. ' +
    'Use só imagens sem dados de pacientes.'));
  if (!minha) {
    ext.append(el('p', { class: 'pequeno' }, 'Disponível depois da sua avaliação humana, para não influenciá-la.'));
  } else {
    const baixar = el('button', { class: 'btn btn--sec btn--mini', type: 'button',
      onclick: async () => {
        try {
          const blob = await blobDeDataUrl(urlImagemSegura(l.imageUrl));
          const url = URL.createObjectURL(blob);
          const a = Object.assign(document.createElement('a'), { href: url, download: `${idPublico(l.patientId).texto.replace(/\s+/g, '_')}.jpg` });
          document.body.append(a); a.click(); a.remove();
          setTimeout(() => URL.revokeObjectURL(url), 2000);
        } catch (e) { falha(e); }
      } }, '1. Baixar a imagem da lâmina');
    const abrir = el('a', { class: 'btn btn--sec btn--mini', href: VALIDADOR_EXTERNO, target: '_blank', rel: 'noopener noreferrer' },
      '2. Abrir o validador externo');
    const sel = el('select', { 'aria-label': 'Resultado do validador externo' },
      el('option', { value: '' }, '3. Resultado que o validador deu…'),
      ...CLASSES.map(([sig, nome]) => el('option', { value: sig }, `${sig} — ${nome}`)),
      el('option', { value: 'Inconclusivo' }, 'Inconclusivo / sem resultado'));
    sel.value = minha.externalClassification || '';
    const registrar = el('button', { class: 'btn btn--mini', type: 'button',
      onclick: async () => {
        if (!sel.value) return;
        registrar.disabled = true;
        try {
          const novo = await salvarComplementoAvaliacao(l.id, { externalClassification: sel.value });
          estado.minhas.set(l.id, { ...minha, ...novo });
          desenharComparacao();
        } catch (e) { falha(e); } finally { registrar.disabled = false; }
      } }, 'Registrar resultado externo');
    ext.append(el('div', { class: 'acoes-dono' }, baixar, abrir), el('div', { class: 'acoes-dono' }, sel, registrar));
  }
  caixa.append(ext);

  // --- comparação
  const comp = el('div', { class: 'cartao analise', id: 'comparacao' });
  function desenharComparacao() {
    limpar(comp);
    const m = estado.minhas.get(l.id);
    comp.append(el('h4', {}, 'Comparação das análises'));
    if (!m) { comp.append(el('p', { class: 'pequeno' }, 'Avalie a lâmina para liberar a comparação entre humano, IA e validador externo.')); return; }
    const iaSigla = m.aiClassification || estado.minhasIA.get(l.id)?.sigla || null;
    const iaConf = m.aiClassification ? m.aiConfidence : estado.minhasIA.get(l.id)?.confianca;
    const linhas = [
      ['Análise humana (sua)', m.classification],
      ['IA interna', iaSigla ? `${iaSigla}${iaConf ? ` (${pct(iaConf)})` : ''}` : null],
      ['Validador externo', m.externalClassification || null],
    ];
    comp.append(el('table', { class: 'tabela' }, el('tbody', {}, ...linhas.map(([a, b]) =>
      el('tr', {}, el('td', {}, a), el('td', {}, el('strong', {}, b || 'ainda não feita'))))))); 
    const hi = comparar(m.classification, iaSigla);
    const he = comparar(m.classification, m.externalClassification === 'Inconclusivo' ? null : m.externalClassification);
    const ie = comparar(iaSigla, m.externalClassification === 'Inconclusivo' ? null : m.externalClassification);
    const notas = [['humano × IA', hi], ['humano × externo', he], ['IA × externo', ie]].filter(([, v]) => v);
    if (notas.length) {
      comp.append(el('p', { class: 'pequeno' }, notas.map(([k, v]) => `${k}: ${v}`).join(' · ')));
      comp.append(el('p', { class: 'legenda' }, 'O modelo só distingue NILM, LSIL, HSIL e SCC; ASC-US, ASC-H, Adenocarcinoma e Outro não têm equivalente direto.'));
    }
  }
  desenharComparacao();
  caixa.append(comp);
  return caixa;
}

// ---------------------------------------------------- seleção e análise em lote

const barraSel = $('#selecao');
const selTexto = $('#selecao-texto');
const selAnalisar = $('#selecao-analisar');

function atualizarBarraSelecao() {
  const n = estado.selecionadas.size;
  barraSel.hidden = estado.laminas.length === 0;
  selTexto.textContent = n === 0 ? 'Nenhuma lâmina selecionada' : `${n} lâmina${n > 1 ? 's' : ''} selecionada${n > 1 ? 's' : ''}`;
  selAnalisar.disabled = n === 0;
  $('#selecao-limpar').disabled = n === 0;
}

$('#selecao-todas').addEventListener('click', () => {
  for (const l of filtradas()) estado.selecionadas.add(l.id);
  for (const c of grade.querySelectorAll('.sel input')) c.checked = true;
  atualizarBarraSelecao();
});
$('#selecao-limpar').addEventListener('click', () => {
  estado.selecionadas.clear();
  for (const c of grade.querySelectorAll('.sel input')) c.checked = false;
  atualizarBarraSelecao();
});

const loteDlg = $('#lote');
const loteCorpo = $('#lote-corpo');
const loteMsg = $('#lote-msg');
const loteProg = $('#lote-prog');
const loteIncluir = $('#lote-incluir');
const loteRodar = $('#lote-rodar');
const loteCsv = $('#lote-csv');
let loteLinhas = [];

function lamPorId(id) { return estado.laminas.find((x) => x.id === id); }
const podeVerIAde = (l) => estado.minhas.has(l.id) || souCriador(l) || loteIncluir.checked;

function desenharLote() {
  limpar(loteCorpo);
  loteLinhas = [...estado.selecionadas].map(lamPorId).filter(Boolean).map((l) => ({ l, r: null, erro: null }));
  const corpo = el('tbody', {});
  loteLinhas.forEach((lin, i) => {
    const id = idPublico(lin.l.patientId);
    lin.celula = el('td', { class: 'lote-res' }, '—');
    corpo.append(el('tr', {},
      el('td', {}, el('img', { class: 'lote-mini', src: urlImagemSegura(lin.l.imageUrl), alt: '' })),
      el('td', {}, id.texto),
      el('td', {}, estado.minhas.has(lin.l.id) ? 'sim' : 'não'),
      lin.celula));
  });
  loteCorpo.append(el('div', { class: 'tabela-rolagem' }, el('table', { class: 'tabela' },
    el('thead', {}, el('tr', {}, el('th', {}, 'Imagem'), el('th', {}, 'ID'), el('th', {}, 'Avaliada por você'), el('th', {}, 'IA interna'))),
    corpo)));
  loteCsv.disabled = true;
}

function celulaResultado(lin) {
  limpar(lin.celula);
  if (lin.erro) { lin.celula.textContent = lin.erro; return; }
  if (!lin.r) { lin.celula.textContent = lin.oculto ? 'oculto: avalie a lâmina primeiro' : '—'; return; }
  const info = CATEGORIAS[lin.r.sigla];
  lin.celula.append(el('strong', { style: `color:${info?.cor || 'inherit'}` }, lin.r.sigla), ` ${pct(lin.r.confianca)}`,
    lin.r.instavel ? ' · baixa confiança' : '');
}

$('#selecao-analisar').addEventListener('click', () => {
  loteIncluir.checked = estado.revelarIA;
  loteMsg.textContent = '';
  desenharLote();
  loteDlg.showModal();
});
$('#lote-fechar').addEventListener('click', () => loteDlg.close());
loteIncluir.addEventListener('change', () => { estado.revelarIA = loteIncluir.checked; });

loteRodar.addEventListener('click', async () => {
  loteRodar.disabled = true;
  try {
    if (!(await prepararIA(loteMsg, loteProg))) return;
    const ativas = loteLinhas.filter((lin) => {
      lin.r = null; lin.erro = null; lin.oculto = !podeVerIAde(lin.l);
      celulaResultado(lin);
      return !lin.oculto;
    });
    if (ativas.length === 0) {
      loteMsg.textContent = 'Nenhuma das lâminas selecionadas pode ser analisada agora: avalie-as primeiro ou marque a opção de incluir as não avaliadas.';
      return;
    }
    const blobs = [];
    for (const lin of ativas) blobs.push(await blobDeDataUrl(urlImagemSegura(lin.l.imageUrl)));
    loteProg.hidden = false; loteProg.max = ativas.length; loteProg.value = 0;
    await classificarLote(blobs, {
      aoItem: (i, r, erro) => {
        ativas[i].r = r; ativas[i].erro = erro; celulaResultado(ativas[i]);
        loteProg.value = i + 1;
        loteMsg.textContent = `Analisadas ${i + 1} de ${ativas.length}…`;
      },
    });
    const feitas = ativas.filter((lin) => lin.r);
    let gravadas = 0;
    let motivoFalha = null;
    for (let i = 0; i < feitas.length; i++) {
      loteMsg.textContent = `Salvando predições no banco: ${i + 1} de ${feitas.length}…`;
      const g = await registrarPredicao(feitas[i].l, feitas[i].r);
      if (g.ok) gravadas++; else motivoFalha = motivoFalha || g.erro;
    }
    loteMsg.textContent = `Concluído: ${ativas.length} lâmina(s) analisada(s) pela IA neste aparelho. ` +
      `${gravadas} predição(ões) salva(s) no banco de dados` +
      (gravadas < feitas.length ? `; ${feitas.length - gravadas} não puderam ser salvas. ${motivoFalha}` : '') + '.' +
      (ativas.length < loteLinhas.length ? ` ${loteLinhas.length - ativas.length} ficaram ocultas para preservar a avaliação cega.` : '');
    loteCsv.disabled = false;
  } catch (e) {
    loteMsg.textContent = e.message || String(e);
  } finally {
    loteProg.hidden = true;
    loteRodar.disabled = false;
  }
});

loteCsv.addEventListener('click', () => {
  const linhas = [['ID', 'Data', 'Avaliada por você', 'IA: classe', 'IA: confiança', 'IA: baixa confiança']];
  for (const lin of loteLinhas) {
    if (!lin.r) continue;
    linhas.push([idPublico(lin.l.patientId).texto, lin.l.date || '', estado.minhas.has(lin.l.id) ? 'sim' : 'não',
      lin.r.sigla, lin.r.confianca.toFixed(4).replace('.', ','), lin.r.instavel ? 'sim' : 'não']);
  }
  baixarTexto('analise-ia-catalogo.csv', csv(linhas));
});

// ----------------------------------------------------------- cadastro/edição

const formDlg = $('#form-lamina');
const form = $('#form');
const fFoto = $('#f-foto');
const fPrevia = $('#f-previa');
const fFotoInfo = $('#f-foto-info');
const fId = $('#f-id');
const fData = $('#f-data');
const fPasta = $('#f-pasta');
const fClasses = $('#f-classes');
const fDesc = $('#f-desc');
const fObs = $('#f-obs');
const fErro = $('#f-erro');
const fSalvar = $('#f-salvar');
const fProg = $('#f-progresso');
let editando = null; // lâmina em edição ou null (nova)
let imagemPronta = null; // data URL comprimido

function hoje() {
  const d = new Date();
  const dois = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${dois(d.getMonth() + 1)}-${dois(d.getDate())}`;
}

function erroForm(msg) {
  fErro.hidden = !msg;
  fErro.textContent = msg || '';
}

async function abrirFormulario(l = null) {
  editando = l;
  imagemPronta = null;
  erroForm('');
  fProg.textContent = '';
  fSalvar.disabled = false;
  fFoto.value = '';
  fFotoInfo.textContent = l ? 'Deixe em branco para manter a foto atual.' : 'Obrigatória. A foto é reduzida no seu aparelho antes de enviar.';
  fPrevia.hidden = true;
  fPrevia.removeAttribute('src');
  $('#form-titulo').textContent = l ? 'Editar lâmina' : 'Cadastrar lâmina';
  fId.value = l ? (l.patientId || '') : '';
  fData.value = l ? (l.date || hoje()) : hoje();
  fPasta.value = l ? (l.pasta || '') : '';
  fDesc.value = l ? (l.morphologyDescription || '') : '';
  fObs.value = l ? (l.notes || '') : '';
  while (fClasses.children.length > 1) fClasses.lastChild.remove();
  for (const n of listaClasses('f-classe', l?.classification)) fClasses.append(n);
  if (visor.open) visor.close();
  formDlg.showModal();
  if (!l) {
    try { fId.value = await proximoId(); } catch (_) { fId.placeholder = 'L-0001'; }
  }
}

fFoto.addEventListener('change', async () => {
  erroForm('');
  imagemPronta = null;
  const f = fFoto.files?.[0];
  if (!f) { fPrevia.hidden = true; return; }
  fFotoInfo.textContent = 'Preparando a imagem…';
  try {
    imagemPronta = await comprimirImagem(f);
    fPrevia.src = imagemPronta;
    fPrevia.alt = 'Prévia da foto escolhida';
    fPrevia.hidden = false;
    fFotoInfo.textContent = `Imagem pronta (${Math.round(imagemPronta.length / 1024)} KB).`;
  } catch (e) {
    fFotoInfo.textContent = '';
    erroForm(e.message);
  }
});

form.addEventListener('submit', async (ev) => {
  ev.preventDefault();
  erroForm('');
  if (!editando && !imagemPronta) { erroForm('Escolha a foto da lâmina.'); return; }
  const id = fId.value.trim().toUpperCase();
  if (!/^L-\d{1,6}$/.test(id)) { erroForm('Use um ID no formato L-0001 (sem nome nem dados do paciente).'); return; }
  if (!/^\d{4}-\d{2}-\d{2}$/.test(fData.value)) { erroForm('Informe a data.'); return; }
  const marcada = form.querySelector('input[name="f-classe"]:checked');
  if (!marcada || !SIGLAS.has(marcada.value)) { erroForm('Escolha a classificação.'); return; }
  fSalvar.disabled = true;
  fProg.textContent = 'Salvando…';
  try {
    const dados = {
      patientId: id, data: fData.value, classificacao: marcada.value,
      descricao: fDesc.value.trim(), observacoes: fObs.value.trim(),
      pasta: fPasta.value.trim(),
    };
    if (editando) {
      const novo = await atualizarLamina(editando.id, { ...dados, ...(imagemPronta ? { imagem: imagemPronta } : {}) });
      const i = estado.laminas.findIndex((x) => x.id === editando.id);
      const mesclada = { ...editando, ...novo, imageUrl: imagemPronta || editando.imageUrl };
      if (i >= 0) estado.laminas[i] = mesclada;
    } else {
      const nova = await criarLamina({ ...dados, imagem: imagemPronta });
      estado.laminas.unshift(nova);
    }
    formDlg.close();
    renderizar();
  } catch (e) {
    erroForm(e.message || String(e));
    fProg.textContent = '';
    fSalvar.disabled = false;
  }
});

$('#form-fechar').addEventListener('click', () => formDlg.close());
$('#nova').addEventListener('click', () => abrirFormulario(null));

// ------------------------------------------------------------------ início

buscaEl.addEventListener('input', renderizar);
filtroVotosEl.addEventListener('change', renderizar);
filtroPastaEl.addEventListener('change', renderizar);
maisBtn.addEventListener('click', carregarMais);
todasBtn.addEventListener('click', carregarTodas);
filtroLesaoEl.addEventListener('change', () => renderizar());

// -------------------------------------------- opções de exibição e download

const mostrarIAEl = $('#mostrar-ia');
const mostrarVotosEl = $('#mostrar-votos');
mostrarIAEl.checked = estado.mostrarIA;
mostrarVotosEl.checked = estado.mostrarVotos;
atualizarFiltroLesao();
mostrarIAEl.addEventListener('change', () => {
  estado.mostrarIA = mostrarIAEl.checked;
  salvarPrefs();
  atualizarFiltroLesao();
  if (estado.mostrarIA && !estado.resumo) carregarResumo();
  renderizar();
  if (visor.open) desenharResumoVisor();
});
mostrarVotosEl.addEventListener('change', () => {
  estado.mostrarVotos = mostrarVotosEl.checked;
  salvarPrefs();
  renderizar();
  if (visor.open) { atualizarInfoVisor(); desenharResumoVisor(); }
});

const dlDlg = $('#dialogo-download');
const dlProg = $('#dl-prog');
const dlErro = $('#dl-erro');
const dlMsg = $('#dl-msg');
const dlIniciar = $('#dl-iniciar');
const dlCancelar = $('#dl-cancelar');
let dlCancelador = null;

const NOME_CLASSE = (s) => String(s).replace(/[^\w.-]+/g, '_');

/** Carrega o JSZip (gerador de .zip) do cdnjs sob demanda. */
function garantirJSZip() {
  if (window.JSZip) return Promise.resolve();
  return new Promise((ok, falha) => {
    const s = document.createElement('script');
    s.src = 'https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js';
    s.onload = () => ok();
    s.onerror = () => falha(new Error('Não foi possível carregar o gerador de .zip. Verifique a conexão.'));
    document.head.append(s);
  });
}

/** Lê todas as lâminas do catálogo, página por página. */
async function todasAsLaminas(sinal) {
  const saida = [];
  let token = null;
  do {
    const { itens, proximo } = await listarLaminas({ tamanho: 100, token, sinal });
    for (const it of itens) if (urlImagemSegura(it.imageUrl)) saida.push(it);
    token = proximo;
    dlMsg.textContent = `Lendo o catálogo: ${saida.length} lâminas…`;
  } while (token);
  return saida;
}

async function baixarPorClasse(fonte, sinal) {
  dlMsg.textContent = 'Lendo o catálogo…';
  const lams = await todasAsLaminas(sinal);
  if (!lams.length) throw new Error('O catálogo não tem lâminas para baixar.');

  let resumo = estado.resumo;
  if (fonte === 'humano' && !resumo) {
    dlMsg.textContent = 'Lendo as avaliações…';
    resumo = (await resumoAvaliacoes({ sinal })).mapa;
  }

  const classes = new Map(); // id -> { classe, origem, confianca }
  if (fonte === 'ia') {
    dlMsg.textContent = 'Preparando a IA neste navegador…';
    const pronto = await prepararIA(dlMsg, dlProg);
    if (!pronto) throw new Error('A IA não pôde ser carregada.');
    dlProg.hidden = false; dlProg.max = lams.length; dlProg.value = 0;
    const blobs = [];
    for (const l of lams) blobs.push(await blobDeDataUrl(urlImagemSegura(l.imageUrl)));
    const saida = await classificarLote(blobs, {
      robustez: false, sinal,
      aoItem: (i) => { dlProg.value = i + 1; dlMsg.textContent = `IA analisando ${i + 1} de ${lams.length}…`; },
    });
    if (sinal?.aborted) throw new DOMException('Cancelado', 'AbortError');
    saida.forEach((s, i) => {
      classes.set(lams[i].id, s.resultado
        ? { classe: s.resultado.sigla, origem: 'IA', confianca: s.resultado.confianca }
        : { classe: 'sem-analise', origem: 'IA', confianca: null });
    });
  } else {
    for (const l of lams) {
      const c = consensoHumano(resumo.get(l.id) || []);
      classes.set(l.id, c ? { classe: c.sigla, origem: 'humano', confianca: null } : { classe: 'sem-consenso', origem: 'humano', confianca: null });
    }
  }

  dlProg.hidden = false; dlProg.max = lams.length; dlProg.value = 0;
  dlMsg.textContent = 'Gerando o arquivo .zip…';
  await garantirJSZip();
  const zip = new window.JSZip();
  const linhas = [['arquivo', 'id', 'pasta', 'data', 'classe', 'origem', 'confianca_ia']];
  for (let i = 0; i < lams.length; i++) {
    if (sinal?.aborted) throw new DOMException('Cancelado', 'AbortError');
    const l = lams[i];
    const info = classes.get(l.id);
    const id = idPublico(l.patientId).texto;
    const arq = `${String(i + 1).padStart(4, '0')}_${NOME_CLASSE(id)}.jpg`;
    const bytes = await (await fetch(urlImagemSegura(l.imageUrl))).arrayBuffer();
    zip.file(`${NOME_CLASSE(info.classe)}/${arq}`, bytes);
    linhas.push([arq, id, l.pasta || '', l.date || '', info.classe, info.origem,
      info.confianca != null ? (info.confianca * 100).toFixed(1).replace('.', ',') : '']);
    dlProg.value = i + 1;
    if (i % 25 === 0) dlMsg.textContent = `Empacotando ${i + 1} de ${lams.length}…`;
  }
  zip.file('indice.csv', csv(linhas));
  const blob = await zip.generateAsync({ type: 'blob', compression: 'STORE' }, (m) => {
    dlMsg.textContent = `Compactando: ${Math.round(m.percent)}%`;
  });
  const data = new Date().toISOString().slice(0, 10);
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `citoilumina-classes-${fonte === 'ia' ? 'ia' : 'humano'}-${data}.zip`;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60000);
  dlMsg.textContent = `Pronto: ${lams.length} imagens em pastas por classe.`;
}

$('#baixar-classes').addEventListener('click', () => {
  dlErro.hidden = true; dlProg.hidden = true; dlMsg.textContent = '';
  dlIniciar.disabled = false;
  dlDlg.showModal();
});
$('#dl-fechar').addEventListener('click', () => { dlCancelador?.abort(); dlDlg.close(); });
dlCancelar.addEventListener('click', () => { dlCancelador?.abort(); dlDlg.close(); });
dlIniciar.addEventListener('click', async () => {
  const fonte = dlDlg.querySelector('input[name="dl-fonte"]:checked')?.value || 'ia';
  dlErro.hidden = true;
  dlIniciar.disabled = true;
  dlCancelador = new AbortController();
  try {
    await baixarPorClasse(fonte, dlCancelador.signal);
  } catch (e) {
    if (e?.name !== 'AbortError') {
      dlErro.hidden = false;
      dlErro.textContent = e.message || String(e);
    } else {
      dlMsg.textContent = 'Download cancelado.';
    }
  } finally {
    dlIniciar.disabled = false;
    dlCancelador = null;
  }
});

const painelConexao = $('#conexao');
const textoConexao = $('#conexao-texto');
const erroConexao = $('#erro-conexao');
const tentarBtn = $('#conexao-tentar');
const conteudoCatalogo = $('#catalogo');

async function iniciar() {
  painelConexao.hidden = false;
  conteudoCatalogo.hidden = true;
  erroConexao.hidden = true;
  tentarBtn.hidden = true;
  textoConexao.textContent = 'Conectando ao banco de dados do catálogo…';
  try {
    await entrar();
  } catch (e) {
    textoConexao.textContent = 'Não foi possível conectar.';
    erroConexao.hidden = false;
    erroConexao.textContent = e.message || String(e);
    tentarBtn.hidden = false;
    return;
  }
  painelConexao.hidden = true;
  conteudoCatalogo.hidden = false;
  try { estado.minhas = await minhasAvaliacoes(); } catch (e) { console.warn('Minhas avaliações indisponíveis:', e); }
  try { estado.minhasIA = await minhasPredicoes(); } catch (e) { console.warn('Minhas predições indisponíveis:', e); }
  carregarMais();
  carregarVotos();
}

tentarBtn.addEventListener('click', iniciar);
iniciar();
