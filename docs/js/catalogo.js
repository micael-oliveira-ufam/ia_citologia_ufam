// Módulo "Catálogo de lâminas" — leitura pública do acervo do CitoAtlas.
//
// Protocolo cego preservado: o CitoAtlas esconde a classificação de quem ainda
// não avaliou, e aqui ninguém avalia, então nenhuma classificação, descrição
// morfológica, observação ou nome de avaliador é lida ou exibida. Aparecem
// apenas a imagem, o ID, a data e o número de votos.

import { CATALOGO } from './config.js';
import { $, el, limpar, formatarData, idPublico, urlImagemSegura, iniciarPagina } from './comum.js';
import { listarLaminas, contarVotos, ErroFirestore } from './firestore.js';

iniciarPagina('catalogo');

const estado = {
  laminas: [],
  proximo: null,
  fim: false,
  carregando: false,
  votos: null, // Map(slideId -> n) ou null se indisponível
  votosCompleto: true,
  erro: null,
};

const grade = $('#grade');
const status = $('#status');
const maisBtn = $('#mais');
const contagemEl = $('#contagem');
const buscaEl = $('#busca');
const filtroVotosEl = $('#filtro-votos');
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

function filtradas() {
  const termo = buscaEl.value.trim().toLowerCase();
  const f = filtroVotosEl.value;
  return estado.laminas.filter((l) => {
    if (termo && !idPublico(l.patientId).texto.toLowerCase().includes(termo)) return false;
    const n = votosDe(l);
    if (f !== 'todas' && n != null) {
      if (f === 'sem' && n !== 0) return false;
      if (f === 'andamento' && !(n >= 1 && n < CATALOGO.votosParaConsenso)) return false;
      if (f === 'quorum' && n < CATALOGO.votosParaConsenso) return false;
    }
    return true;
  });
}

function cartao(l) {
  const id = idPublico(l.patientId);
  const n = votosDe(l);
  const selo = seloVotos(n);
  const url = urlImagemSegura(l.imageUrl);
  return el('button', {
    class: 'lamina', type: 'button',
    'aria-label': `Abrir ${id.texto}`,
    onclick: () => abrirVisor(l),
  },
  el('div', { class: 'lamina__foto' },
    el('img', { src: url, alt: `Campo microscópico da ${id.texto}`, loading: 'lazy', decoding: 'async' }),
    el('span', { class: 'lamina__selo' }, 'Classificação oculta')),
  el('div', { class: 'lamina__corpo' },
    el('div', { class: 'lamina__linha' },
      el('span', { class: 'lamina__id', title: id.oculto ? 'ID original ocultado por privacidade' : null }, id.texto),
      el('span', { class: 'lamina__data' }, formatarData(l.date || l.createdAt))),
    el('span', { class: `votos ${selo.classe}` }, selo.texto)));
}

function renderizar() {
  limpar(grade);
  const lista = filtradas();
  for (const l of lista) grade.append(cartao(l));
  contagemEl.textContent = `${lista.length} de ${estado.laminas.length} carregadas`;

  if (estado.erro) mostrarStatus('erro');
  else if (estado.carregando) mostrarStatus('carregando');
  else if (lista.length === 0) mostrarStatus('vazio');
  else mostrarStatus(null);

  maisBtn.hidden = estado.fim || estado.carregando || !!estado.erro;
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

function abrirVisor(l) {
  const id = idPublico(l.patientId);
  const n = votosDe(l);
  visorTitulo.textContent = id.texto;
  visorInfo.textContent =
    `Data: ${formatarData(l.date || l.createdAt)} · ${seloVotos(n).texto} · ` +
    'A classificação fica oculta: a avaliação no CitoAtlas é cega.';
  visorImg.alt = `Campo microscópico da ${id.texto}`;
  visorImg.onload = ajustarVista;
  visorImg.src = urlImagemSegura(l.imageUrl);
  if (visorImg.complete && visorImg.naturalWidth) ajustarVista();
  visor.showModal();
  palco.focus();
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

// ------------------------------------------------------------------ início

buscaEl.addEventListener('input', renderizar);
filtroVotosEl.addEventListener('change', renderizar);
maisBtn.addEventListener('click', carregarMais);

carregarMais();
carregarVotos();
