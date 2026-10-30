// Módulo "Catálogo de lâminas" — acervo do CitoAtlas com identidade anônima do Firebase.
//
// Protocolo cego preservado: a lista e o visor nunca mostram classificação,
// descrição, observação nem avaliador de lâminas de OUTRAS pessoas. O criador só vê
// o que ele mesmo cadastrou (no formulário de edição) e cada pessoa só vê a
// própria avaliação. Todo texto vindo do banco entra na página por textContent.

import { CATALOGO } from './config.js';
import { $, el, limpar, formatarData, idPublico, urlImagemSegura, iniciarPagina } from './comum.js';
import {
  listarLaminas, contarVotos, minhasAvaliacoes, proximoId, criarLamina, atualizarLamina,
  excluirLamina, salvarAvaliacao, entrar, uidAtual, ErroFirestore,
} from './firestore.js?v=4';
import { comprimirImagem } from './imagem.js?v=4';

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

const estado = {
  laminas: [],
  proximo: null,
  fim: false,
  carregando: false,
  votos: null, // Map(slideId -> n) ou null se indisponível
  votosCompleto: true,
  erro: null,
  minhas: new Map(), // slideId -> minha avaliação
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
    el('span', { class: 'lamina__selo' }, souCriador(l) ? 'Sua lâmina' : 'Classificação oculta'),
    estado.minhas.has(l.id) ? el('span', { class: 'lamina__selo lamina__selo--esq' }, 'Avaliada por você') : null),
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

let atual = null; // lâmina aberta no visor

function abrirVisor(l) {
  atual = l;
  const id = idPublico(l.patientId);
  const n = votosDe(l);
  visorTitulo.textContent = id.texto;
  visorInfo.textContent =
    `Data: ${formatarData(l.date || l.createdAt)} · ${seloVotos(n).texto} · ` +
    'A classificação fica oculta: a avaliação é cega.';
  visorImg.alt = `Campo microscópico da ${id.texto}`;
  visorImg.onload = ajustarVista;
  visorImg.src = urlImagemSegura(l.imageUrl);
  if (visorImg.complete && visorImg.naturalWidth) ajustarVista();
  desenharPainel();
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

function desenharPainel(refazer = false) {
  limpar(painelEl);
  const l = atual;
  if (!l) return;
  const minha = estado.minhas.get(l.id);
  const caixa = el('div', { class: 'aviso aviso--erro', role: 'alert', hidden: true });
  const falha = (e) => { caixa.hidden = false; caixa.textContent = e.message || String(e); };

  if (minha && !refazer) {
    painelEl.append(el('div', { class: 'aviso aviso--ok' },
      el('strong', {}, 'Sua avaliação: '), minha.classification || '—', ' ',
      el('button', { class: 'btn btn--sec btn--mini', type: 'button', onclick: () => desenharPainel(true) },
        'Refazer minha avaliação')));
  } else {
    const nome = 'aval-classe';
    const desc = el('textarea', { rows: 2, maxlength: 2000, 'aria-label': 'Descrição morfológica (opcional)', placeholder: 'Descrição morfológica (opcional)' });
    const obs = el('textarea', { rows: 2, maxlength: 2000, 'aria-label': 'Observações (opcional)', placeholder: 'Observações (opcional)' });
    if (minha) { desc.value = minha.morphologyDescription || ''; obs.value = minha.notes || ''; }
    const salvar = el('button', { class: 'btn', type: 'submit' }, 'Salvar avaliação');
    const form = el('form', { class: 'aval-form', novalidate: true,
      onsubmit: async (ev) => {
        ev.preventDefault();
        caixa.hidden = true;
        const marcada = form.querySelector(`input[name="${nome}"]:checked`);
        if (!marcada || !SIGLAS.has(marcada.value)) { falha(new Error('Escolha uma classificação.')); return; }
        salvar.disabled = true;
        try {
          const a = await salvarAvaliacao({
            slideId: l.id, classificacao: marcada.value, descricao: desc.value.trim(), observacoes: obs.value.trim(),
          });
          estado.minhas.set(l.id, a);
          const antes = estado.votos?.get(l.id) || 0;
          if (estado.votos && !minha) estado.votos.set(l.id, antes + 1);
          desenharPainel();
          renderizar();
          atualizarInfoVisor();
        } catch (e) { falha(e); salvar.disabled = false; }
      } },
    el('p', { class: 'pequeno' }, minha ? 'Refazendo a sua avaliação:' : 'Avalie esta lâmina (a avaliação é cega):'),
    el('div', { class: 'opcoes' }, listaClasses(nome, minha?.classification)),
    desc, obs, salvar);
    painelEl.append(form);
  }

  if (souCriador(l)) {
    painelEl.append(el('div', { class: 'botoes--linha acoes-dono' },
      el('button', { class: 'btn btn--sec btn--mini', type: 'button', onclick: () => abrirFormulario(l) }, 'Editar minha lâmina'),
      el('button', { class: 'btn btn--sec btn--mini', type: 'button', onclick: (ev) => confirmarExclusao(ev.currentTarget, l, falha) }, 'Excluir minha lâmina')));
  }
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
  visorInfo.textContent =
    `Data: ${formatarData(atual.date || atual.createdAt)} · ${seloVotos(votosDe(atual)).texto} · ` +
    'A classificação fica oculta: a avaliação é cega.';
}

// ----------------------------------------------------------- cadastro/edição

const formDlg = $('#form-lamina');
const form = $('#form');
const fFoto = $('#f-foto');
const fPrevia = $('#f-previa');
const fFotoInfo = $('#f-foto-info');
const fId = $('#f-id');
const fData = $('#f-data');
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
maisBtn.addEventListener('click', carregarMais);

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
  carregarMais();
  carregarVotos();
}

tentarBtn.addEventListener('click', iniciar);
iniciar();
