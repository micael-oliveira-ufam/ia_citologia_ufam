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
  excluirLamina, salvarAvaliacao, salvarComplementoAvaliacao, entrar, uidAtual, ErroFirestore,
} from './firestore.js?v=5';
import { comprimirImagem } from './imagem.js?v=5';
import {
  situacaoModelo, carregarModelo, modeloPronto, classificarLote, blobDeDataUrl, csv, baixarTexto,
} from './ia.js?v=5';
import { CATEGORIAS } from './bethesda.js';

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
  selecionadas: new Set(),
  revelarIA: false, // o usuário aceitou ver a IA antes de avaliar (pode influenciar a avaliação cega)
};

const grade = $('#grade');
const status = $('#status');
const maisBtn = $('#mais');
const contagemEl = $('#contagem');
const buscaEl = $('#busca');
const filtroVotosEl = $('#filtro-votos');
const filtroPastaEl = $('#filtro-pasta');
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

function filtradas() {
  const termo = buscaEl.value.trim().toLowerCase();
  const f = filtroVotosEl.value;
  const pasta = filtroPastaEl.value;
  return estado.laminas.filter((l) => {
    if (termo && !idPublico(l.patientId).texto.toLowerCase().includes(termo)) return false;
    if (pasta === '__sem__' && l.pasta) return false;
    if (pasta !== 'todas' && pasta !== '__sem__' && l.pasta !== pasta) return false;
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
  const botao = el('button', {
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
    l.pasta ? el('span', { class: 'lamina__pasta' }, `Pasta: ${l.pasta}`) : null,
    el('span', { class: `votos ${selo.classe}` }, selo.texto)));
  const caixa = el('input', {
    type: 'checkbox', checked: estado.selecionadas.has(l.id), 'aria-label': `Selecionar ${id.texto} para análise em lote`,
    onchange: (ev) => {
      if (ev.target.checked) estado.selecionadas.add(l.id); else estado.selecionadas.delete(l.id);
      atualizarBarraSelecao();
    },
  });
  return el('div', { class: 'lamina-wrap' }, botao, el('label', { class: 'sel', title: 'Selecionar para análise em lote' }, caixa, 'Selecionar'));
}

function renderizar() {
  limpar(grade);
  atualizarPastas();
  const lista = filtradas();
  for (const l of lista) grade.append(cartao(l));
  contagemEl.textContent = `${lista.length} de ${estado.laminas.length} carregadas`;

  if (estado.erro) mostrarStatus('erro');
  else if (estado.carregando) mostrarStatus('carregando');
  else if (lista.length === 0) mostrarStatus('vazio');
  else mostrarStatus(null);

  maisBtn.hidden = estado.fim || estado.carregando || !!estado.erro;
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
  painelEl.append(painelAnalises(l, minha, falha));
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

/** Painel "Análises": humana (já acima), IA interna, validação externa e comparação. */
function painelAnalises(l, minha, falha) {
  const caixa = el('section', { class: 'analises', 'aria-label': 'Análises da lâmina' });
  const podeVerIA = !!minha || souCriador(l) || estado.revelarIA;

  // --- IA interna
  const iaMsg = el('p', { class: 'pequeno', role: 'status' });
  const iaProg = el('progress', { max: 100, hidden: true, 'aria-label': 'Progresso' });
  const iaSaida = el('div', {});
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
    const linhas = [
      ['Análise humana (sua)', m.classification],
      ['IA interna', m.aiClassification ? `${m.aiClassification}${m.aiConfidence ? ` (${pct(m.aiConfidence)})` : ''}` : null],
      ['Validador externo', m.externalClassification || null],
    ];
    comp.append(el('table', { class: 'tabela' }, el('tbody', {}, ...linhas.map(([a, b]) =>
      el('tr', {}, el('td', {}, a), el('td', {}, el('strong', {}, b || 'ainda não feita'))))))); 
    const hi = comparar(m.classification, m.aiClassification);
    const he = comparar(m.classification, m.externalClassification === 'Inconclusivo' ? null : m.externalClassification);
    const ie = comparar(m.aiClassification, m.externalClassification === 'Inconclusivo' ? null : m.externalClassification);
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
    loteMsg.textContent = `Concluído: ${ativas.length} lâmina(s) analisada(s) pela IA neste aparelho.` +
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
