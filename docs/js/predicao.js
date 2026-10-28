// Módulo "Predição de citologia": a imagem é lida e analisada neste navegador.
// Nenhum byte da imagem é enviado a servidor; a rede só é usada para baixar o
// modelo (uma vez; depois fica no armazenamento do navegador).

import { MODELO } from './config.js';
import { $, el, limpar, bytesLegiveis, percentual, iniciarPagina } from './comum.js';
import { CATEGORIAS, POR_GRAVIDADE, ORDEM_DO_MODELO, EXEMPLOS } from './bethesda.js';
import {
  lerManifesto, obterModelo, criarSessao, modeloEstaEmCache, removerModeloSalvo,
  prepararImagem, analisar, desenharCampo, desenharSobreposicao, ErroModelo,
} from './modelo.js';

iniciarPagina('predicao');

const ui = {
  modeloMsg: $('#modelo-msg'),
  modeloProg: $('#modelo-prog'),
  modeloBaixar: $('#modelo-baixar'),
  modeloRemover: $('#modelo-remover'),
  zona: $('#zona'),
  arquivo: $('#arquivo'),
  camera: $('#camera'),
  previa: $('#previa'),
  previaMsg: $('#previa-msg'),
  exemplos: $('#exemplos'),
  robustez: $('#robustez'),
  analisar: $('#analisar'),
  progresso: $('#analise-prog'),
  resultado: $('#resultado'),
};

const estado = {
  manifesto: null,
  modelo: null, // { ort, sessao, entrada }
  imagem: null, // Blob escolhido
  ocupado: false,
};

// ------------------------------------------------------------------ modelo

function mensagemModelo(texto, tipo = 'info') {
  ui.modeloMsg.className = tipo === 'erro' ? 'aviso aviso--erro' : tipo === 'ok' ? 'aviso aviso--ok' : 'pequeno';
  ui.modeloMsg.textContent = texto;
}

function atualizarBotao() {
  ui.analisar.disabled = !(estado.modelo && estado.imagem) || estado.ocupado;
}

async function instalarModelo() {
  ui.modeloBaixar.hidden = true;
  ui.modeloProg.hidden = false;
  ui.modeloProg.removeAttribute('value');
  try {
    const m = estado.manifesto;
    const buffer = await obterModelo(m, {
      aoProgredir: (r, t) => {
        ui.modeloProg.max = t;
        ui.modeloProg.value = r;
        mensagemModelo(`Baixando o modelo… ${bytesLegiveis(r)} de ${bytesLegiveis(t)}`);
      },
    });
    ui.modeloProg.removeAttribute('value');
    mensagemModelo('Abrindo o modelo no navegador…');
    estado.modelo = await criarSessao(buffer);
    ui.modeloProg.hidden = true;
    ui.modeloRemover.hidden = false;
    mensagemModelo(
      `Modelo pronto — versão ${m.versao}` +
      (m.arquitetura ? `, ${m.arquitetura}` : '') +
      (m.quantizacao ? `, ${m.quantizacao}` : '') +
      ` (${bytesLegiveis(m.bytes)}). Conferido por SHA-256.`,
      'ok',
    );
  } catch (e) {
    ui.modeloProg.hidden = true;
    ui.modeloBaixar.hidden = false;
    ui.modeloBaixar.textContent = 'Tentar baixar de novo';
    mensagemModelo(e instanceof ErroModelo ? e.message : `Erro inesperado: ${e}`, 'erro');
    console.error(e, e.detalhe);
  }
  atualizarBotao();
}

async function iniciarModelo() {
  mensagemModelo('Consultando o modelo publicado…');
  try {
    estado.manifesto = await lerManifesto(MODELO.manifesto);
  } catch (e) {
    mensagemModelo(e instanceof ErroModelo ? e.message : `Erro inesperado: ${e}`, 'erro');
    if (e.codigo === 'rede') {
      ui.modeloBaixar.hidden = false;
      ui.modeloBaixar.textContent = 'Tentar de novo';
      ui.modeloBaixar.onclick = () => { ui.modeloBaixar.hidden = true; iniciarModelo(); };
    }
    return;
  }
  const m = estado.manifesto;
  ui.modeloBaixar.onclick = instalarModelo;
  if (await modeloEstaEmCache(m)) {
    await instalarModelo(); // já está neste navegador: sem baixar nada
  } else {
    mensagemModelo(
      `O modelo (versão ${m.versao}, ${bytesLegiveis(m.bytes)}) ainda não está neste navegador. ` +
      'O download acontece uma única vez; use Wi-Fi se estiver no celular.',
    );
    ui.modeloBaixar.hidden = false;
    ui.modeloBaixar.textContent = `Baixar o modelo (${bytesLegiveis(m.bytes)})`;
  }
}

ui.modeloRemover.addEventListener('click', async () => {
  await removerModeloSalvo();
  estado.modelo = null;
  ui.modeloRemover.hidden = true;
  mensagemModelo('Modelo removido deste navegador. Baixe de novo quando quiser analisar.');
  ui.modeloBaixar.hidden = false;
  ui.modeloBaixar.textContent = `Baixar o modelo (${bytesLegiveis(estado.manifesto.bytes)})`;
  atualizarBotao();
});

// ------------------------------------------------------------------ imagem

let urlPrevia = null;
function escolherImagem(blob, descricao) {
  if (urlPrevia) URL.revokeObjectURL(urlPrevia);
  estado.imagem = blob;
  urlPrevia = URL.createObjectURL(blob);
  ui.previa.src = urlPrevia;
  ui.previa.alt = descricao;
  ui.previa.hidden = false;
  ui.previaMsg.textContent = descricao;
  limpar(ui.resultado);
  ui.resultado.hidden = true;
  atualizarBotao();
}

function aoEscolherArquivo(e) {
  const f = e.target.files?.[0];
  e.target.value = '';
  if (!f) return;
  if (!f.type.startsWith('image/')) {
    ui.previaMsg.textContent = 'Esse arquivo não é uma imagem.';
    return;
  }
  for (const b of ui.exemplos.querySelectorAll('button')) b.setAttribute('aria-pressed', 'false');
  escolherImagem(f, `Imagem escolhida: ${f.name}`);
}
ui.arquivo.addEventListener('change', aoEscolherArquivo);
ui.camera.addEventListener('change', aoEscolherArquivo);

for (const ev of ['dragenter', 'dragover']) {
  ui.zona.addEventListener(ev, (e) => { e.preventDefault(); ui.zona.classList.add('sobre'); });
}
for (const ev of ['dragleave', 'drop']) {
  ui.zona.addEventListener(ev, (e) => { e.preventDefault(); ui.zona.classList.remove('sobre'); });
}
ui.zona.addEventListener('drop', (e) => {
  const f = e.dataTransfer?.files?.[0];
  if (f && f.type.startsWith('image/')) {
    for (const b of ui.exemplos.querySelectorAll('button')) b.setAttribute('aria-pressed', 'false');
    escolherImagem(f, `Imagem escolhida: ${f.name}`);
  }
});

for (const ex of EXEMPLOS) {
  const botao = el('button', {
    type: 'button', 'aria-pressed': 'false', 'aria-label': `${ex.titulo}: ${ex.descricao}`, title: ex.descricao,
    onclick: async () => {
      const resp = await fetch(ex.arquivo);
      escolherImagem(await resp.blob(), `${ex.titulo}: ${ex.descricao}`);
      for (const b of ui.exemplos.querySelectorAll('button')) b.setAttribute('aria-pressed', 'false');
      botao.setAttribute('aria-pressed', 'true');
    },
  }, el('img', { src: ex.arquivo, alt: '', loading: 'lazy' }));
  ui.exemplos.append(botao);
}

// ---------------------------------------------------------------- análise

function linhaBarra(sigla, valor, destaque, extra) {
  return el('li', { class: `barra-linha cores-${sigla}` },
    el('b', {}, sigla),
    el('div', { class: 'trilho', role: 'presentation' },
      el('span', { style: `width:${Math.max(0.5, valor * 100).toFixed(1)}%` })),
    el('output', { 'aria-label': `${sigla}: ${percentual(valor)}` },
      percentual(valor), destaque ? ' ◀' : '', extra || ''));
}

function mostrarResultado(r, preparada) {
  limpar(ui.resultado);
  ui.resultado.hidden = false;
  const c = CATEGORIAS[r.sigla];

  const cartao = el('div', { class: 'cartao pilha' });
  cartao.append(
    el('div', { class: 'resultado__topo' },
      el('span', { class: `sigla cores-${r.sigla}` }, r.sigla),
      el('div', {},
        el('div', { class: 'resultado__nome' }, c.nome),
        el('div', { class: 'pequeno' },
          `Confiança: ${percentual(r.confianca)}` +
          (r.concordancia != null ? ` · Concordância entre orientações: ${Math.round(r.concordancia * 100)}%` : '') +
          ` · ${(r.milissegundos / 1000).toFixed(1).replace('.', ',')} s`))),
    el('p', { class: 'pequeno' }, c.explicacao));

  if (r.instavel) {
    cartao.append(el('div', { class: 'aviso', role: 'alert' },
      el('strong', {}, 'Predição instável: '),
      'a rede muda de opinião quando o campo é girado ou espelhado (ou a confiança é baixa). ' +
      'Trate este resultado como indefinido.'));
  } else if (r.siglaRobustez && r.siglaRobustez !== r.sigla) {
    cartao.append(el('div', { class: 'aviso', role: 'alert' },
      `A média das oito orientações aponta ${r.siglaRobustez}, diferente da leitura direta. ` +
      'Vale repetir com outra imagem do mesmo campo.'));
  }

  // Régua de gravidade, com texto além da cor.
  const regua = el('div', { class: 'regua', role: 'img', 'aria-label': `Régua de Bethesda; resultado: ${r.sigla}` });
  for (const k of POR_GRAVIDADE) {
    regua.append(el('div', {
      class: `cores-${k.sigla}`,
      style: k.sigla === r.sigla ? '' : 'opacity:.35',
    }, k.sigla));
  }
  cartao.append(regua);

  // Probabilidades das quatro classes, da menos para a mais grave.
  const barras = el('ul', { class: 'barras', 'aria-label': 'Probabilidade por categoria' });
  for (const k of POR_GRAVIDADE) {
    const i = ORDEM_DO_MODELO.indexOf(k.sigla);
    barras.append(linhaBarra(k.sigla, r.probabilidades[i], k.sigla === r.sigla));
  }
  cartao.append(barras);

  // Campo e mapa de calor.
  const cvCampo = el('canvas', { role: 'img', 'aria-label': 'Campo analisado, redimensionado para 224 por 224' });
  const cvCam = el('canvas', { role: 'img', 'aria-label': 'Campo analisado com mapa de calor sobreposto' });
  desenharCampo(cvCampo, preparada.rgb);
  desenharSobreposicao(cvCam, preparada.rgb, r.cam);
  cartao.append(
    el('div', { class: 'imagens' },
      el('figure', {}, cvCampo, el('figcaption', {}, 'Campo analisado (224 × 224)')),
      el('figure', {}, cvCam, el('figcaption', {}, 'Mapa de calor: regiões claras influenciaram mais a classe escolhida'))),
    el('p', { class: 'pequeno' },
      'Se o destaque cair em fundo, muco ou artefato de preparo, desconfie da predição.'));

  if (r.probRobustez) {
    const rob = el('ul', { class: 'barras', 'aria-label': 'Média das oito orientações' });
    for (const k of POR_GRAVIDADE) {
      rob.append(linhaBarra(k.sigla, r.probRobustez[ORDEM_DO_MODELO.indexOf(k.sigla)], k.sigla === r.siglaRobustez));
    }
    cartao.append(
      el('h3', {}, 'Verificação de robustez'),
      el('p', { class: 'pequeno' },
        'O campo foi analisado nas oito orientações possíveis (giros e rotações de 90°). Como o ' +
        'treinamento usou essas mesmas transformações, a predição não deveria mudar entre elas. ' +
        'Médias das oito leituras:'),
      rob);
  }

  ui.resultado.append(cartao);
  ui.resultado.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

ui.analisar.addEventListener('click', async () => {
  if (!(estado.modelo && estado.imagem) || estado.ocupado) return;
  estado.ocupado = true;
  atualizarBotao();
  ui.progresso.hidden = false;
  ui.progresso.removeAttribute('value');
  ui.analisar.textContent = 'Analisando…';
  try {
    const preparada = await prepararImagem(estado.imagem);
    const r = await analisar(estado.modelo, preparada, {
      robustez: ui.robustez.checked,
      aoProgredir: (i, n) => { ui.progresso.max = n; ui.progresso.value = i; },
    });
    mostrarResultado(r, preparada);
  } catch (e) {
    limpar(ui.resultado);
    ui.resultado.hidden = false;
    ui.resultado.append(el('div', { class: 'aviso aviso--erro', role: 'alert' },
      e instanceof ErroModelo ? e.message : `Erro inesperado na análise: ${e}`));
    console.error(e);
  } finally {
    estado.ocupado = false;
    ui.progresso.hidden = true;
    ui.analisar.textContent = 'Analisar o campo';
    atualizarBotao();
  }
});

iniciarModelo();
