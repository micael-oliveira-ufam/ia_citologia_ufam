// Seção "Análise": a imagem é lida e analisada neste navegador. Nenhum byte da imagem é
// enviado a servidor; a rede só é usada para baixar o modelo (uma vez; depois fica no
// armazenamento do navegador). Fluxo e textos espelham o Streamlit (secao_analise).

import { MODELO } from './config.js';
import { $, el, limpar, bytesLegiveis, percentual } from './comum.js';
import { CATEGORIAS, POR_GRAVIDADE, ORDEM_DO_MODELO, EXEMPLOS, LIMIAR_CONFIANCA, LIMIAR_CONCORDANCIA } from './bethesda.js';
import {
  lerManifesto, obterModelo, criarSessao, modeloEstaEmCache, removerModeloSalvo,
  prepararImagem, analisar, desenharCampo, desenharSobreposicao, ErroModelo,
} from './modelo.js';

const virgula = (n, casas = 1) => n.toFixed(casas).replace('.', ',');

export function iniciarAnalise({ irPara }) {
  const ui = {
    modeloMsg: $('#modelo-msg'), modeloProg: $('#modelo-prog'), modeloRemover: $('#modelo-remover'),
    zona: $('#zona'), arquivo: $('#arquivo'), camera: $('#camera'), robustez: $('#robustez'),
    painel: $('#painel-imagem'), previa: $('#previa'), legenda: $('#previa-legenda'),
    dim: $('#previa-dim'), desc: $('#previa-desc'), ref: $('#previa-ref'),
    analisar: $('#analisar'), limpar: $('#limpar'), progresso: $('#analise-prog'), msg: $('#analise-msg'),
    resultado: $('#resultado'), boasVindas: $('#boas-vindas'),
  };

  const estado = {
    manifesto: null, emCache: false, modelo: null,
    imagem: null, nome: '', referencia: null, descricao: null,
    ocupado: false, urlPrevia: null, fila: 0,
  };

  // ------------------------------------------------------------- modelo

  function mensagemModelo(texto, tipo = 'info') {
    ui.modeloMsg.className = tipo === 'erro' ? 'aviso aviso--erro' : tipo === 'ok' ? 'aviso aviso--ok' : 'pequeno';
    ui.modeloMsg.textContent = texto;
  }

  function descreverModelo(m) {
    return `versão ${m.versao}` + (m.arquitetura ? `, ${m.arquitetura}` : '') + (m.quantizacao ? `, ${m.quantizacao}` : '');
  }

  async function iniciarModelo() {
    mensagemModelo('Consultando o modelo publicado…');
    try {
      estado.manifesto = await lerManifesto(MODELO.manifesto);
    } catch (e) {
      mensagemModelo(e instanceof ErroModelo ? e.message : `Erro inesperado: ${e}`, 'erro');
      return;
    }
    const m = estado.manifesto;
    estado.emCache = await modeloEstaEmCache(m);
    if (estado.emCache) {
      mensagemModelo(`Modelo local pronto · ${bytesLegiveis(m.bytes)} guardados neste navegador (${descreverModelo(m)}).`);
      ui.modeloRemover.hidden = false;
    } else {
      mensagemModelo(
        `O modelo (${bytesLegiveis(m.bytes)}) ainda não está neste navegador. Ele será baixado uma única vez, ` +
        'na primeira análise; use Wi-Fi se estiver no celular.',
      );
    }
  }

  /** Garante o modelo na memória: baixa (se preciso), confere o SHA-256 e abre a sessão. */
  async function garantirModelo() {
    if (estado.modelo) return true;
    if (!estado.manifesto) await iniciarModelo();
    const m = estado.manifesto;
    if (!m) return false;
    ui.modeloProg.hidden = false;
    ui.modeloProg.removeAttribute('value');
    try {
      const buffer = await obterModelo(m, {
        aoProgredir: (r, t) => {
          ui.modeloProg.max = t;
          ui.modeloProg.value = r;
          ui.msg.textContent = `Baixando o modelo: ${Math.round((r / t) * 100)}% de ${bytesLegiveis(t)}`;
          mensagemModelo(`Baixando o modelo… ${bytesLegiveis(r)} de ${bytesLegiveis(t)}`);
        },
      });
      ui.modeloProg.removeAttribute('value');
      ui.msg.textContent = 'Preparando o modelo…';
      mensagemModelo('Abrindo o modelo no navegador…');
      estado.modelo = await criarSessao(buffer);
      estado.emCache = true;
      ui.modeloRemover.hidden = false;
      mensagemModelo(`Modelo pronto · ${bytesLegiveis(m.bytes)} (${descreverModelo(m)}). Conferido por SHA-256.`, 'ok');
      return true;
    } catch (e) {
      mensagemModelo(e instanceof ErroModelo ? e.message : `Erro inesperado: ${e}`, 'erro');
      console.error(e, e.detalhe);
      return false;
    } finally {
      ui.modeloProg.hidden = true;
    }
  }

  ui.modeloRemover.addEventListener('click', async () => {
    await removerModeloSalvo();
    estado.modelo = null;
    estado.emCache = false;
    ui.modeloRemover.hidden = true;
    mensagemModelo(`Modelo removido deste navegador. Será baixado de novo (${bytesLegiveis(estado.manifesto.bytes)}) na próxima análise.`);
  });

  // ------------------------------------------------------------- imagem

  async function dimensoes(blob) {
    try {
      const bmp = await createImageBitmap(blob);
      const d = { w: bmp.width, h: bmp.height };
      bmp.close?.();
      return d;
    } catch (_) { return null; }
  }

  async function escolherImagem(blob, { nome, referencia = null, descricao = null }) {
    if (estado.urlPrevia) URL.revokeObjectURL(estado.urlPrevia);
    estado.imagem = blob;
    estado.nome = nome;
    estado.referencia = referencia;
    estado.descricao = descricao;
    estado.urlPrevia = URL.createObjectURL(blob);
    ui.previa.src = estado.urlPrevia;
    ui.previa.alt = `Lâmina carregada: ${nome}`;
    ui.legenda.textContent = nome;
    const d = await dimensoes(blob);
    ui.dim.textContent = (d ? `Dimensões originais: ${d.w} × ${d.h} px. ` : '') +
      'A imagem é redimensionada para 224 × 224 e normalizada antes de entrar na rede.';
    ui.desc.hidden = !descricao;
    ui.desc.textContent = descricao || '';
    ui.ref.hidden = !referencia;
    ui.ref.textContent = '';
    if (referencia) {
      ui.ref.append('Classe de referência do atlas: ', el('strong', {}, referencia), '.');
    }
    ui.painel.hidden = false;
    ui.boasVindas.hidden = true;
    limpar(ui.resultado);
    ui.resultado.hidden = true;
    ui.msg.hidden = true;
  }

  function limparImagem() {
    if (estado.urlPrevia) URL.revokeObjectURL(estado.urlPrevia);
    Object.assign(estado, { imagem: null, nome: '', referencia: null, descricao: null, urlPrevia: null });
    ui.previa.removeAttribute('src');
    ui.painel.hidden = true;
    limpar(ui.resultado);
    ui.resultado.hidden = true;
    ui.boasVindas.hidden = false;
    ui.msg.hidden = true;
  }
  ui.limpar.addEventListener('click', limparImagem);

  async function aoEscolherArquivo(e) {
    const f = e.target.files?.[0];
    e.target.value = '';
    if (!f) return;
    if (!f.type.startsWith('image/')) {
      mostrarErro('Esse arquivo não é uma imagem. Use JPEG, PNG ou WebP.');
      return;
    }
    await escolherImagem(f, { nome: f.name });
  }
  ui.arquivo.addEventListener('change', aoEscolherArquivo);
  ui.camera.addEventListener('change', aoEscolherArquivo);

  for (const ev of ['dragenter', 'dragover']) {
    ui.zona.addEventListener(ev, (e) => { e.preventDefault(); ui.zona.classList.add('sobre'); });
  }
  for (const ev of ['dragleave', 'drop']) {
    ui.zona.addEventListener(ev, (e) => { e.preventDefault(); ui.zona.classList.remove('sobre'); });
  }
  ui.zona.addEventListener('drop', async (e) => {
    const f = e.dataTransfer?.files?.[0];
    if (f && f.type.startsWith('image/')) await escolherImagem(f, { nome: f.name });
  });

  // Botões "Analisar" da seção Imagens de teste.
  for (const botao of document.querySelectorAll('[data-analisar]')) {
    botao.addEventListener('click', async () => {
      const ex = EXEMPLOS.find((x) => x.arquivo === `exemplos/${botao.dataset.analisar}`);
      if (!ex) return;
      botao.disabled = true;
      try {
        const resp = await fetch(ex.arquivo);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        await escolherImagem(await resp.blob(), { nome: botao.dataset.analisar, referencia: ex.referencia, descricao: ex.descricao });
        irPara('analise');
        executar();
      } catch (e) {
        irPara('analise');
        mostrarErro('Arquivo de exemplo não encontrado. Recarregue a página e tente de novo.');
        console.error(e);
      } finally {
        botao.disabled = false;
      }
    });
  }

  // ------------------------------------------------------------ resultado

  function mostrarErro(texto) {
    ui.resultado.hidden = false;
    limpar(ui.resultado);
    ui.resultado.append(el('div', { class: 'aviso aviso--erro', role: 'alert' }, texto));
  }

  function linhaBarra(sigla, valor, destaque) {
    return el('li', { class: `barra-linha${destaque ? ' destaque' : ''}`, style: `--c:${CATEGORIAS[sigla].cor}` },
      el('b', {}, sigla),
      el('div', { class: 'trilho', role: 'presentation' },
        el('span', { style: `width:${Math.max(0.5, valor * 100).toFixed(1)}%` })),
      el('output', { 'aria-label': `${sigla}: ${percentual(valor)}` }, percentual(valor)));
  }

  function reguaResultado(ativa) {
    const r = el('div', { class: 'regua regua--resultado', role: 'img', 'aria-label': `Escala de gravidade; resultado: ${ativa}` });
    for (const k of POR_GRAVIDADE) {
      r.append(el('div', { class: `degrau${k.sigla === ativa ? ' ativo' : ''}`, style: `border-color:${k.cor}` },
        el('div', { class: 'sigla', style: `color:${k.cor}` }, k.sigla),
        el('div', { class: 'desc' }, k.nome)));
    }
    return r;
  }

  /** Miniatura transformada do campo, para o painel de aumentos (ilustrativo). */
  function miniatura(origem, rotulo, tipo) {
    const lado = 220;
    const esc = Math.min(lado / origem.width, lado / origem.height);
    const w = Math.round(origem.width * esc), h = Math.round(origem.height * esc);
    const cv = el('canvas', { role: 'img', 'aria-label': rotulo });
    cv.width = tipo === 'giro90' ? h : w;
    cv.height = tipo === 'giro90' ? w : h;
    const ctx = cv.getContext('2d');
    ctx.save();
    if (tipo === 'horizontal') { ctx.translate(w, 0); ctx.scale(-1, 1); }
    else if (tipo === 'vertical') { ctx.translate(0, h); ctx.scale(1, -1); }
    else if (tipo === 'giro90') { ctx.translate(h, 0); ctx.rotate(Math.PI / 2); }
    ctx.drawImage(origem, 0, 0, w, h);
    ctx.restore();
    if (tipo === 'cor') cv.style.filter = 'brightness(1.12) contrast(1.12) saturate(1.2) hue-rotate(14deg)';
    return el('figure', {}, cv, el('figcaption', {}, rotulo));
  }

  function mostrarResultado(r, preparada) {
    limpar(ui.resultado);
    ui.resultado.hidden = false;
    const info = CATEGORIAS[r.sigla];
    const saida = ui.resultado;

    saida.append(el('div', { class: 'cartao res-cartao', style: `--c:${info.cor}` },
      el('div', { class: 'sobrancelha' }, 'Sugestão do modelo'),
      el('div', { class: 'res-titulo' }, `${r.sigla}. ${info.nome}`),
      el('div', { class: 'res-linha' },
        'Confiança do algoritmo: ', el('b', {}, percentual(r.confianca)),
        r.concordancia != null ? [' · Concordância entre 8 orientações: ', el('b', {}, `${Math.round(r.concordancia * 100)}%`)] : null),
      el('div', { class: 'res-linha' }, info.resumo)));

    if (r.confianca < LIMIAR_CONFIANCA || (r.concordancia != null && r.concordancia < LIMIAR_CONCORDANCIA)) {
      saida.append(el('div', { class: 'aviso', role: 'alert' },
        'Predição instável: a rede muda de opinião quando a imagem é girada ou espelhada. ' +
        'Trate este resultado como indefinido.'));
    } else if (r.probRobustez && r.siglaRobustez !== r.sigla) {
      saida.append(el('div', { class: 'aviso', role: 'alert' },
        `A média sobre as oito orientações aponta ${r.siglaRobustez}, diferente da leitura direta. ` +
        'Vale repetir com outra imagem do mesmo campo.'));
    }

    if (estado.referencia) {
      if (estado.referencia === r.sigla) {
        saida.append(el('div', { class: 'aviso aviso--ok', role: 'status' },
          `A predição coincide com a referência do atlas (${estado.referencia}).`));
      } else {
        saida.append(el('div', { class: 'aviso aviso--erro', role: 'status' },
          `A predição (${r.sigla}) diverge da referência do atlas (${estado.referencia}). ` +
          'Divergências como esta são o material mais útil para melhorar o modelo.'));
      }
    }

    saida.append(reguaResultado(r.sigla), el('p', { class: 'legenda' }, 'Gravidade crescente da esquerda para a direita.'));

    const cvOriginal = el('canvas', { role: 'img', 'aria-label': 'Campo original' });
    const cvCam = el('canvas', { role: 'img', 'aria-label': 'Campo com Grad-CAM sobreposto' });
    desenharCampo(cvOriginal, preparada.exibicao);
    desenharSobreposicao(cvCam, preparada.exibicao, r.cam);

    const barras = el('ul', { class: 'barras', 'aria-label': 'Confiança por classe' });
    for (const k of POR_GRAVIDADE) {
      barras.append(linhaBarra(k.sigla, r.probabilidades[ORDEM_DO_MODELO.indexOf(k.sigla)], k.sigla === r.sigla));
    }

    saida.append(el('div', { class: 'res-grade' },
      el('div', {},
        el('h3', {}, 'Onde a rede olhou'),
        el('div', { class: 'res-imagens' },
          el('figure', {}, cvOriginal, el('figcaption', {}, 'Campo original')),
          el('figure', {}, cvCam, el('figcaption', {}, 'Grad-CAM sobreposto'))),
        el('p', { class: 'legenda' },
          'Tons claros marcam as regiões de maior influência na classe escolhida. Se o destaque cair em ' +
          'fundo, muco ou artefato de preparo, desconfie da predição.')),
      el('div', {}, el('h3', {}, 'Confiança por classe'), barras)));

    if (r.probRobustez) {
      const linhas = ORDEM_DO_MODELO.map((s, i) => el('tr', {},
        el('td', {}, el('b', { style: `color:${CATEGORIAS[s].cor}` }, s)),
        el('td', {}, percentual(r.probabilidades[i])),
        el('td', {}, percentual(r.probRobustez[i]))));
      saida.append(el('details', { class: 'expansor' },
        el('summary', {}, 'Teste de robustez: mesma lâmina, oito orientações'),
        el('p', {},
          'O treinamento usou giros e rotações de 90°, então a predição não deveria mudar com a orientação ' +
          'da imagem. A tabela compara a leitura direta com a média das oito simetrias.'),
        el('div', { class: 'tabela-rolagem' }, el('table', { class: 'tabela' },
          el('thead', {}, el('tr', {}, el('th', {}, 'Classe'), el('th', {}, 'Leitura direta'), el('th', {}, 'Média das 8 orientações'))),
          el('tbody', {}, linhas))),
        el('div', { class: 'aumentos' },
          miniatura(preparada.exibicao, 'Giro horizontal', 'horizontal'),
          miniatura(preparada.exibicao, 'Giro vertical', 'vertical'),
          miniatura(preparada.exibicao, 'Rotação 90°', 'giro90'),
          miniatura(preparada.exibicao, 'Variação de cor', 'cor')),
        el('p', { class: 'legenda' }, 'Ilustração das transformações usadas no treino (a variação de cor é aproximada).')));
    }

    saida.append(el('p', { class: 'legenda' },
      `Análise feita neste dispositivo em ${virgula(r.milissegundos / 1000)} s. A imagem não saiu do navegador.`));
    saida.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // ------------------------------------------------------------- execução

  async function executar() {
    if (!estado.imagem || estado.ocupado) return;
    estado.ocupado = true;
    ui.analisar.disabled = true;
    ui.limpar.disabled = true;
    ui.msg.hidden = false;
    ui.msg.textContent = 'Preparando o modelo…';
    limpar(ui.resultado);
    ui.resultado.hidden = true;
    try {
      if (!(await garantirModelo())) {
        mostrarErro(ui.modeloMsg.textContent || 'O modelo não está disponível.');
        return;
      }
      ui.progresso.hidden = false;
      ui.progresso.removeAttribute('value');
      ui.msg.textContent = 'Extraindo características celulares…';
      const preparada = await prepararImagem(estado.imagem);
      const r = await analisar(estado.modelo, preparada, {
        robustez: ui.robustez.checked,
        aoProgredir: (i, n) => { ui.progresso.max = n; ui.progresso.value = i; },
      });
      mostrarResultado(r, preparada);
    } catch (e) {
      mostrarErro(e instanceof ErroModelo ? e.message : `A análise falhou. Detalhe técnico: ${e}`);
      console.error(e);
    } finally {
      estado.ocupado = false;
      ui.analisar.disabled = false;
      ui.limpar.disabled = false;
      ui.progresso.hidden = true;
      ui.msg.hidden = true;
    }
  }
  ui.analisar.addEventListener('click', executar);

  iniciarModelo();
}
