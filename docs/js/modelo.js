// Modelo de citologia no navegador: manifesto, download verificado, sessão ONNX,
// pré-processamento, inferência, mapa de calor e verificação de robustez.
//
// Tudo aqui espelha o aplicativo (app/lib/servicos/classificador.dart e
// imagem.dart) e o script de exportação (app/tools/export_onnx.py):
//   entrada  RGB em [0,1], NCHW, 1x3x224x224 (normalização ImageNet no grafo)
//   saídas   logits [1,4] e cam [1,4,7,7]
// O redimensionamento é o bilinear com suavização do Pillow (Image.BILINEAR),
// o mesmo usado na conferência numérica do export_onnx.py.

import { ORDEM_DO_MODELO } from './bethesda.js';

export const LADO = 224;
export const LADO_CAM = 7;
const NOME_CACHE = 'colo-ia-modelo';

export class ErroModelo extends Error {
  constructor(codigo, mensagem, detalhe) {
    super(mensagem);
    this.codigo = codigo;
    this.detalhe = detalhe;
  }
}

// ------------------------------------------------------------- manifesto

export async function lerManifesto(url, sinal) {
  let resp;
  try {
    resp = await fetch(url, { cache: 'no-store', signal: sinal });
  } catch (e) {
    throw new ErroModelo('rede', 'Não foi possível consultar o modelo. Verifique a conexão.', String(e));
  }
  if (resp.status === 404) {
    throw new ErroModelo(
      'nao-publicado',
      'O modelo de citologia ainda não foi publicado neste site.',
    );
  }
  if (!resp.ok) throw new ErroModelo('manifesto', `Falha ao ler o manifesto do modelo (HTTP ${resp.status}).`);
  let m;
  try { m = await resp.json(); } catch (e) {
    throw new ErroModelo('manifesto', 'O manifesto do modelo está ilegível.', String(e));
  }
  return validarManifesto(m, url);
}

function validarManifesto(m, urlManifesto) {
  const ruim = (msg) => new ErroModelo('manifesto', `Manifesto do modelo inválido: ${msg}.`);
  if (!m || typeof m !== 'object') throw ruim('formato');
  if (typeof m.versao !== 'string' || !m.versao) throw ruim('versão ausente');
  if (typeof m.arquivo !== 'string' || !m.arquivo) throw ruim('arquivo ausente');
  if (!Number.isInteger(m.bytes) || m.bytes <= 0 || m.bytes > 1.5e9) throw ruim('tamanho');
  if (typeof m.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(m.sha256)) throw ruim('SHA-256');
  const mesmaOrdem =
    Array.isArray(m.classes) &&
    m.classes.length === ORDEM_DO_MODELO.length &&
    m.classes.every((c, i) => c === ORDEM_DO_MODELO[i]);
  if (!mesmaOrdem) {
    throw new ErroModelo(
      'classes',
      `A ordem das classes do modelo (${JSON.stringify(m.classes)}) não bate com a esperada ` +
      `(${JSON.stringify(ORDEM_DO_MODELO)}). A análise foi bloqueada para não inverter diagnósticos.`,
    );
  }
  const url = new URL(m.arquivo, urlManifesto);
  if (url.origin !== new URL(urlManifesto).origin) throw ruim('o arquivo precisa estar no mesmo site');
  return { ...m, urlArquivo: url.href };
}

function urlVersionada(m) {
  // O GitHub Pages ignora a query; ela só separa as versões no cache do navegador.
  return `${m.urlArquivo}?v=${m.sha256.slice(0, 16)}`;
}

// ---------------------------------------------------------- download + hash

export async function sha256Hex(buffer) {
  const h = await crypto.subtle.digest('SHA-256', buffer);
  return [...new Uint8Array(h)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

export async function modeloEstaEmCache(m) {
  try {
    const c = await caches.open(NOME_CACHE);
    return !!(await c.match(urlVersionada(m)));
  } catch (_) { return false; }
}

export async function removerModeloSalvo() {
  try { await caches.delete(NOME_CACHE); } catch (_) { /* ignora */ }
}

async function lerDoCache(m) {
  try {
    const c = await caches.open(NOME_CACHE);
    const r = await c.match(urlVersionada(m));
    if (!r) return null;
    const buf = await r.arrayBuffer();
    if (buf.byteLength === m.bytes && (await sha256Hex(buf)) === m.sha256) return buf;
    await c.delete(urlVersionada(m)); // corrompido: descarta
  } catch (_) { /* sem cache: baixa de novo */ }
  return null;
}

async function gravarNoCache(m, buffer) {
  try {
    const c = await caches.open(NOME_CACHE);
    for (const req of await c.keys()) {
      if (req.url !== urlVersionada(m)) await c.delete(req); // versões antigas
    }
    await c.put(urlVersionada(m), new Response(buffer, {
      headers: { 'content-type': 'application/octet-stream' },
    }));
  } catch (_) { /* cache cheio ou indisponível: segue sem */ }
}

/** Devolve o ArrayBuffer do modelo, do cache ou baixado e conferido (tamanho + SHA-256). */
export async function obterModelo(m, { aoProgredir, sinal } = {}) {
  const guardado = await lerDoCache(m);
  if (guardado) { aoProgredir?.(m.bytes, m.bytes); return guardado; }

  let resp;
  try {
    resp = await fetch(urlVersionada(m), { signal: sinal, cache: 'no-store' });
  } catch (e) {
    throw new ErroModelo('rede', 'O download do modelo falhou. Verifique a conexão e tente de novo.', String(e));
  }
  if (!resp.ok) {
    throw new ErroModelo('download', `O arquivo do modelo não foi encontrado (HTTP ${resp.status}).`);
  }
  const corpo = new Uint8Array(m.bytes);
  let recebido = 0;
  const leitor = resp.body.getReader();
  for (;;) {
    const { done, value } = await leitor.read();
    if (done) break;
    if (recebido + value.length > m.bytes) {
      throw new ErroModelo('tamanho', 'O arquivo baixado é maior do que o manifesto declara. Descartado.');
    }
    corpo.set(value, recebido);
    recebido += value.length;
    aoProgredir?.(recebido, m.bytes);
  }
  if (recebido !== m.bytes) {
    throw new ErroModelo('tamanho', 'O download terminou incompleto. Tente de novo.');
  }
  if ((await sha256Hex(corpo.buffer)) !== m.sha256) {
    throw new ErroModelo('hash', 'O arquivo baixado não confere com o SHA-256 do manifesto. Descartado.');
  }
  await gravarNoCache(m, corpo.buffer);
  return corpo.buffer;
}

// ------------------------------------------------------------------ sessão

let promessaOrt = null;
function carregarOrt() {
  promessaOrt ??= import('../vendor/onnxruntime-web/ort.wasm.min.mjs').then((ort) => {
    ort.env.wasm.wasmPaths = new URL('../vendor/onnxruntime-web/', import.meta.url).href;
    ort.env.wasm.numThreads = 1; // GitHub Pages não envia COOP/COEP: sem multithread
    ort.env.wasm.proxy = false;
    ort.env.logLevel = 'error'; // silencia avisos de CPU do runtime
    return ort;
  });
  return promessaOrt;
}

export async function criarSessao(buffer) {
  let ort;
  try { ort = await carregarOrt(); } catch (e) {
    throw new ErroModelo('runtime', 'Não foi possível carregar o motor de inferência (ONNX Runtime Web).', String(e));
  }
  let sessao;
  try {
    sessao = await ort.InferenceSession.create(new Uint8Array(buffer), {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
  } catch (e) {
    throw new ErroModelo('sessao', 'O navegador não conseguiu abrir o modelo.', String(e));
  }
  const faltam = ['logits', 'cam'].filter((n) => !sessao.outputNames.includes(n));
  if (faltam.length) {
    throw new ErroModelo('saidas', `O modelo não tem a(s) saída(s) esperada(s): ${faltam.join(', ')}.`);
  }
  return { ort, sessao, entrada: sessao.inputNames[0] };
}

// ------------------------------------------------- pré-processamento (Pillow)

const PRECISAO = 22; // bits de ponto fixo, como no Pillow

function filtroBilinear(x) {
  x = Math.abs(x);
  return x < 1 ? 1 - x : 0;
}

function coeficientes(entrada, saida) {
  const escala = entrada / saida;
  const escalaFiltro = Math.max(escala, 1);
  const suporte = escalaFiltro; // suporte do bilinear = 1
  const tam = Math.ceil(suporte) * 2 + 1;
  const inicios = new Int32Array(saida);
  const contagens = new Int32Array(saida);
  const pesos = new Int32Array(saida * tam);
  const tmp = new Float64Array(tam);
  for (let i = 0; i < saida; i++) {
    const centro = (i + 0.5) * escala;
    let xmin = Math.trunc(centro - suporte + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(centro + suporte + 0.5);
    if (xmax > entrada) xmax = entrada;
    xmax -= xmin;
    let soma = 0;
    for (let x = 0; x < xmax; x++) {
      const w = filtroBilinear((x + xmin - centro + 0.5) / escalaFiltro);
      tmp[x] = w;
      soma += w;
    }
    for (let x = 0; x < xmax; x++) {
      const w = soma !== 0 ? tmp[x] / soma : tmp[x];
      pesos[i * tam + x] = Math.trunc(w * (1 << PRECISAO) + (w < 0 ? -0.5 : 0.5));
    }
    inicios[i] = xmin;
    contagens[i] = xmax;
  }
  return { inicios, contagens, pesos, tam };
}

function limitar8(v) {
  v >>= PRECISAO;
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

/**
 * Redimensiona RGBA (Uint8ClampedArray, largura x altura) para RGB do tamanho
 * pedido, com o mesmo algoritmo do Pillow: passada horizontal e depois vertical,
 * aritmética de ponto fixo com arredondamento a cada passada.
 */
export function redimensionarComoPillow(rgba, larg, alt, novaLarg, novaAlt) {
  const h = coeficientes(larg, novaLarg);
  const v = coeficientes(alt, novaAlt);
  const meio = new Uint8Array(alt * novaLarg * 3);
  const metade = 1 << (PRECISAO - 1);

  for (let y = 0; y < alt; y++) {
    for (let x = 0; x < novaLarg; x++) {
      const ini = h.inicios[x];
      const n = h.contagens[x];
      let r = metade, g = metade, b = metade;
      for (let k = 0; k < n; k++) {
        const w = h.pesos[x * h.tam + k];
        const p = (y * larg + ini + k) * 4;
        r += rgba[p] * w;
        g += rgba[p + 1] * w;
        b += rgba[p + 2] * w;
      }
      const o = (y * novaLarg + x) * 3;
      meio[o] = limitar8(r);
      meio[o + 1] = limitar8(g);
      meio[o + 2] = limitar8(b);
    }
  }

  const saida = new Uint8Array(novaAlt * novaLarg * 3);
  for (let y = 0; y < novaAlt; y++) {
    const ini = v.inicios[y];
    const n = v.contagens[y];
    for (let x = 0; x < novaLarg; x++) {
      let r = metade, g = metade, b = metade;
      for (let k = 0; k < n; k++) {
        const w = v.pesos[y * v.tam + k];
        const p = ((ini + k) * novaLarg + x) * 3;
        r += meio[p] * w;
        g += meio[p + 1] * w;
        b += meio[p + 2] * w;
      }
      const o = (y * novaLarg + x) * 3;
      saida[o] = limitar8(r);
      saida[o + 1] = limitar8(g);
      saida[o + 2] = limitar8(b);
    }
  }
  return saida;
}

async function decodificar(blob) {
  try {
    return await createImageBitmap(blob);
  } catch (_) {
    // Navegadores sem createImageBitmap(Blob): usa <img>.
    const url = URL.createObjectURL(blob);
    try {
      const img = new Image();
      img.src = url;
      await img.decode();
      return img;
    } finally {
      URL.revokeObjectURL(url);
    }
  }
}

/** Imagem (File/Blob) -> { tensor Float32 [3*224*224], rgb Uint8 [224*224*3], largura, altura }. */
export async function prepararImagem(blob) {
  let fonte;
  try { fonte = await decodificar(blob); } catch (_) {
    throw new ErroModelo('imagem', 'Não foi possível ler esta imagem. Use JPEG, PNG ou WebP.');
  }
  const larg = fonte.width;
  const alt = fonte.height;
  if (!larg || !alt || larg * alt > 1.0e8) {
    throw new ErroModelo('imagem', 'A imagem é grande demais (acima de 100 megapixels) ou está vazia.');
  }
  const cv = document.createElement('canvas');
  cv.width = larg;
  cv.height = alt;
  const ctx = cv.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#fff'; // transparência vira branco, como o convert('RGB') faria em fundo claro
  ctx.fillRect(0, 0, larg, alt);
  ctx.drawImage(fonte, 0, 0);
  fonte.close?.();
  const rgba = ctx.getImageData(0, 0, larg, alt).data;

  const rgb = redimensionarComoPillow(rgba, larg, alt, LADO, LADO);
  const total = LADO * LADO;
  const tensor = new Float32Array(3 * total);
  for (let i = 0; i < total; i++) {
    tensor[i] = rgb[i * 3] / 255;
    tensor[total + i] = rgb[i * 3 + 1] / 255;
    tensor[2 * total + i] = rgb[i * 3 + 2] / 255;
  }
  return { tensor, rgb, largura: larg, altura: alt };
}

/** Uma das oito simetrias do quadrado (idêntica a aplicarSimetria do aplicativo). */
export function aplicarSimetria(tensor, indice) {
  if (indice === 0) return tensor;
  const rotacoes = indice >> 1;
  const espelhar = (indice & 1) === 1;
  const lado = LADO;
  const area = lado * lado;
  const saida = new Float32Array(tensor.length);
  for (let c = 0; c < 3; c++) {
    const base = c * area;
    for (let y = 0; y < lado; y++) {
      for (let x = 0; x < lado; x++) {
        let sx = x, sy = y;
        for (let r = 0; r < rotacoes; r++) {
          const tx = sx;
          sx = sy;
          sy = lado - 1 - tx;
        }
        if (espelhar) sx = lado - 1 - sx;
        saida[base + y * lado + x] = tensor[base + sy * lado + sx];
      }
    }
  }
  return saida;
}

// --------------------------------------------------------------- inferência

function softmax(logits) {
  const max = Math.max(...logits);
  const exp = logits.map((v) => Math.exp(v - max));
  const soma = exp.reduce((a, b) => a + b, 0);
  return exp.map((v) => v / soma);
}

function indiceMaior(v) {
  let m = 0;
  for (let i = 1; i < v.length; i++) if (v[i] > v[m]) m = i;
  return m;
}

async function rodar(modelo, tensor) {
  const entrada = new modelo.ort.Tensor('float32', tensor, [1, 3, LADO, LADO]);
  const saidas = await modelo.sessao.run({ [modelo.entrada]: entrada });
  const logits = Array.from(saidas.logits.data, Number);
  const camPlano = saidas.cam.data;
  if (logits.length !== ORDEM_DO_MODELO.length || camPlano.length !== ORDEM_DO_MODELO.length * LADO_CAM * LADO_CAM) {
    throw new ErroModelo('saidas', 'As saídas do modelo têm formato inesperado.');
  }
  const prob = softmax(logits);
  const melhor = indiceMaior(prob);
  const area = LADO_CAM * LADO_CAM;
  const cam = Float32Array.from(camPlano.subarray(melhor * area, (melhor + 1) * area));
  return { prob, cam };
}

const cedeAoNavegador = () => new Promise((r) => setTimeout(r, 0));

/**
 * Analisa uma imagem já preparada. Com `robustez`, repete nas oito simetrias
 * e informa a concordância e a média das probabilidades.
 */
export async function analisar(modelo, preparada, { robustez = false, aoProgredir } = {}) {
  const t0 = performance.now();
  const direta = await rodar(modelo, preparada.tensor);
  const melhor = indiceMaior(direta.prob);

  let probRobustez = null;
  let concordancia = null;
  if (robustez) {
    const soma = new Array(direta.prob.length).fill(0);
    let acordos = 0;
    for (let s = 0; s < 8; s++) {
      aoProgredir?.(s, 8);
      await cedeAoNavegador();
      const variante = s === 0 ? direta.prob : (await rodar(modelo, aplicarSimetria(preparada.tensor, s))).prob;
      if (indiceMaior(variante) === melhor) acordos++;
      for (let i = 0; i < soma.length; i++) soma[i] += variante[i] / 8;
    }
    aoProgredir?.(8, 8);
    probRobustez = soma;
    concordancia = acordos / 8;
  }

  const confianca = direta.prob[melhor];
  return {
    probabilidades: direta.prob,
    indice: melhor,
    sigla: ORDEM_DO_MODELO[melhor],
    confianca,
    cam: direta.cam,
    probRobustez,
    siglaRobustez: probRobustez ? ORDEM_DO_MODELO[indiceMaior(probRobustez)] : null,
    concordancia,
    instavel: confianca < 0.6 || (concordancia != null && concordancia < 0.75),
    milissegundos: Math.round(performance.now() - t0),
  };
}

// ------------------------------------------------------------ mapa de calor

// Rampa quente (aproximação de "inferno"), igual à do aplicativo: legível também
// em escala de cinza.
const RAMPA = [[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]];

function corDoValor(t) {
  const v = Math.min(Math.max(t, 0), 1) * (RAMPA.length - 1);
  const i = Math.min(Math.max(Math.floor(v), 0), RAMPA.length - 2);
  const f = v - i;
  const a = RAMPA[i], b = RAMPA[i + 1];
  return [0, 1, 2].map((k) => Math.round(a[k] + (b[k] - a[k]) * f));
}

function amostrar(mapa, x, y) {
  const fx = Math.min(Math.max(x * (LADO_CAM - 1), 0), LADO_CAM - 1);
  const fy = Math.min(Math.max(y * (LADO_CAM - 1), 0), LADO_CAM - 1);
  const x0 = Math.floor(fx), y0 = Math.floor(fy);
  const x1 = Math.min(x0 + 1, LADO_CAM - 1), y1 = Math.min(y0 + 1, LADO_CAM - 1);
  const dx = fx - x0, dy = fy - y0;
  const v00 = mapa[y0 * LADO_CAM + x0], v10 = mapa[y0 * LADO_CAM + x1];
  const v01 = mapa[y1 * LADO_CAM + x0], v11 = mapa[y1 * LADO_CAM + x1];
  return v00 * (1 - dx) * (1 - dy) + v10 * dx * (1 - dy) + v01 * (1 - dx) * dy + v11 * dx * dy;
}

export function desenharCampo(canvas, rgb) {
  canvas.width = LADO;
  canvas.height = LADO;
  const dados = new ImageData(LADO, LADO);
  for (let i = 0; i < LADO * LADO; i++) {
    dados.data[i * 4] = rgb[i * 3];
    dados.data[i * 4 + 1] = rgb[i * 3 + 1];
    dados.data[i * 4 + 2] = rgb[i * 3 + 2];
    dados.data[i * 4 + 3] = 255;
  }
  canvas.getContext('2d').putImageData(dados, 0, 0);
}

export function desenharSobreposicao(canvas, rgb, cam, opacidade = 0.45) {
  let maximo = 0;
  for (const v of cam) if (v > maximo) maximo = v;
  if (maximo <= 0) maximo = 1;
  const norm = Float32Array.from(cam, (v) => Math.max(0, v) / maximo);

  canvas.width = LADO;
  canvas.height = LADO;
  const dados = new ImageData(LADO, LADO);
  for (let y = 0; y < LADO; y++) {
    for (let x = 0; x < LADO; x++) {
      const i = (y * LADO + x) * 3;
      const intensidade = amostrar(norm, x / (LADO - 1), y / (LADO - 1));
      const cor = corDoValor(intensidade);
      const alfa = opacidade * intensidade;
      const o = (y * LADO + x) * 4;
      dados.data[o] = Math.round(rgb[i] * (1 - alfa) + cor[0] * alfa);
      dados.data[o + 1] = Math.round(rgb[i + 1] * (1 - alfa) + cor[1] * alfa);
      dados.data[o + 2] = Math.round(rgb[i + 2] * (1 - alfa) + cor[2] * alfa);
      dados.data[o + 3] = 255;
    }
  }
  canvas.getContext('2d').putImageData(dados, 0, 0);
}
