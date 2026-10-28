// Modelo de citologia no navegador: manifesto, download verificado, sessão ONNX,
// pré-processamento, inferência, mapa de calor e verificação de robustez.
//
// Tudo aqui espelha o aplicativo (app/lib/servicos/classificador.dart e
// imagem.dart) e o script de exportação (app/tools/export_onnx.py):
//   entrada  RGB em [0,1], NCHW, 1x3x224x224 (normalização ImageNet no grafo)
//   saídas   logits [1,4] e cam [1,4,7,7]
// O redimensionamento é o do treinamento e da plataforma Streamlit: Albumentations Resize,
// ou seja, cv2.INTER_LINEAR (bilinear de 2x2 vizinhos, sem suavização ao reduzir).
// O mapa de calor segue o do Streamlit: cv2.resize bilinear do mapa 7x7 até o tamanho
// da imagem, colormap INFERNO e mistura 0,6 imagem + 0,4 cor.

import { ORDEM_DO_MODELO, LIMIAR_CONFIANCA, LIMIAR_CONCORDANCIA } from './bethesda.js';

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
  const origem = new URL(urlManifesto).origin;
  const url = new URL(m.arquivo, urlManifesto);
  if (url.origin !== origem) throw ruim('o arquivo precisa estar no mesmo site');
  // Arquivos de 100 MB ou mais não cabem no GitHub: o modelo pode vir em partes,
  // concatenadas aqui e conferidas pelo SHA-256 do arquivo inteiro.
  let partes = [{ url: url.href, bytes: m.bytes }];
  if (m.partes !== undefined) {
    if (!Array.isArray(m.partes) || m.partes.length < 1 || m.partes.length > 20) throw ruim('partes');
    partes = m.partes.map((p) => {
      if (!p || typeof p.arquivo !== 'string' || !Number.isInteger(p.bytes) || p.bytes <= 0) throw ruim('parte inválida');
      const u = new URL(p.arquivo, urlManifesto);
      if (u.origin !== origem) throw ruim('as partes precisam estar no mesmo site');
      return { url: u.href, bytes: p.bytes };
    });
    if (partes.reduce((a, p) => a + p.bytes, 0) !== m.bytes) throw ruim('a soma das partes difere do tamanho');
  }
  return { ...m, urlArquivo: url.href, partes };
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

  const corpo = new Uint8Array(m.bytes);
  let recebido = 0;
  for (const parte of m.partes) {
    const url = m.partes.length === 1 ? urlVersionada(m) : `${parte.url}?v=${m.sha256.slice(0, 16)}`;
    let resp;
    try {
      resp = await fetch(url, { signal: sinal, cache: 'no-store' });
    } catch (e) {
      throw new ErroModelo('rede', 'O download do modelo falhou. Verifique a conexão e tente de novo.', String(e));
    }
    if (!resp.ok) {
      throw new ErroModelo('download', `O arquivo do modelo não foi encontrado (HTTP ${resp.status}).`);
    }
    const limiteParte = recebido + parte.bytes;
    const leitor = resp.body.getReader();
    for (;;) {
      const { done, value } = await leitor.read();
      if (done) break;
      if (recebido + value.length > limiteParte) {
        throw new ErroModelo('tamanho', 'O arquivo baixado é maior do que o manifesto declara. Descartado.');
      }
      corpo.set(value, recebido);
      recebido += value.length;
      aoProgredir?.(recebido, m.bytes);
    }
    if (recebido !== limiteParte) {
      throw new ErroModelo('tamanho', 'O download terminou incompleto. Tente de novo.');
    }
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

// ------------------------------------------- pré-processamento (cv2.INTER_LINEAR)

/**
 * Reproduz cv2.resize(..., INTER_LINEAR) para imagens de 8 bits: coordenada de origem
 * (x + 0,5) * escala - 0,5, pesos em ponto fixo de 11 bits e arredondamento no fim.
 * Não há suavização ao reduzir (só os 2x2 vizinhos contam), igual ao treinamento.
 */
export function redimensionarLinear(rgba, larg, alt, novaLarg, novaAlt) {
  const BITS = 11;
  const ESC = 1 << BITS;
  const eixo = (n, novo) => {
    const ini = new Int32Array(novo), fim = new Int32Array(novo), w1 = new Int32Array(novo);
    const escala = n / novo;
    for (let i = 0; i < novo; i++) {
      let f = (i + 0.5) * escala - 0.5;
      let s0 = Math.floor(f);
      f -= s0;
      if (s0 < 0) { f = 0; s0 = 0; }
      if (s0 >= n - 1) { f = 0; s0 = n - 1; }
      ini[i] = s0;
      fim[i] = Math.min(s0 + 1, n - 1);
      w1[i] = Math.round(f * ESC);
    }
    return { ini, fim, w1 };
  };
  const X = eixo(larg, novaLarg);
  const Y = eixo(alt, novaAlt);
  const saida = new Uint8Array(novaAlt * novaLarg * 3);
  const meio = 1 << (2 * BITS - 1);
  for (let y = 0; y < novaAlt; y++) {
    const r0 = Y.ini[y] * larg, r1 = Y.fim[y] * larg;
    const b1 = Y.w1[y], b0 = ESC - b1;
    for (let x = 0; x < novaLarg; x++) {
      const a1 = X.w1[x], a0 = ESC - a1;
      const p00 = (r0 + X.ini[x]) * 4, p01 = (r0 + X.fim[x]) * 4;
      const p10 = (r1 + X.ini[x]) * 4, p11 = (r1 + X.fim[x]) * 4;
      const o = (y * novaLarg + x) * 3;
      for (let c = 0; c < 3; c++) {
        const topo = rgba[p00 + c] * a0 + rgba[p01 + c] * a1;
        const base = rgba[p10 + c] * a0 + rgba[p11 + c] * a1;
        saida[o + c] = (topo * b0 + base * b1 + meio) >> (2 * BITS);
      }
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

/** Imagem (File/Blob) -> { tensor Float32 [3*224*224], rgb Uint8 [224*224*3], exibicao (canvas), largura, altura }. */
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

  const rgb = redimensionarLinear(rgba, larg, alt, LADO, LADO);

  // Cópia para exibição (o mapa de calor é desenhado sobre o campo original, como no
  // Streamlit), limitada a 1024 px no maior lado para não pesar no celular.
  const reducao = Math.min(1, 1024 / Math.max(larg, alt));
  const exibicao = document.createElement('canvas');
  exibicao.width = Math.max(1, Math.round(larg * reducao));
  exibicao.height = Math.max(1, Math.round(alt * reducao));
  exibicao.getContext('2d').drawImage(cv, 0, 0, exibicao.width, exibicao.height);
  const total = LADO * LADO;
  const tensor = new Float32Array(3 * total);
  for (let i = 0; i < total; i++) {
    tensor[i] = rgb[i * 3] / 255;
    tensor[total + i] = rgb[i * 3 + 1] / 255;
    tensor[2 * total + i] = rgb[i * 3 + 2] / 255;
  }
  return { tensor, rgb, exibicao, largura: larg, altura: alt };
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
    instavel: confianca < LIMIAR_CONFIANCA || (concordancia != null && concordancia < LIMIAR_CONCORDANCIA),
    milissegundos: Math.round(performance.now() - t0),
  };
}

// ------------------------------------------------------------ mapa de calor

// Colormap INFERNO (256 entradas, as mesmas de cv2.COLORMAP_INFERNO), em hexadecimal.
const INFERNO_HEX = "00000401000501010601010802010a02020c02020e03021004031204031405041706041907051b08051d09061f0a07220b07240c08260d08290e092b10092d110a30120a32140b34150b37160b39180c3c190c3e1b0c411c0c431e0c451f0c48210c4a230c4c240c4f260c51280b53290b552b0b572d0b592f0a5b310a5c320a5e340a5f3609613809623909633b09643d09653e0966400a67420a68440a68450a69470b6a490b6a4a0c6b4c0c6b4d0d6c4f0d6c510e6c520e6d540f6d550f6d57106e59106e5a116e5c126e5d126e5f136e61136e62146e64156e65156e67166e69166e6a176e6c186e6d186e6f196e71196e721a6e741a6e751b6e771c6d781c6d7a1d6d7c1d6d7d1e6d7f1e6c801f6c82206c84206b85216b87216b88226a8a226a8c23698d23698f24699025689225689326679526679727669827669a28659b29649d29649f2a63a02a63a22b62a32c61a52c60a62d60a82e5fa92e5eab2f5ead305dae305cb0315bb1325ab3325ab43359b63458b73557b93556ba3655bc3754bd3853bf3952c03a51c13a50c33b4fc43c4ec63d4dc73e4cc83f4bca404acb4149cc4248ce4347cf4446d04545d24644d34743d44842d54a41d74b3fd84c3ed94d3dda4e3cdb503bdd513ade5238df5337e05536e15635e25734e35933e45a31e55c30e65d2fe75e2ee8602de9612bea632aeb6429eb6628ec6726ed6925ee6a24ef6c23ef6e21f06f20f1711ff1731df2741cf3761bf37819f47918f57b17f57d15f67e14f68013f78212f78410f8850ff8870ef8890cf98b0bf98c0af98e09fa9008fa9207fa9407fb9606fb9706fb9906fb9b06fb9d07fc9f07fca108fca309fca50afca60cfca80dfcaa0ffcac11fcae12fcb014fcb216fcb418fbb61afbb81dfbba1ffbbc21fbbe23fac026fac228fac42afac62df9c72ff9c932f9cb35f8cd37f8cf3af7d13df7d340f6d543f6d746f5d949f5db4cf4dd4ff4df53f4e156f3e35af3e55df2e661f2e865f2ea69f1ec6df1ed71f1ef75f1f179f2f27df2f482f3f586f3f68af4f88ef5f992f6fa96f8fb9af9fc9dfafda1fcffa4";
const INFERNO = Uint8Array.from({ length: 768 }, (_, i) => parseInt(INFERNO_HEX.substr(i * 2, 2), 16));

/** Mapa 7x7 -> w x h, bilinear como cv2.resize (coordenadas de centro de pixel). */
function ampliarMapa(mapa, w, h) {
  const n = LADO_CAM;
  const saida = new Float32Array(w * h);
  const xs = new Int32Array(w), xs1 = new Int32Array(w), fx = new Float32Array(w);
  for (let x = 0; x < w; x++) {
    let f = (x + 0.5) * (n / w) - 0.5;
    let i0 = Math.floor(f);
    f -= i0;
    if (i0 < 0) { f = 0; i0 = 0; }
    if (i0 >= n - 1) { f = 0; i0 = n - 1; }
    xs[x] = i0; xs1[x] = Math.min(i0 + 1, n - 1); fx[x] = f;
  }
  for (let y = 0; y < h; y++) {
    let g = (y + 0.5) * (n / h) - 0.5;
    let j0 = Math.floor(g);
    g -= j0;
    if (j0 < 0) { g = 0; j0 = 0; }
    if (j0 >= n - 1) { g = 0; j0 = n - 1; }
    const j1 = Math.min(j0 + 1, n - 1);
    for (let x = 0; x < w; x++) {
      const a = mapa[j0 * n + xs[x]] * (1 - fx[x]) + mapa[j0 * n + xs1[x]] * fx[x];
      const b = mapa[j1 * n + xs[x]] * (1 - fx[x]) + mapa[j1 * n + xs1[x]] * fx[x];
      saida[y * w + x] = a * (1 - g) + b * g;
    }
  }
  return saida;
}

/** Copia o campo original (canvas de exibição) para o canvas de destino. */
export function desenharCampo(canvas, origem) {
  canvas.width = origem.width;
  canvas.height = origem.height;
  canvas.getContext('2d').drawImage(origem, 0, 0);
}

/** Grad-CAM sobreposto: 0,6 · imagem + 0,4 · INFERNO(mapa), como no Streamlit. */
export function desenharSobreposicao(canvas, origem, cam) {
  const w = origem.width, h = origem.height;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(origem, 0, 0);
  const dados = ctx.getImageData(0, 0, w, h);
  let maximo = 0;
  for (const v of cam) if (v > maximo) maximo = v;
  const norm = Float32Array.from(cam, (v) => (maximo > 0 ? Math.max(0, v) / maximo : 0));
  const grande = ampliarMapa(norm, w, h);
  const px = dados.data;
  for (let i = 0; i < w * h; i++) {
    const idx = Math.min(255, Math.floor(255 * grande[i])) * 3; // np.uint8(255 * mapa)
    const o = i * 4;
    px[o] = Math.round(px[o] * 0.6 + INFERNO[idx] * 0.4);
    px[o + 1] = Math.round(px[o + 1] * 0.6 + INFERNO[idx + 1] * 0.4);
    px[o + 2] = Math.round(px[o + 2] * 0.6 + INFERNO[idx + 2] * 0.4);
  }
  ctx.putImageData(dados, 0, 0);
}
