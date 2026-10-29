// Banco de dados do catálogo (Firestore) pela API REST, sem SDK.
// Todas as chamadas levam o token da identidade anônima (autenticacao.js).
//
// Coleções (mesmas do CitoAtlas / aplicativo):
//   slides       patientId, imageUrl (data URL jpeg), classification, date (AAAA-MM-DD),
//                morphologyDescription, notes, creatorId, creatorName, createdAt, updatedAt
//   evaluations  id (= slideId_userId), slideId, userId, userName, userEmail,
//                classification, morphologyDescription, notes, createdAt

import { FIREBASE } from './config.js';
import { token as tokenFirebase, uidAtual, entrar, ErroAutenticacao } from './autenticacao.js?v=3';

const RAIZ =
  `https://firestore.googleapis.com/v1/projects/${FIREBASE.projectId}` +
  `/databases/${encodeURIComponent(FIREBASE.databaseId)}/documents`;
const CHAVE = encodeURIComponent(FIREBASE.apiKey);

export class ErroFirestore extends Error {
  constructor(tipo, mensagem, detalhe) {
    super(mensagem);
    this.tipo = tipo; // 'login' | 'rede' | 'permissao' | 'chave' | 'tamanho' | 'inesperado'
    this.detalhe = detalhe;
  }
}

// ----------------------------------------------------------- conversão de valores

function valor(v) {
  if (v == null) return null;
  if ('stringValue' in v) return v.stringValue;
  if ('integerValue' in v) return Number(v.integerValue);
  if ('doubleValue' in v) return v.doubleValue;
  if ('booleanValue' in v) return v.booleanValue;
  if ('timestampValue' in v) return Date.parse(v.timestampValue);
  if ('nullValue' in v) return null;
  if ('arrayValue' in v) return (v.arrayValue.values || []).map(valor);
  if ('mapValue' in v) return campos(v.mapValue.fields || {});
  return null;
}
function campos(f) {
  const o = {};
  for (const [k, v] of Object.entries(f)) o[k] = valor(v);
  return o;
}
export function documentoParaObjeto(doc) {
  return { id: doc.name.split('/').pop(), ...campos(doc.fields || {}) };
}

function codificar(obj) {
  const f = {};
  for (const [k, v] of Object.entries(obj)) {
    if (v == null) f[k] = { nullValue: null };
    else if (typeof v === 'number') f[k] = { integerValue: String(Math.trunc(v)) };
    else f[k] = { stringValue: String(v) };
  }
  return f;
}

// ---------------------------------------------------------------- requisição

async function chamar(metodo, caminho, { consulta = [], corpo, sinal, tempoMs = 30000 } = {}) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(new Error('tempo esgotado')), tempoMs);
  sinal?.addEventListener('abort', () => ctl.abort(sinal.reason), { once: true });
  let resp;
  try {
    let idToken;
    try {
      idToken = await tokenFirebase();
    } catch (e) {
      if (e instanceof ErroAutenticacao) {
        throw new ErroFirestore(e.tipo === 'rede' ? 'rede' : 'login', e.message, e.detalhe);
      }
      throw e;
    }
    const p = new URLSearchParams();
    for (const [k, v] of consulta) p.append(k, v);
    p.append('key', FIREBASE.apiKey);
    try {
      resp = await fetch(`${RAIZ}${caminho}?${p}`, {
        method: metodo,
        signal: ctl.signal,
        headers: {
          Authorization: `Bearer ${idToken}`,
          ...(corpo ? { 'Content-Type': 'application/json' } : {}),
        },
        body: corpo ? JSON.stringify(corpo) : undefined,
      });
    } catch (e) {
      throw new ErroFirestore('rede', 'Sem conexão com o banco de dados do catálogo.', String(e));
    }
  } finally {
    clearTimeout(timer);
  }
  if (resp.ok) return resp.status === 204 ? {} : resp.json();

  let txt = '';
  try { txt = await resp.text(); } catch (_) { /* ignora */ }
  const det = txt.slice(0, 300);
  if (resp.status === 403 && /referer|API_KEY_HTTP_REFERRER_BLOCKED|blocked/i.test(txt)) {
    throw new ErroFirestore('chave',
      'A chave de API do projeto bloqueia este endereço. No Google Cloud, em Credenciais, ' +
      'autorize este site na chave do navegador.', det);
  }
  if (resp.status === 403 || resp.status === 401) {
    throw new ErroFirestore('permissao',
      'O banco de dados recusou a operação. As regras do Firestore precisam permitir usuários ' +
      'autenticados (anônimos) — veja firebase/firestore.rules.sugerido.', det);
  }
  if (resp.status === 400 && /too large|exceeds|size/i.test(txt)) {
    throw new ErroFirestore('tamanho', 'A imagem ficou grande demais para o banco de dados. Use uma foto menor.', det);
  }
  throw new ErroFirestore('inesperado', `O banco de dados respondeu com erro ${resp.status}.`, det);
}

// ------------------------------------------------------------------- leitura

export { entrar, uidAtual };

/** Uma página de lâminas (ordenada da mais nova para a mais antiga dentro da página). */
export async function listarLaminas({ tamanho, token, sinal }) {
  const q = [['pageSize', String(tamanho)]];
  if (token) q.push(['pageToken', token]);
  const dados = await chamar('GET', '/slides', { consulta: q, sinal });
  // O banco não aceita orderBy na listagem: ordena aqui, dentro da página.
  const itens = (dados.documents || []).map(documentoParaObjeto)
    .sort((a, b) => (b.createdAt || 0) - (a.createdAt || 0));
  return { itens, proximo: dados.nextPageToken || null };
}

export async function obterLamina(id, { sinal } = {}) {
  const d = await chamar('GET', `/slides/${encodeURIComponent(id)}`, { sinal });
  return documentoParaObjeto(d);
}

/**
 * Número de avaliações por lâmina. Pede ao servidor SÓ o campo slideId (máscara):
 * os documentos de avaliação têm nome e classificação dos avaliadores, que este
 * site não baixa nem exibe.
 */
export async function contarVotos({ sinal, maxPaginas = 40 } = {}) {
  const contagem = new Map();
  let pagina = null;
  for (let i = 0; i < maxPaginas; i++) {
    const q = [['pageSize', '300'], ['mask.fieldPaths', 'slideId']];
    if (pagina) q.push(['pageToken', pagina]);
    const dados = await chamar('GET', '/evaluations', { consulta: q, sinal });
    for (const d of dados.documents || []) {
      const id = d.fields?.slideId?.stringValue;
      if (id) contagem.set(id, (contagem.get(id) || 0) + 1);
    }
    pagina = dados.nextPageToken || null;
    if (!pagina) return { contagem, completo: true };
  }
  return { contagem, completo: false };
}

async function consultar(colecao, campo, valorTexto, { mascara, limite = 500, sinal } = {}) {
  const sq = {
    from: [{ collectionId: colecao }],
    where: { fieldFilter: { field: { fieldPath: campo }, op: 'EQUAL', value: { stringValue: valorTexto } } },
    limit: limite,
  };
  if (mascara) sq.select = { fields: mascara.map((fieldPath) => ({ fieldPath })) };
  const dados = await chamar('POST', ':runQuery', { corpo: { structuredQuery: sq }, sinal });
  return (Array.isArray(dados) ? dados : []).filter((r) => r.document).map((r) => r.document);
}

/** Avaliações feitas por este navegador (só as próprias: userId == uid). */
export async function minhasAvaliacoes({ sinal } = {}) {
  await entrar();
  const docs = await consultar('evaluations', 'userId', uidAtual(), { sinal });
  const mapa = new Map();
  for (const d of docs) {
    const a = documentoParaObjeto(d);
    if (a.slideId) mapa.set(a.slideId, a);
  }
  return mapa;
}

/** Próximo ID livre no formato L-0001, a partir dos IDs já cadastrados. */
export async function proximoId({ sinal } = {}) {
  let maior = 0;
  let pagina = null;
  for (let i = 0; i < 40; i++) {
    const q = [['pageSize', '300'], ['mask.fieldPaths', 'patientId']];
    if (pagina) q.push(['pageToken', pagina]);
    const dados = await chamar('GET', '/slides', { consulta: q, sinal });
    for (const d of dados.documents || []) {
      const m = /^L-(\d{1,6})$/i.exec(d.fields?.patientId?.stringValue || '');
      if (m) maior = Math.max(maior, Number(m[1]));
    }
    pagina = dados.nextPageToken || null;
    if (!pagina) break;
  }
  return `L-${String(maior + 1).padStart(4, '0')}`;
}

// ------------------------------------------------------------------- escrita

const agora = () => Date.now();

/** Cadastra uma lâmina nova. `imagem` é um data URL JPEG já comprimido. */
export async function criarLamina({ patientId, imagem, classificacao, data, descricao, observacoes, nome }) {
  const uid = await entrar();
  const t = agora();
  const doc = await chamar('POST', '/slides', {
    corpo: {
      fields: codificar({
        patientId, imageUrl: imagem, classification: classificacao, date: data,
        morphologyDescription: descricao || '', notes: observacoes || '',
        creatorId: uid, creatorName: nome || 'Citopatologista', createdAt: t, updatedAt: t,
      }),
    },
  });
  return documentoParaObjeto(doc);
}

/** Atualiza campos da lâmina (só o criador consegue, pelas regras). */
export async function atualizarLamina(id, mudancas) {
  const m = {
    ...(mudancas.patientId !== undefined && { patientId: mudancas.patientId }),
    ...(mudancas.imagem !== undefined && { imageUrl: mudancas.imagem }),
    ...(mudancas.classificacao !== undefined && { classification: mudancas.classificacao }),
    ...(mudancas.data !== undefined && { date: mudancas.data }),
    ...(mudancas.descricao !== undefined && { morphologyDescription: mudancas.descricao }),
    ...(mudancas.observacoes !== undefined && { notes: mudancas.observacoes }),
    updatedAt: agora(),
  };
  const q = Object.keys(m).map((k) => ['updateMask.fieldPaths', k]);
  const doc = await chamar('PATCH', `/slides/${encodeURIComponent(id)}`, { consulta: q, corpo: { fields: codificar(m) } });
  return documentoParaObjeto(doc);
}

/** Exclui a lâmina e, se as regras deixarem, as avaliações dela. */
export async function excluirLamina(id) {
  await entrar();
  try {
    const avs = await consultar('evaluations', 'slideId', id, { mascara: ['slideId'] });
    await Promise.all(avs.map((d) =>
      chamar('DELETE', `/evaluations/${encodeURIComponent(d.name.split('/').pop())}`).catch(() => null)));
  } catch (_) { /* melhor esforço */ }
  await chamar('DELETE', `/slides/${encodeURIComponent(id)}`);
}

/** Grava (ou refaz) a avaliação cega deste navegador para uma lâmina. */
export async function salvarAvaliacao({ slideId, classificacao, descricao, observacoes, nome }) {
  const uid = await entrar();
  const id = `${slideId}_${uid}`;
  await chamar('PATCH', `/evaluations/${encodeURIComponent(id)}`, {
    corpo: {
      fields: codificar({
        id, slideId, userId: uid, userName: nome || 'Avaliador anônimo', userEmail: '',
        classification: classificacao, morphologyDescription: descricao || '', notes: observacoes || '',
        createdAt: agora(),
      }),
    },
  });
  return { id, slideId, userId: uid, classification: classificacao, morphologyDescription: descricao || '', notes: observacoes || '' };
}
