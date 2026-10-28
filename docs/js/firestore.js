// Leitura do Firestore pela API REST (sem SDK). Só leitura: este site não escreve.

import { FIREBASE } from './config.js';

const BASE =
  `https://firestore.googleapis.com/v1/projects/${FIREBASE.projectId}` +
  `/databases/${encodeURIComponent(FIREBASE.databaseId)}/documents`;

export class ErroFirestore extends Error {
  constructor(tipo, mensagem, detalhe) {
    super(mensagem);
    this.tipo = tipo; // 'rede' | 'permissao' | 'chave' | 'inesperado'
    this.detalhe = detalhe;
  }
}

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

async function buscar(url, { sinal, tempoMs = 25000 } = {}) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(new Error('tempo esgotado')), tempoMs);
  sinal?.addEventListener('abort', () => ctl.abort(sinal.reason), { once: true });
  let resp;
  try {
    resp = await fetch(url, { signal: ctl.signal });
  } catch (e) {
    throw new ErroFirestore('rede', 'Sem conexão com o banco de dados do catálogo.', String(e));
  } finally {
    clearTimeout(timer);
  }
  if (resp.ok) return resp.json();

  let corpo = '';
  try { corpo = await resp.text(); } catch (_) { /* ignora */ }
  if (resp.status === 403 && /referer|API_KEY_HTTP_REFERRER_BLOCKED|blocked/i.test(corpo)) {
    throw new ErroFirestore(
      'chave',
      'A chave de API do projeto bloqueia este endereço. No Google Cloud, em Credenciais, ' +
      'autorize este site na chave do navegador.',
      corpo.slice(0, 300),
    );
  }
  if (resp.status === 403 || resp.status === 401) {
    throw new ErroFirestore(
      'permissao',
      'O banco de dados recusou a leitura pública. As regras do Firestore podem ter sido restringidas.',
      corpo.slice(0, 300),
    );
  }
  throw new ErroFirestore('inesperado', `O banco de dados respondeu com erro ${resp.status}.`, corpo.slice(0, 300));
}

/** Uma página de lâminas, da mais nova para a mais antiga. */
export async function listarLaminas({ tamanho, token, sinal }) {
  const p = new URLSearchParams({
    pageSize: String(tamanho),
    orderBy: 'createdAt desc',
    key: FIREBASE.apiKey,
  });
  if (token) p.set('pageToken', token);
  const dados = await buscar(`${BASE}/slides?${p}`, { sinal });
  return {
    itens: (dados.documents || []).map(documentoParaObjeto),
    proximo: dados.nextPageToken || null,
  };
}

/**
 * Número de avaliações por lâmina. Pede ao servidor SÓ o campo slideId (máscara):
 * os documentos de avaliação têm nome e e-mail dos avaliadores, que este site
 * não baixa nem exibe.
 */
export async function contarVotos({ sinal, maxPaginas = 40 } = {}) {
  const contagem = new Map();
  let token = null;
  for (let i = 0; i < maxPaginas; i++) {
    const p = new URLSearchParams({ pageSize: '300', key: FIREBASE.apiKey });
    p.append('mask.fieldPaths', 'slideId');
    if (token) p.set('pageToken', token);
    const dados = await buscar(`${BASE}/evaluations?${p}`, { sinal });
    for (const d of dados.documents || []) {
      const id = d.fields?.slideId?.stringValue;
      if (id) contagem.set(id, (contagem.get(id) || 0) + 1);
    }
    token = dados.nextPageToken || null;
    if (!token) return { contagem, completo: true };
  }
  return { contagem, completo: false };
}
