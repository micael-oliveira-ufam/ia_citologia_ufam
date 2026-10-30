// Identidade anônima do Firebase (sem conta Google e sem tela de login).
//
// Ao abrir o catálogo, o site cria (ou retoma) um usuário anônimo pela API REST
// do Firebase Authentication. O Firestore passa a enxergar um `uid` estável neste
// navegador: é ele que identifica "minhas lâminas" e "minhas avaliações".
// O token de renovação fica no localStorage; se o armazenamento do navegador for
// apagado, nasce um novo uid e as lâminas antigas deixam de ser editáveis aqui.
//
// Requer ativar o provedor "Anônimo" em Firebase > Authentication > Método de login.

import { FIREBASE } from './config.js?v=4';

const CHAVE_LS = 'citoatlas.anonimo.v1';
const chave = encodeURIComponent(FIREBASE.apiKey);

export class ErroAutenticacao extends Error {
  constructor(tipo, mensagem, detalhe) {
    super(mensagem);
    this.tipo = tipo; // 'desativado' | 'chave' | 'rede' | 'inesperado'
    this.detalhe = detalhe;
  }
}

let uid = null;
let idToken = null;
let renovacao = null;
let expira = 0;
let emAndamento = null;

function lerSalvo() {
  try {
    const j = JSON.parse(localStorage.getItem(CHAVE_LS) || 'null');
    if (j && typeof j.refresh === 'string' && j.refresh) return j;
  } catch (_) { /* sem armazenamento */ }
  return null;
}
function salvar() {
  try { localStorage.setItem(CHAVE_LS, JSON.stringify({ uid, refresh: renovacao })); } catch (_) { /* ignora */ }
}

function guardar(id, ren, segundos, novoUid) {
  idToken = id;
  renovacao = ren;
  if (novoUid) uid = novoUid;
  expira = Date.now() + (Number(segundos) || 3600) * 1000;
  salvar();
}

async function pedir(url, init) {
  let r;
  try {
    r = await fetch(url, init);
  } catch (e) {
    throw new ErroAutenticacao('rede', 'Sem conexão com o Firebase. Verifique a internet e tente de novo.', String(e));
  }
  if (r.ok) return r.json();
  let corpo = '';
  try { corpo = await r.text(); } catch (_) { /* ignora */ }
  const e = new ErroAutenticacao('inesperado', `O Firebase Authentication respondeu com erro ${r.status}.`, corpo.slice(0, 300));
  e.status = r.status;
  throw e;
}

async function criarAnonimo() {
  try {
    const j = await pedir(`https://identitytoolkit.googleapis.com/v1/accounts:signUp?key=${chave}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ returnSecureToken: true }),
    });
    guardar(j.idToken, j.refreshToken, j.expiresIn, j.localId);
  } catch (e) {
    if (e instanceof ErroAutenticacao && e.status) {
      const d = e.detalhe || '';
      if (/OPERATION_NOT_ALLOWED|ADMIN_ONLY_OPERATION/i.test(d)) {
        throw new ErroAutenticacao('desativado',
          'O login anônimo não está ativado no Firebase. No console do Firebase, abra Authentication > ' +
          'Método de login e ative o provedor "Anônimo".', d);
      }
      if (/API_KEY|referer|blocked|PERMISSION_DENIED/i.test(d) || e.status === 403) {
        throw new ErroAutenticacao('chave',
          'A chave de API do projeto bloqueia este endereço ou o serviço. No Google Cloud, em Credenciais, ' +
          'autorize este site na chave do navegador (e a API Identity Toolkit).', d);
      }
    }
    throw e;
  }
}

async function renovar() {
  const j = await pedir(`https://securetoken.googleapis.com/v1/token?key=${chave}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: `grant_type=refresh_token&refresh_token=${encodeURIComponent(renovacao || '')}`,
  });
  guardar(j.id_token, j.refresh_token, j.expires_in, j.user_id);
}

/** Garante uma identidade anônima válida. Retorna o uid. */
export function entrar() {
  if (uid && idToken && expira - Date.now() > 120000) return Promise.resolve(uid);
  if (emAndamento) return emAndamento;
  emAndamento = (async () => {
    try {
      if (!renovacao) {
        const s = lerSalvo();
        if (s) { renovacao = s.refresh; uid = s.uid; }
      }
      if (renovacao) {
        try {
          await renovar();
          return uid;
        } catch (e) {
          if (e instanceof ErroAutenticacao && e.tipo === 'rede') throw e;
          renovacao = null; uid = null; // token recusado: cria outra identidade
        }
      }
      await criarAnonimo();
      return uid;
    } finally {
      emAndamento = null;
    }
  })();
  return emAndamento;
}

/** Token do Firebase válido (entra se preciso). */
export async function token() {
  await entrar();
  return idToken;
}

export const uidAtual = () => uid;
