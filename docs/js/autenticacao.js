// Login com Google para o catálogo (somente leitura).
//
// O Firestore do CitoAtlas exige usuário autenticado. O botão do Google
// (Google Identity Services) entrega um token do Google; ele é trocado por um
// token do Firebase pela API REST do Firebase Auth, que o Firestore entende.
// Os tokens ficam só na memória desta aba: ao recarregar a página, é preciso
// entrar de novo.

import { FIREBASE, GOOGLE_CLIENTE_WEB } from './config.js';

const ouvintes = new Set();
let usuario = null;
let idToken = null;
let renovacao = null;
let expira = 0;

export const usuarioAtual = () => usuario;
export const aoMudar = (fn) => { ouvintes.add(fn); return () => ouvintes.delete(fn); };
const avisar = () => ouvintes.forEach((f) => { try { f(usuario); } catch (_) { /* ignora */ } });

const chave = encodeURIComponent(FIREBASE.apiKey);

function guardar(id, ren, segundos) {
  idToken = id;
  renovacao = ren;
  expira = Date.now() + (Number(segundos) || 3600) * 1000;
}

async function trocar(credencialGoogle) {
  const r = await fetch(`https://identitytoolkit.googleapis.com/v1/accounts:signInWithIdp?key=${chave}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      postBody: `id_token=${encodeURIComponent(credencialGoogle)}&providerId=google.com`,
      requestUri: 'http://localhost',
      returnIdpCredential: true,
      returnSecureToken: true,
    }),
  });
  if (!r.ok) {
    throw new Error(
      `O Firebase recusou o login (${r.status}). Confira se o provedor Google está ativado ` +
      'no Firebase Authentication.');
  }
  const j = await r.json();
  const email = j.email || '';
  usuario = {
    uid: j.localId,
    email,
    nome: (j.displayName || '').trim() || (email.split('@')[0] || 'Citopatologista'),
  };
  guardar(j.idToken, j.refreshToken, j.expiresIn);
}

async function renovar() {
  const r = await fetch(`https://securetoken.googleapis.com/v1/token?key=${chave}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: `grant_type=refresh_token&refresh_token=${encodeURIComponent(renovacao || '')}`,
  });
  if (!r.ok) throw new Error('renovação recusada');
  const j = await r.json();
  guardar(j.id_token, j.refresh_token, j.expires_in);
}

/** Token do Firebase válido, ou null se ninguém entrou. */
export async function token() {
  if (!usuario) return null;
  if (idToken && expira - Date.now() > 120000) return idToken;
  try {
    await renovar();
  } catch (_) {
    sair();
    return null;
  }
  return idToken;
}

export function sair() {
  usuario = null; idToken = null; renovacao = null; expira = 0;
  try { window.google?.accounts.id.disableAutoSelect(); } catch (_) { /* ignora */ }
  avisar();
}

function carregarGIS() {
  return new Promise((ok, falha) => {
    if (window.google?.accounts?.id) return ok();
    const s = document.createElement('script');
    s.src = 'https://accounts.google.com/gsi/client';
    s.async = true;
    s.onload = () => ok();
    s.onerror = () => falha(new Error('Não foi possível carregar o login do Google (sem conexão ou bloqueado).'));
    document.head.append(s);
  });
}

/** Desenha o botão "Entrar com o Google" em [alvo]. */
export async function montarBotao(alvo, aoErro) {
  try {
    await carregarGIS();
    window.google.accounts.id.initialize({
      client_id: GOOGLE_CLIENTE_WEB,
      auto_select: false,
      callback: async (resp) => {
        try {
          await trocar(resp.credential);
          avisar();
        } catch (e) {
          aoErro?.(e);
        }
      },
    });
    window.google.accounts.id.renderButton(alvo, {
      theme: 'outline', size: 'large', text: 'signin_with', locale: 'pt-BR',
    });
  } catch (e) {
    aoErro?.(e);
  }
}
