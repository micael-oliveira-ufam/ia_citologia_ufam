// Utilitários compartilhados. Regra de ouro deste site: nada que veio do banco de
// dados entra no HTML como texto marcado (innerHTML). O Firestore do CitoAtlas
// aceita escrita de qualquer pessoa, então todo campo deve ser tratado como
// não confiável e inserido só por textContent / atributos validados.

import { CATALOGO, REPOSITORIO } from './config.js';

export const $ = (seletor, raiz = document) => raiz.querySelector(seletor);

/** Cria elemento. Filhos string viram texto puro. */
export function el(tag, atributos = {}, ...filhos) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(atributos)) {
    if (v === false || v == null) continue;
    if (k === 'class') e.className = v;
    else if (k.startsWith('on') && typeof v === 'function') e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? '' : String(v));
  }
  for (const f of filhos.flat()) {
    if (f == null || f === false) continue;
    e.append(f instanceof Node ? f : document.createTextNode(String(f)));
  }
  return e;
}

export function limpar(no) {
  while (no.firstChild) no.removeChild(no.firstChild);
}

export function bytesLegiveis(n) {
  if (!Number.isFinite(n)) return '?';
  if (n < 1024 * 1024) return `${Math.max(1, Math.round(n / 1024))} KB`;
  return `${(n / 1048576).toFixed(1).replace('.', ',')} MB`;
}

export function percentual(fracao, casas = 1) {
  return `${(fracao * 100).toFixed(casas).replace('.', ',')}%`;
}

export function formatarData(valor) {
  if (valor == null || valor === '') return '—';
  let d;
  if (typeof valor === 'number') d = new Date(valor);
  else if (/^\d{4}-\d{2}-\d{2}$/.test(String(valor))) {
    const [a, m, dia] = String(valor).split('-').map(Number);
    d = new Date(a, m - 1, dia);
  } else d = new Date(valor);
  if (Number.isNaN(d.getTime())) return '—';
  return d.toLocaleDateString('pt-BR');
}

/**
 * ID exibido de uma lâmina. O campo "patientId" é texto livre no CitoAtlas
 * (o formulário permite sobrescrever o L-0001 automático). Num site público, só
 * mostramos IDs no formato do sistema; qualquer outra coisa pode ser nome ou
 * prontuário de paciente e vira um código curto derivado, que não revela o
 * original.
 */
export function idPublico(patientId) {
  const bruto = String(patientId ?? '').trim();
  if (/^L-\d{1,6}$/i.test(bruto)) return { texto: bruto.toUpperCase(), oculto: false };
  if (CATALOGO.exibirIdLivre && bruto) return { texto: bruto.slice(0, 40), oculto: false };
  let h = 0x811c9dc5;
  for (let i = 0; i < bruto.length; i++) {
    h ^= bruto.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return { texto: `Lâmina ${h.toString(16).slice(0, 4).toUpperCase()}`, oculto: true };
}

/** Só aceita imagem embutida (jpeg/png/webp em base64) ou https. */
export function urlImagemSegura(url) {
  if (typeof url !== 'string') return null;
  if (/^data:image\/(jpeg|png|webp);base64,[A-Za-z0-9+/=]+$/.test(url)) return url;
  try {
    const u = new URL(url);
    if (u.protocol === 'https:') return u.href;
  } catch (_) { /* inválida */ }
  return null;
}

/** Preenche os links do repositório (data-repo). */
export function iniciarPagina() {
  for (const a of document.querySelectorAll('[data-repo]')) {
    a.href = REPOSITORIO + (a.dataset.repo || '');
  }
}
