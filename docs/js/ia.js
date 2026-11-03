// IA interna (o mesmo modelo ONNX da aba Análise) para uso em qualquer página: carrega o
// modelo uma única vez (cache do navegador, SHA-256 conferido) e classifica imagens em lote.
// Nenhuma imagem sai do navegador.

import { MODELO } from './config.js';
import { bytesLegiveis } from './comum.js';
import {
  lerManifesto, obterModelo, criarSessao, modeloEstaEmCache, prepararImagem, analisar, ErroModelo,
} from './modelo.js';

let manifesto = null;
let sessao = null;
let carregando = null;

/** Situação do modelo sem baixar nada: { tamanho, versao, emCache } ou lança ErroModelo. */
export async function situacaoModelo() {
  manifesto ??= await lerManifesto(MODELO.manifesto);
  return {
    tamanho: manifesto.bytes,
    tamanhoTexto: bytesLegiveis(manifesto.bytes),
    versao: manifesto.versao,
    emCache: !!sessao || (await modeloEstaEmCache(manifesto)),
    pronto: !!sessao,
  };
}

/** Baixa (se preciso), confere e abre o modelo. `aoProgredir(recebido, total)`. */
export function carregarModelo({ aoProgredir } = {}) {
  if (sessao) return Promise.resolve(sessao);
  carregando ??= (async () => {
    manifesto ??= await lerManifesto(MODELO.manifesto);
    const buffer = await obterModelo(manifesto, { aoProgredir });
    sessao = await criarSessao(buffer);
    return sessao;
  })().finally(() => { carregando = null; });
  return carregando;
}

export const modeloPronto = () => !!sessao;

/**
 * Classifica uma lista de imagens (Blob/File) em sequência, sem travar a página.
 * `aoItem(i, resultado|null, erro|null)` é chamado a cada imagem; devolve o vetor final.
 * Resultado: { sigla, confianca, probabilidades, instavel, concordancia, milissegundos, miniatura }.
 */
export async function classificarLote(blobs, { robustez = false, aoItem, sinal } = {}) {
  if (!sessao) throw new ErroModelo('modelo', 'O modelo ainda não foi carregado.');
  const saida = [];
  for (let i = 0; i < blobs.length; i++) {
    if (sinal?.aborted) break;
    let r = null;
    let erro = null;
    try {
      const preparada = await prepararImagem(blobs[i]);
      const a = await analisar(sessao, preparada, { robustez });
      const mini = document.createElement('canvas');
      const esc = 96 / Math.max(preparada.exibicao.width, preparada.exibicao.height);
      mini.width = Math.max(1, Math.round(preparada.exibicao.width * esc));
      mini.height = Math.max(1, Math.round(preparada.exibicao.height * esc));
      mini.getContext('2d').drawImage(preparada.exibicao, 0, 0, mini.width, mini.height);
      r = {
        sigla: a.sigla, confianca: a.confianca, probabilidades: a.probabilidades, cam: a.cam,
        instavel: a.instavel, concordancia: a.concordancia, milissegundos: a.milissegundos,
        miniatura: mini.toDataURL('image/jpeg', 0.7),
      };
    } catch (e) {
      erro = e instanceof ErroModelo ? e.message : `Falha ao analisar: ${e}`;
    }
    saida.push({ resultado: r, erro });
    aoItem?.(i, r, erro);
    await new Promise((ok) => setTimeout(ok, 0)); // devolve o controle à página
  }
  return saida;
}

/** Converte uma imagem de data URL em Blob (para classificar lâminas do catálogo). */
export async function blobDeDataUrl(url) {
  const r = await fetch(url);
  return r.blob();
}

/** CSV simples (UTF-8 com BOM, aspas escapadas) a partir de linhas de texto. */
export function csv(linhas) {
  const esc = (v) => `"${String(v ?? '').replace(/"/g, '""')}"`;
  return '﻿' + linhas.map((l) => l.map(esc).join(';')).join('\r\n');
}

export function baixarTexto(nome, texto, tipo = 'text/csv;charset=utf-8') {
  const url = URL.createObjectURL(new Blob([texto], { type: tipo }));
  const a = Object.assign(document.createElement('a'), { href: url, download: nome });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}
