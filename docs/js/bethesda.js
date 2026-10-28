// Categorias do Sistema Bethesda usadas pelo modelo. Os dados vêm de dados.js (gerado a
// partir de ferramentas/conteudo_site.py, a mesma fonte do HTML).
//
// ORDEM_DO_MODELO corresponde às pastas do treinamento (Carcinoma, HSIL, LSIL, Normal) em
// ordem alfabética, como o ImageFolder as indexou. Mexer nela inverte diagnósticos sem erro
// visível; o manifesto do modelo também declara a ordem e a página recusa carregar um
// modelo cuja ordem seja diferente.
import { ORDEM_DO_MODELO, ORDEM_GRAVIDADE, CLASS_INFO, EXEMPLOS, LIMIAR_CONFIANCA, LIMIAR_CONCORDANCIA } from './dados.js';

export { ORDEM_DO_MODELO, ORDEM_GRAVIDADE, EXEMPLOS, LIMIAR_CONFIANCA, LIMIAR_CONCORDANCIA };

export const CATEGORIAS = Object.fromEntries(
  Object.entries(CLASS_INFO).map(([sigla, i]) => [sigla, { sigla, ...i }]),
);

export const POR_GRAVIDADE = ORDEM_GRAVIDADE.map((s) => CATEGORIAS[s]);
