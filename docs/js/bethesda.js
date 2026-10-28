// Categorias do Sistema Bethesda usadas pelo modelo. Textos iguais aos do
// aplicativo (app/lib/data/bethesda.dart).
//
// ORDEM_DO_MODELO corresponde às pastas do treinamento (Carcinoma, HSIL, LSIL,
// Normal) em ordem alfabética, como o ImageFolder as indexou. Mexer aqui inverte
// diagnósticos sem erro visível; o manifesto do modelo também declara a ordem e
// a página recusa carregar um modelo cuja ordem seja diferente.

export const ORDEM_DO_MODELO = ['SCC', 'HSIL', 'LSIL', 'NILM'];

export const CATEGORIAS = {
  NILM: {
    sigla: 'NILM',
    nome: 'Negativo para lesão intraepitelial ou malignidade',
    explicacao:
      'Nenhuma alteração celular sugestiva de lesão precursora ou câncer no campo analisado. ' +
      'Corresponde ao resultado esperado na maioria dos exames de rastreio.',
    gravidade: 0,
  },
  LSIL: {
    sigla: 'LSIL',
    nome: 'Lesão intraepitelial escamosa de baixo grau',
    explicacao:
      'Alterações associadas à infecção por HPV, como coilocitose. A maioria regride ' +
      'espontaneamente, mas exige seguimento.',
    gravidade: 1,
  },
  HSIL: {
    sigla: 'HSIL',
    nome: 'Lesão intraepitelial escamosa de alto grau',
    explicacao:
      'Lesão precursora com risco relevante de progressão. Indica necessidade de colposcopia ' +
      'e conduta específica.',
    gravidade: 2,
  },
  SCC: {
    sigla: 'SCC',
    nome: 'Carcinoma de células escamosas',
    explicacao:
      'Achados citológicos compatíveis com carcinoma invasor. Encaminhamento imediato para ' +
      'investigação.',
    gravidade: 3,
  },
};

export const POR_GRAVIDADE = Object.values(CATEGORIAS).sort((a, b) => a.gravidade - b.gravidade);

export const EXEMPLOS = [
  { arquivo: 'exemplos/cyto5950.jpg', titulo: 'Campo 1', descricao: 'Escamosas maduras, núcleos pequenos, flora bacilar' },
  { arquivo: 'exemplos/cyto2870.jpg', titulo: 'Campo 2', descricao: 'Colunares endocervicais em paliçada, fundo inflamatório' },
  { arquivo: 'exemplos/cyt10131a.jpg', titulo: 'Campo 3', descricao: 'Halos perinucleares indicados por setas, exsudato abundante' },
  { arquivo: 'exemplos/cyt16243.jpg', titulo: 'Campo 4', descricao: 'Agrupamentos densos, alta relação núcleo/citoplasma' },
];
