// GERADO por ferramentas/gerar_site.py a partir de ferramentas/conteudo_site.py. Não edite à mão.
export const ORDEM_DO_MODELO = ["SCC", "HSIL", "LSIL", "NILM"];
export const ORDEM_GRAVIDADE = ["NILM", "LSIL", "HSIL", "SCC"];
export const LIMIAR_CONFIANCA = 0.6;
export const LIMIAR_CONCORDANCIA = 0.75;
export const CLASS_INFO = {
  "NILM": {
    "nome": "Negativo para lesão intraepitelial ou malignidade",
    "cor": "#2E7D62",
    "gravidade": 0,
    "resumo": "Sem alterações sugestivas de lesão precursora ou câncer."
  },
  "LSIL": {
    "nome": "Lesão intraepitelial escamosa de baixo grau",
    "cor": "#C08A1E",
    "gravidade": 1,
    "resumo": "Alterações associadas ao HPV. Maioria regride, mas exige seguimento."
  },
  "HSIL": {
    "nome": "Lesão intraepitelial escamosa de alto grau",
    "cor": "#D9642E",
    "gravidade": 2,
    "resumo": "Lesão precursora com risco relevante de progressão."
  },
  "SCC": {
    "nome": "Carcinoma de células escamosas",
    "cor": "#B23A3A",
    "gravidade": 3,
    "resumo": "Achados compatíveis com carcinoma invasor."
  }
};
export const EXEMPLOS = [
  {
    "arquivo": "exemplos/cyto5940.jpg",
    "titulo": "Ectocérvice normal",
    "referencia": "NILM",
    "descricao": "Ectocérvice normal: células escamosas intermediárias e superficiais, basofílicas ou eosinofílicas. Presença de alguns polimorfonucleares. (obj. 10x)"
  },
  {
    "arquivo": "exemplos/cyt14686a.jpg",
    "titulo": "Carcinoma invasivo",
    "referencia": "SCC",
    "descricao": "Carcinoma de células escamosas invasivo: agrupamento de células malignas pleomórficas, predominantemente pouco diferenciadas, e células isoladas mais diferenciadas, necróticas ou queratinizadas, com formas anômalas (elipses). Inflamação, sangue e necrose ao fundo. (obj. 20x)"
  },
  {
    "arquivo": "exemplos/cyto5950.jpg",
    "titulo": "Campo adicional 1",
    "referencia": null,
    "descricao": "Escamosas maduras, núcleos pequenos, flora bacilar."
  },
  {
    "arquivo": "exemplos/cyto2870.jpg",
    "titulo": "Campo adicional 2",
    "referencia": null,
    "descricao": "Colunares endocervicais em paliçada, fundo inflamatório."
  },
  {
    "arquivo": "exemplos/cyt10131a.jpg",
    "titulo": "Campo adicional 3",
    "referencia": null,
    "descricao": "Halos perinucleares indicados por setas, exsudato abundante."
  },
  {
    "arquivo": "exemplos/cyt16243.jpg",
    "titulo": "Campo adicional 4",
    "referencia": null,
    "descricao": "Agrupamentos densos, alta relação núcleo/citoplasma."
  }
];
