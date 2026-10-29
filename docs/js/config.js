// Configuração pública do projeto Firebase do CitoAtlas (a mesma chave web que já
// vai no código do CitoAtlas original; chaves web do Firebase identificam o
// projeto, não são segredo). Quem protege os dados são as REGRAS do Firestore:
// veja firebase/firestore.rules.sugerido.
export const FIREBASE = {
  projectId: "gen-lang-client-0447277814",
  databaseId: "ai-studio-citoatlaslabseba-4d2d79ae-46ba-4b4b-8813-8448586da766",
  apiKey: "AIzaSyARQE9R_y8ZXGpl-rkWhpKiyp-wVwf3EpE",
};

// Repositório e site no GitHub (ajuste se o nome do repositório for outro).
export const REPOSITORIO = 'https://github.com/micael-oliveira-ufam/ia_citologia_ufam';

export const CATALOGO = {
  // Mostra o ID digitado pelo cadastrador mesmo quando não tem o formato L-0001.
  // Deixe false em site público: o campo é de texto livre e alguém pode ter
  // digitado nome ou prontuário de paciente.
  exibirIdLivre: false,
  tamanhoPagina: 24,
  votosParaConsenso: 3,
};

export const MODELO = {
  manifesto: new URL('../model/manifesto.json', import.meta.url).href,
};
