// Configuração pública do projeto Firebase "citoatlas" (banco Firestore padrão).
// O site usa só a API REST do Firestore e do Authentication (sem o SDK do Firebase e
// sem o Google Analytics, para não rastrear quem visita o site).
// Configuração pública do projeto Firebase do CitoAtlas (a mesma chave web que já
// vai no código do CitoAtlas original; chaves web do Firebase identificam o
// projeto, não são segredo). Quem protege os dados são as REGRAS do Firestore:
// veja firebase/firestore.rules.sugerido.
export const FIREBASE = {
  projectId: "citoatlas",
  databaseId: "(default)",
  apiKey: "AIzaSyDIye72T8UWKS04vheKeFlQhZiofFoDY14",
  authDomain: "citoatlas.firebaseapp.com",
  storageBucket: "citoatlas.firebasestorage.app",
  appId: "1:764790014755:web:6f9c761987f5f32fc40296",
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
