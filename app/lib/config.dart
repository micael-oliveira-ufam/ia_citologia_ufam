/// Mostra o selo "a confirmar" nos números de assets/dados/estatisticas.json
/// cujo campo "verificado" é false. Depois de conferir cada número na fonte
/// indicada, troque "verificado" para true no JSON (ou ponha esta constante em
/// false para esconder todos os selos).
const bool exibirSeloAConfirmar = true;

/// Pasta do GitHub Pages onde estão o modelo de citologia e o manifesto
/// (gerados por tools/publicar_modelo.py). O aplicativo NÃO embarca o modelo:
/// baixa daqui, confere o SHA-256 e guarda no aparelho.
///
/// Para trocar sem editar o código:
///   flutter build apk --dart-define=URL_BASE_MODELO=https://SEU-USUARIO.github.io/SEU-REPO/model/
/// O nome do repositório abaixo é uma suposição: ajuste se o seu for outro.
const String urlBaseModelo = String.fromEnvironment(
  'URL_BASE_MODELO',
  defaultValue: 'https://micael-oliveira-ufam.github.io/ia_citologia_ufam/model/',
);

/// Ordem das classes que o aplicativo espera do modelo. O manifesto precisa
/// declarar exatamente esta ordem, senão o modelo é recusado.
const List<String> classesEsperadas = ['SCC', 'HSIL', 'LSIL', 'NILM'];

/// Catálogo de lâminas (CitoAtlas): mesmo projeto Firebase do CitoAtlas web.
/// A chave de API é a chave web pública do projeto (identifica o projeto, não
/// é segredo); quem protege os dados são as regras do Firestore
/// (veja firebase/firestore.rules.sugerido).
const String firebaseProjeto = "citoatlas";
const String firebaseBanco = "(default)";
const String firebaseChaveApi = "AIzaSyDIye72T8UWKS04vheKeFlQhZiofFoDY14";

/// ATENÇÃO: este cliente OAuth é do projeto Firebase ANTERIOR (número 657710120359).
/// Para o login Google funcionar com o projeto "citoatlas" (764790014755), troque pelo ID
/// do cliente "Web" dele (Firebase > Authentication > Google, ou Google Cloud > Credenciais).
/// ID do cliente OAuth "Web" do projeto. É o `serverClientId` do login Google:
/// o token resultante é aceito pelo Firebase Auth.
const String googleClienteWeb = "657710120359-e9ojdqbiv7dkcdpkols2maq2dd32et6e.apps.googleusercontent.com";

/// ID do cliente OAuth do iOS (termina em .apps.googleusercontent.com). Só é
/// preciso no iOS: --dart-define=GOOGLE_CLIENTE_IOS=... (veja o README).
const String googleClienteIos = String.fromEnvironment('GOOGLE_CLIENTE_IOS');

/// IDs fora do formato L-0001 são texto livre e podem conter nome ou
/// prontuário de paciente. Com false, o catálogo os troca por um código curto.
const bool catalogoExibeIdLivre = false;

/// Avaliações necessárias para o quórum do CitoAtlas.
const int catalogoVotosParaQuorum = 3;
