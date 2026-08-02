/// Conteúdo da tela "Sobre o projeto".
///
/// Tudo o que é texto institucional mora aqui, para ser editado sem mexer na
/// interface. Acrescente pessoas à lista `equipe` conforme o projeto crescer.
library;

class Pessoa {
  const Pessoa({
    required this.nome,
    required this.papel,
    required this.vinculo,
  });

  final String nome;
  final String papel;
  final String vinculo;
}

class Instituicao {
  const Instituicao({
    required this.nome,
    required this.sigla,
    required this.papel,
  });

  final String nome;
  final String sigla;
  final String papel;
}

const List<Pessoa> equipe = [
  Pessoa(
    nome: 'Micael Davi Lima de Oliveira',
    papel: 'Desenvolvimento do modelo e das aplicações',
    vinculo: 'Iniciação Científica, Faculdade de Ciências Farmacêuticas, UFAM',
  ),
  Pessoa(
    nome: 'Prof. Dr. Toni Ricardo Martins',
    papel: 'Coordenação e orientação',
    vinculo: 'Faculdade de Ciências Farmacêuticas, UFAM',
  ),
  Pessoa(
    nome: 'Profa. Dra. Fabíola Nakamura',
    papel: 'Pesquisadora parceira em computação',
    vinculo: 'Instituto de Computação (IComp), UFAM',
  ),
  Pessoa(
    nome: 'Prof. Dr. Felipe Gomes de Oliveira',
    papel: 'Pesquisador parceiro em computação',
    vinculo: 'Instituto de Computação (IComp), UFAM',
  ),
  Pessoa(
    nome: 'Dra. Ivanete',
    papel: 'Especialista clínica e citopatologista',
    vinculo: 'SEMSA Manaus',
  ),
  Pessoa(
    nome: 'Dra. Carol',
    papel: 'Especialista clínica e citopatologista',
    vinculo: 'SEMSA Manaus',
  ),
];

const List<Instituicao> instituicoes = [
  Instituicao(
    nome: 'Faculdade de Ciências Farmacêuticas, Universidade Federal do Amazonas',
    sigla: 'UFAM / FCF',
    papel: 'Instituição executora. Concepção, treinamento e validação do modelo.',
  ),
  Instituicao(
    nome: 'Instituto de Computação, Universidade Federal do Amazonas',
    sigla: 'UFAM / IComp',
    papel:
        'Parceria em visão computacional e aprendizado profundo, com os professores '
        'pesquisadores Dra. Fabíola Nakamura e Dr. Felipe Gomes de Oliveira.',
  ),
  Instituicao(
    nome: 'Secretaria Municipal de Saúde de Manaus, Laboratório Sebastião Marinho',
    sigla: 'SEMSA Manaus',
    papel:
        'Parceria institucional de validação clínica e citopatológica, promovendo '
        'suporte prático e validação independente do catálogo digital de lâminas. '
        'Especialistas clínicas e citopatologistas envolvidas: Dra. Ivanete e '
        'Dra. Carol.',
  ),
];

/// Conjuntos de dados públicos usados no treinamento e na validação.
class Dataset {
  const Dataset({
    required this.nome,
    required this.sigla,
    required this.url,
    this.referencia,
  });

  final String nome;
  final String sigla;
  final String url;
  final String? referencia;
}

const List<Dataset> datasets = [
  Dataset(
    nome: 'IARC Digital Atlas of Cervical Cytology',
    sigla: 'IARC',
    url: 'https://screening.iarc.fr/atlascyto.php',
  ),
  Dataset(
    nome: 'Liquid based cytology Pap smear images for multi-class diagnosis '
        'of cervical cancer',
    sigla: 'Mendeley Data',
    url: 'https://data.mendeley.com/datasets/zddtpgzv63/2',
    referencia: 'HUSSAIN, Elima; MAHANTA, Lipi B.; BORAH, Himakshi; '
        'DAS, Chandana Ray. Liquid based-cytology Pap smear dataset for '
        'automated multi-class diagnosis of pre-cancerous and cervical cancer '
        'lesions. Data in Brief, v. 30, p. 105589, jun. 2020. '
        'DOI: https://doi.org/10.1016/j.dib.2020.105589',
  ),
];

/// Transparência sobre o processo de desenvolvimento.
const String creditoFerramentas =
    'Código desenvolvido com apoio do modelo de IA generativa Claude Opus 5 '
    '(Anthropic). Concepção científica, treinamento, validação e revisão do '
    'código são de responsabilidade da equipe do projeto.';

/// Números da partição de validação. Alterar aqui reflete na tela "Desempenho".
class Metricas {
  static const String protocolo = 'Validação cruzada estratificada, 5 folds';
  static const int imagens = 1096;
  static const int acertos = 1076;
  static const double acuracia = 0.9818;
  static const double f1Macro = 0.9717;
  static const double precisaoMacro = 0.9798;
  static const double recallMacro = 0.9647;
  static const double kappa = 0.9694;

  /// sigla -> [precisão, recall, f1, nº de imagens]
  static const Map<String, List<double>> porClasse = {
    'NILM': [0.9906, 0.9953, 0.9930, 637],
    'LSIL': [0.9929, 0.9456, 0.9686, 147],
    'HSIL': [0.9469, 0.9907, 0.9683, 216],
    'SCC': [0.9889, 0.9271, 0.9570, 96],
  };
}

const String textoSobreCitopatologia = '''
A citologia em meio líquido é a evolução do exame de Papanicolaou. Em vez de \
esfregar o material coletado direto na lâmina, ele é suspenso num frasco com \
líquido conservante, e a lâmina é preparada no laboratório. O resultado é um \
campo mais limpo, com menos sangue e muco cobrindo as células.

A leitura segue o Sistema Bethesda, que separa os achados em categorias por \
gravidade. É esse mesmo vocabulário que o aplicativo usa.

O gargalo do rastreio não é a coleta, é a leitura: cada lâmina exige minutos de \
microscopia de um profissional escasso, e a fadiga visual ao longo do dia \
degrada a sensibilidade. É aí que uma triagem automatizada pode ajudar. Não \
substituindo o citopatologista, mas ordenando a fila para que os casos \
suspeitos cheguem primeiro à mesa dele.
''';

const String textoSobreIA = '''
O modelo é uma rede neural convolucional ConvNeXt, treinada por transferência \
de aprendizado: partiu de pesos aprendidos em milhões de fotografias comuns e \
foi reajustada em campos de citologia.

Durante o treinamento cada imagem foi girada, espelhada, rotacionada em \
múltiplos de 90° e teve cor e brilho variados. Isso ensina a rede a não depender \
da orientação da lâmina nem do lote de coloração.

Toda a inferência acontece dentro do aparelho. A imagem escolhida não sai do \
celular, não há envio para servidor e o aplicativo funciona sem internet.
''';

const String textoSobreMapaDeCalor = '''
Redes neurais costumam ser tratadas como caixas-pretas. O mapa de calor abre \
essa caixa: ele mostra quais regiões do campo mais pesaram na classe escolhida.

Regiões claras indicam maior influência. Se o destaque cair sobre fundo, muco \
ou artefato de preparo em vez de células, a predição merece desconfiança. O \
mapa é, antes de tudo, uma ferramenta de auditoria.
''';

const String textoLimitacoes = '''
O modelo foi treinado exclusivamente com dois acervos públicos e curados de \
imagens. Nenhuma lâmina de paciente da rede municipal foi utilizada.

Disso decorrem limites concretos:

• O projeto ainda não foi aprovado por Comitê de Ética em Pesquisa. A submissão \
ao CEP é pré-requisito da próxima etapa.
• A validação foi interna. Desempenho em lâminas locais, com outro scanner e \
outra coloração, não foi medido.
• A prevalência do acervo não é a de um programa de rastreio: 42% dos campos \
são alterados, proporção bem maior que a da rotina. O valor preditivo positivo \
em campo será menor que a precisão medida.
• Métricas por imagem não equivalem a métricas por lâmina ou por paciente. Uma \
lâmina tem centenas de campos, e a decisão clínica é sobre a lâmina.
• Não há registro na ANVISA como software como dispositivo médico, exigido para \
uso assistencial.

Enquanto isso, o aplicativo é material de demonstração acadêmica: não emite \
laudo, não deve receber imagens de pacientes e não substitui o citopatologista.
''';

const String versaoModelo = 'ConvNeXt Large · IARC Digital Atlas · checkpoint 01/08/2026';
