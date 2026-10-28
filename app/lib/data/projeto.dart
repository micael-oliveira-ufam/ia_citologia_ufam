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
    papel: 'Desenvolvimento do modelo e da aplicação',
    vinculo: 'Iniciação Científica — Faculdade de Ciências Farmacêuticas, UFAM',
  ),
  Pessoa(
    nome: 'Prof. Dr. Toni Ricardo Martins',
    papel: 'Coordenação e orientação',
    vinculo: 'Faculdade de Ciências Farmacêuticas, UFAM',
  ),
];

const List<Instituicao> instituicoes = [
  Instituicao(
    nome: 'Universidade Federal do Amazonas — Faculdade de Ciências Farmacêuticas',
    sigla: 'UFAM / FCF',
    papel: 'Instituição executora. Concepção, treinamento e validação do modelo.',
  ),
  Instituicao(
    nome: 'Laboratório Sebastião Marinho — Secretaria Municipal de Saúde de Manaus',
    sigla: 'SEMSA',
    papel:
        'Parceria assistencial. Referência em citopatologia da rede municipal e '
        'sede prevista da validação externa.',
  ),
];

/// Números da partição de validação. Alterar aqui reflete na tela "Desempenho".
class Metricas {
  static const int imagens = 220;
  static const int acertos = 219;
  static const double acuracia = 0.9955;
  static const double f1Macro = 0.9911;
  static const double precisaoMacro = 0.9950;
  static const double recallMacro = 0.9875;
  static const double kappa = 0.9929;

  /// sigla -> [precisão, recall, f1, nº de imagens]
  static const Map<String, List<double>> porClasse = {
    'NILM': [1.0000, 0.9500, 0.9744, 20],
    'LSIL': [1.0000, 1.0000, 1.0000, 114],
    'HSIL': [1.0000, 1.0000, 1.0000, 37],
    'SCC': [0.9800, 1.0000, 0.9899, 49],
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
degrada a sensibilidade. É aí que uma triagem automatizada pode ajudar — não \
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

Toda a inferência acontece dentro do aparelho: a imagem escolhida não sai do \
celular e não há envio para servidor. A internet é usada só uma vez, para \
baixar o arquivo do modelo (conferido por SHA-256); depois disso a análise \
funciona sem conexão.
''';

const String textoSobreMapaDeCalor = '''
Redes neurais costumam ser tratadas como caixas-pretas. O mapa de calor abre \
essa caixa: ele mostra quais regiões do campo mais pesaram na classe escolhida.

Regiões claras indicam maior influência. Se o destaque cair sobre fundo, muco \
ou artefato de preparo em vez de células, a predição merece desconfiança — o \
mapa é, antes de tudo, uma ferramenta de auditoria.
''';

const String textoLimitacoes = '''
O modelo foi treinado exclusivamente com o IARC Digital Atlas, um conjunto \
público e curado de imagens. Nenhuma lâmina de paciente da rede municipal foi \
utilizada.

Disso decorrem limites concretos:

• O projeto ainda não foi aprovado por Comitê de Ética em Pesquisa. A submissão \
ao CEP é pré-requisito da próxima etapa.
• A validação foi interna. Desempenho em lâminas locais, com outro scanner e \
outra coloração, não foi medido.
• A prevalência do atlas é invertida em relação à realidade: lá, 91% dos campos \
são alterados; num programa de rastreio, mais de 90% são normais. O valor \
preditivo positivo em campo será menor que a precisão medida.
• São 220 imagens de validação, das quais apenas 20 normais. A margem de erro \
dessas estimativas é larga.
• Não há registro na ANVISA como software como dispositivo médico, exigido para \
uso assistencial.

Enquanto isso, o aplicativo é material de demonstração acadêmica: não emite \
laudo, não deve receber imagens de pacientes e não substitui o citopatologista.
''';

const String versaoModelo = 'ConvNeXt · IARC Digital Atlas · maio de 2026';
