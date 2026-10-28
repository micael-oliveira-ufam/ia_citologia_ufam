/// O resultado de um exame de citologia em linguagem de paciente.
///
/// As siglas e a ordem seguem [categorias] (bethesda.dart); aqui só mudam os
/// textos. São orientações gerais: o laudo de cada pessoa é interpretado pela
/// equipe de saúde que pediu o exame.
library;

class ExplicacaoPaciente {
  const ExplicacaoPaciente({
    required this.apelido,
    required this.oQueSignifica,
    required this.oQueFazer,
  });

  /// Nome curto e sem jargão.
  final String apelido;
  final String oQueSignifica;
  final String oQueFazer;
}

const Map<String, ExplicacaoPaciente> explicacoesPaciente = {
  'NILM': ExplicacaoPaciente(
    apelido: 'Sem alterações importantes',
    oQueSignifica: 'Não foram vistas células com alterações sugestivas de '
        'lesão ou de câncer neste exame. É o resultado mais comum.',
    oQueFazer: 'Continue fazendo o exame preventivo nos intervalos indicados '
        'pela equipe de saúde. Um resultado normal hoje não dispensa os '
        'próximos exames.',
  ),
  'LSIL': ExplicacaoPaciente(
    apelido: 'Lesão de baixo grau',
    oQueSignifica: 'Mostra alterações nas células do colo do útero, muitas '
        'vezes ligadas à infecção pelo HPV. A maioria regride sozinha, mas '
        'precisa de acompanhamento. Não é câncer.',
    oQueFazer: 'Leve o resultado à equipe de saúde, que vai indicar o '
        'seguimento: repetir o exame ou avaliar o colo do útero mais de perto.',
  ),
  'HSIL': ExplicacaoPaciente(
    apelido: 'Lesão de alto grau',
    oQueSignifica: 'Alterações mais importantes nas células. É uma lesão '
        'precursora: pode evoluir para câncer se não for tratada, e por isso '
        'precisa de investigação.',
    oQueFazer: 'Procure a equipe de saúde logo. O próximo passo costuma ser a '
        'colposcopia, que pode incluir biópsia, para definir o tratamento. '
        'Com acompanhamento, essas lesões costumam ser tratadas antes de '
        'virarem câncer.',
  ),
  'SCC': ExplicacaoPaciente(
    apelido: 'Células com aspecto de câncer',
    oQueSignifica: 'O exame encontrou células com aspecto de carcinoma. Só a '
        'confirmação por outros exames e a avaliação médica definem o '
        'diagnóstico e o tratamento.',
    oQueFazer: 'Procure o serviço de saúde o quanto antes. Quanto mais cedo '
        'forem a investigação e o tratamento, melhores as chances.',
  ),
};
