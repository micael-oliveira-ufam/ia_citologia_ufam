import 'package:flutter/material.dart';

/// Uma categoria do Sistema Bethesda.
///
/// Cada categoria carrega ícone e texto além da cor: quem não distingue cores
/// precisa conseguir ler o resultado do mesmo jeito.
class CategoriaBethesda {
  const CategoriaBethesda({
    required this.sigla,
    required this.nome,
    required this.explicacao,
    required this.cor,
    required this.icone,
    required this.gravidade,
  });

  final String sigla;
  final String nome;
  final String explicacao;
  final Color cor;
  final IconData icone;

  /// 0 = sem lesão, 3 = carcinoma. Define a ordem da régua na tela.
  final int gravidade;
}

/// Ordem exigida pelo modelo. Corresponde às pastas do treinamento
/// (Carcinoma, HSIL, LSIL, Normal) em ordem alfabética, que é como o
/// ImageFolder as indexou. Mexer aqui inverte diagnósticos silenciosamente.
const List<String> ordemDoModelo = ['SCC', 'HSIL', 'LSIL', 'NILM'];

const Map<String, CategoriaBethesda> categorias = {
  'NILM': CategoriaBethesda(
    sigla: 'NILM',
    nome: 'Negativo para lesão intraepitelial ou malignidade',
    explicacao:
        'Nenhuma alteração celular sugestiva de lesão precursora ou câncer no '
        'campo analisado. Corresponde ao resultado esperado na maioria dos '
        'exames de rastreio.',
    cor: Color(0xFF2E7D62),
    icone: Icons.check_circle_outline,
    gravidade: 0,
  ),
  'LSIL': CategoriaBethesda(
    sigla: 'LSIL',
    nome: 'Lesão intraepitelial escamosa de baixo grau',
    explicacao:
        'Alterações associadas à infecção por HPV, como coilocitose. A maioria '
        'regride espontaneamente, mas exige seguimento.',
    cor: Color(0xFFC08A1E),
    icone: Icons.info_outline,
    gravidade: 1,
  ),
  'HSIL': CategoriaBethesda(
    sigla: 'HSIL',
    nome: 'Lesão intraepitelial escamosa de alto grau',
    explicacao:
        'Lesão precursora com risco relevante de progressão. Indica '
        'necessidade de colposcopia e conduta específica.',
    cor: Color(0xFFD9642E),
    icone: Icons.warning_amber_outlined,
    gravidade: 2,
  ),
  'SCC': CategoriaBethesda(
    sigla: 'SCC',
    nome: 'Carcinoma de células escamosas',
    explicacao:
        'Achados citológicos compatíveis com carcinoma invasor. Encaminhamento '
        'imediato para investigação.',
    cor: Color(0xFFB23A3A),
    icone: Icons.report_outlined,
    gravidade: 3,
  ),
};

/// Categorias da menos para a mais grave, na ordem em que aparecem na interface.
List<CategoriaBethesda> get categoriasPorGravidade {
  final lista = categorias.values.toList()
    ..sort((a, b) => a.gravidade.compareTo(b.gravidade));
  return lista;
}

CategoriaBethesda categoriaDe(String sigla) =>
    categorias[sigla] ?? categorias['NILM']!;
