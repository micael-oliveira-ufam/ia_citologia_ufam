import 'package:flutter/material.dart';

import '../catalogo/modelos.dart';

/// Selo de classificação sobre a foto, na cor da gravidade da classe.
///
/// Mesmo destaque do site: o texto é branco ou escuro, o que tiver maior contraste.
class ChipsClassificacao extends StatelessWidget {
  const ChipsClassificacao({super.key, required this.itens});

  /// Pares (rótulo, classificação), por exemplo ('Você', HSIL).
  final List<(String, Classificacao)> itens;

  static const _escuro = Color(0xFF14232F);

  /// Texto branco ou escuro, o que tiver maior contraste sobre [fundo] (WCAG).
  static Color corSobre(Color fundo) {
    final luz = fundo.computeLuminance();
    final contrasteBranco = 1.05 / (luz + 0.05);
    final contrasteEscuro = (luz + 0.05) / (_escuro.computeLuminance() + 0.05);
    return contrasteBranco >= contrasteEscuro ? Colors.white : _escuro;
  }

  @override
  Widget build(BuildContext context) {
    if (itens.isEmpty) return const SizedBox.shrink();
    return Wrap(
      spacing: 6,
      runSpacing: 6,
      children: [
        for (final (rotulo, classe) in itens)
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: classe.cor,
              borderRadius: BorderRadius.circular(999),
              boxShadow: const [
                BoxShadow(color: Color(0x66000000), blurRadius: 3, offset: Offset(0, 1)),
              ],
            ),
            child: Text(
              '$rotulo · ${classe.sigla}',
              style: TextStyle(
                color: corSobre(classe.cor),
                fontWeight: FontWeight.w800,
                fontSize: 12,
              ),
            ),
          ),
      ],
    );
  }
}

/// Selos da lâmina para o usuário: a própria avaliação e, depois dela, a própria predição.
///
/// No app só entram dados do próprio usuário (o protocolo cego não expõe as
/// classificações de outras pessoas), então a predição só aparece após a avaliação.
List<(String, Classificacao)> chipsDaLamina(Avaliacao? minha, PredicaoSalva? predicao) {
  if (minha == null) return const [];
  final ia = predicao == null ? null : Classificacao.deSigla(predicao.sigla);
  return [
    ('Você', minha.classificacao),
    if (ia != null) ('IA', ia),
  ];
}
