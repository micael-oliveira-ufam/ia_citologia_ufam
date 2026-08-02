import 'package:flutter/material.dart';

import '../data/bethesda.dart';
import '../tema.dart';

/// Faixa fixa de advertência. Aparece em todas as telas de propósito.
class AvisoEtico extends StatelessWidget {
  const AvisoEtico({super.key, this.compacto = false});

  final bool compacto;

  static const String texto =
      'Ferramenta acadêmica em desenvolvimento. Treinada apenas com o conjunto '
      'público IARC Digital Atlas, sem aprovação de Comitê de Ética. Não emite '
      'laudo, não substitui o citopatologista e não deve receber imagens de '
      'pacientes.';

  @override
  Widget build(BuildContext context) {
    return Semantics(
      liveRegion: false,
      label: 'Advertência de uso. $texto',
      excludeSemantics: true,
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.fromLTRB(14, 12, 14, 12),
        decoration: BoxDecoration(
          color: Cores.alertaFundo,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: const Color(0xFFF0D6A8)),
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Icon(Icons.info_outline, size: 20, color: Cores.alerta),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                compacto
                    ? 'Uso acadêmico. Sem aprovação de CEP. Não emite laudo.'
                    : texto,
                style: const TextStyle(
                  color: Color(0xFF5A4415),
                  fontSize: 14,
                  height: 1.45,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Escala de gravidade do Sistema Bethesda, com o degrau da predição aceso.
///
/// A posição na escala carrega informação real, já que as categorias são ordinais,
/// então o destaque nunca depende só da cor: o degrau ativo ganha borda
/// grossa, ícone preenchido e rótulo em negrito.
class ReguaBethesda extends StatelessWidget {
  const ReguaBethesda({super.key, this.ativa});

  final String? ativa;

  @override
  Widget build(BuildContext context) {
    final itens = categoriasPorGravidade;
    return Semantics(
      label: ativa == null
          ? 'Escala do Sistema Bethesda, da menor para a maior gravidade: '
              '${itens.map((c) => c.sigla).join(', ')}.'
          : 'Escala do Sistema Bethesda. Posição indicada: $ativa, '
              'nível ${categoriaDe(ativa!).gravidade + 1} de ${itens.length}.',
      excludeSemantics: true,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              for (final c in itens) ...[
                Expanded(child: _Degrau(categoria: c, ativo: c.sigla == ativa)),
                if (c != itens.last) const SizedBox(width: 6),
              ],
            ],
          ),
          const SizedBox(height: 8),
          const Text(
            'Gravidade crescente da esquerda para a direita.',
            style: TextStyle(fontSize: 13, color: Cores.tintaFraca),
          ),
        ],
      ),
    );
  }
}

class _Degrau extends StatelessWidget {
  const _Degrau({required this.categoria, required this.ativo});

  final CategoriaBethesda categoria;
  final bool ativo;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 6),
      decoration: BoxDecoration(
        color: ativo ? categoria.cor.withValues(alpha: 0.10) : Cores.superficie,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(
          color: ativo ? categoria.cor : Cores.linha,
          width: ativo ? 2 : 1,
        ),
      ),
      child: Column(
        children: [
          Icon(
            categoria.icone,
            size: 18,
            color: ativo ? categoria.cor : Cores.tintaFraca,
          ),
          const SizedBox(height: 4),
          Text(
            categoria.sigla,
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 14,
              fontWeight: ativo ? FontWeight.w800 : FontWeight.w500,
              color: ativo ? categoria.cor : Cores.tintaFraca,
            ),
          ),
        ],
      ),
    );
  }
}

/// Barras horizontais de probabilidade por classe.
class BarrasDeProbabilidade extends StatelessWidget {
  const BarrasDeProbabilidade({
    super.key,
    required this.probabilidades,
    required this.destaque,
  });

  /// sigla -> probabilidade em [0, 1].
  final Map<String, double> probabilidades;
  final String destaque;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        for (final c in categoriasPorGravidade) ...[
          _Barra(
            categoria: c,
            valor: probabilidades[c.sigla] ?? 0,
            destacada: c.sigla == destaque,
          ),
          if (c != categoriasPorGravidade.last) const SizedBox(height: 14),
        ],
      ],
    );
  }
}

class _Barra extends StatelessWidget {
  const _Barra({
    required this.categoria,
    required this.valor,
    required this.destacada,
  });

  final CategoriaBethesda categoria;
  final double valor;
  final bool destacada;

  @override
  Widget build(BuildContext context) {
    final percentual = '${(valor * 100).toStringAsFixed(1)}%';
    return Semantics(
      label: '${categoria.sigla}: $percentual'
          '${destacada ? ', classe escolhida pelo modelo' : ''}',
      excludeSemantics: true,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  categoria.sigla,
                  style: TextStyle(
                    fontSize: 15,
                    fontWeight: destacada ? FontWeight.w700 : FontWeight.w500,
                    color: destacada ? categoria.cor : Cores.tinta,
                  ),
                ),
              ),
              Text(
                percentual,
                style: TextStyle(
                  fontSize: 15,
                  fontWeight: destacada ? FontWeight.w700 : FontWeight.w500,
                  color: destacada ? categoria.cor : Cores.tintaFraca,
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
          ClipRRect(
            borderRadius: BorderRadius.circular(6),
            child: LinearProgressIndicator(
              value: valor.clamp(0.0, 1.0),
              minHeight: 10,
              backgroundColor: Cores.linha,
              valueColor: AlwaysStoppedAnimation(
                destacada ? categoria.cor : Cores.tintaFraca.withValues(alpha: 0.35),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// Cartão com título e conteúdo, usado nas telas de informação.
class CartaoSecao extends StatelessWidget {
  const CartaoSecao({
    super.key,
    required this.titulo,
    required this.child,
    this.icone,
  });

  final String titulo;
  final Widget child;
  final IconData? icone;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                if (icone != null) ...[
                  Icon(icone, size: 20, color: Cores.teal),
                  const SizedBox(width: 8),
                ],
                Expanded(
                  child: Text(
                    titulo,
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            child,
          ],
        ),
      ),
    );
  }
}
