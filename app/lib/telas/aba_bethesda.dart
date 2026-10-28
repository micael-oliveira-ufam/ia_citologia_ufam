import 'package:flutter/material.dart';

import '../data/bethesda.dart';
import '../data/projeto.dart';
import '../tema.dart';
import '../widgets/comuns.dart';

/// Referência rápida para o profissional: a escala de Bethesda usada pelo
/// modelo e o contexto da citologia em meio líquido.
class AbaBethesda extends StatelessWidget {
  const AbaBethesda({super.key});

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 32),
      children: [
        const ReguaBethesda(),
        const SizedBox(height: 16),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(18),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'As quatro categorias do modelo',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 14),
                for (final c in categoriasPorGravidade) ...[
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Icon(c.icone, color: c.cor, size: 22),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              '${c.sigla} — ${c.nome}',
                              style: TextStyle(
                                fontWeight: FontWeight.w700,
                                fontSize: 15,
                                color: c.cor,
                              ),
                            ),
                            const SizedBox(height: 4),
                            Text(
                              c.explicacao,
                              style: const TextStyle(
                                fontSize: 14.5,
                                height: 1.5,
                                color: Cores.tintaFraca,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                  if (c != categoriasPorGravidade.last)
                    const Padding(
                      padding: EdgeInsets.symmetric(vertical: 14),
                      child: Divider(height: 1),
                    ),
                ],
              ],
            ),
          ),
        ),
        const SizedBox(height: 14),
        const CartaoSecao(
          titulo: 'Citologia em meio líquido',
          icone: Icons.biotech_outlined,
          child: Text(
            textoSobreCitopatologia,
            style: TextStyle(fontSize: 15, height: 1.55, color: Cores.tintaFraca),
          ),
        ),
        const SizedBox(height: 14),
        const CartaoSecao(
          titulo: 'Fora das quatro categorias',
          icone: Icons.help_outline,
          child: Text(
            'O Sistema Bethesda tem outras categorias (por exemplo ASC-US, '
            'ASC-H e AGC) que o modelo não reconhece: ele só distingue NILM, '
            'LSIL, HSIL e SCC. Um campo com achados fora dessas quatro '
            'classes será forçado a uma delas, o que é mais um motivo para '
            'tratar a saída como apoio à leitura, não como laudo.',
            style: TextStyle(fontSize: 15, height: 1.55, color: Cores.tintaFraca),
          ),
        ),
      ],
    );
  }
}
