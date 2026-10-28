import 'package:flutter/material.dart';

import '../data/bethesda.dart';
import '../data/bethesda_paciente.dart';
import '../tema.dart';
import '../widgets/conteudo.dart';

/// "Entenda seu exame": o resultado de uma citologia em linguagem de paciente.
/// Não analisa imagem nenhuma; a ferramenta de IA é exclusiva do perfil
/// profissional.
class AbaSeuExame extends StatelessWidget {
  const AbaSeuExame({super.key});

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 32),
      children: [
        Text(
          'O que o resultado quer dizer',
          style: Theme.of(context).textTheme.headlineSmall,
        ),
        const SizedBox(height: 6),
        const Text(
          'O laudo do exame preventivo usa uma escala chamada Sistema Bethesda. '
          'Estas são as quatro categorias que aparecem neste aplicativo, da '
          'mais comum à mais grave.',
          style: TextStyle(fontSize: 15, color: Cores.tintaFraca, height: 1.5),
        ),
        const SizedBox(height: 16),
        for (final c in categoriasPorGravidade)
          if (explicacoesPaciente[c.sigla] != null)
            _CartaoResultado(
              categoria: c,
              explicacao: explicacoesPaciente[c.sigla]!,
            ),
        const CartaoInfo(
          icone: Icons.help_outline,
          titulo: 'Seu laudo traz outro termo?',
          texto: 'Existem outras categorias além destas quatro (por exemplo '
              'ASC-US, ASC-H ou AGC). Peça à equipe de saúde que explique o '
              'seu resultado e o que fazer em seguida.',
        ),
        const CartaoInfo(
          icone: Icons.memory_outlined,
          titulo: 'E a inteligência artificial?',
          texto: 'Laboratórios podem usar IA como apoio na leitura das '
              'lâminas, sempre com a conferência de um profissional. Neste '
              'aplicativo, a ferramenta de IA é uma demonstração acadêmica para '
              'profissionais: ela não analisa o seu exame nem emite laudo.',
        ),
        const SizedBox(height: 6),
        const AvisoEducativo(
          'Estas explicações são gerais. Só a equipe de saúde que pediu o seu '
          'exame pode interpretar o seu resultado.',
        ),
      ],
    );
  }
}

class _CartaoResultado extends StatelessWidget {
  const _CartaoResultado({required this.categoria, required this.explicacao});

  final CategoriaBethesda categoria;
  final ExplicacaoPaciente explicacao;

  @override
  Widget build(BuildContext context) {
    final estilo = Theme.of(context).textTheme;
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Semantics(
        container: true,
        label: '${categoria.sigla}, ${explicacao.apelido}. '
            'O que significa: ${explicacao.oQueSignifica} '
            'O que fazer: ${explicacao.oQueFazer}',
        excludeSemantics: true,
        child: Card(
          child: Container(
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(16),
              border: Border(left: BorderSide(color: categoria.cor, width: 5)),
            ),
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(categoria.icone, color: categoria.cor, size: 26),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        '${categoria.sigla} · ${explicacao.apelido}',
                        style: estilo.titleMedium?.copyWith(
                          color: categoria.cor,
                          fontWeight: FontWeight.w800,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 10),
                Text(
                  'O que significa',
                  style: estilo.labelLarge?.copyWith(
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 2),
                Text(explicacao.oQueSignifica, style: estilo.bodyMedium),
                const SizedBox(height: 10),
                Text(
                  'O que fazer',
                  style: estilo.labelLarge?.copyWith(
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 2),
                Text(explicacao.oQueFazer, style: estilo.bodyMedium),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
