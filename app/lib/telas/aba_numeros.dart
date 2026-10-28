import 'package:flutter/material.dart';

import '../config.dart';
import '../data/estatisticas.dart';
import '../perfil.dart';
import '../tema.dart';
import '../widgets/conteudo.dart';

/// Aba "Números" do paciente.
class AbaNumeros extends StatelessWidget {
  const AbaNumeros({
    super.key,
    required this.estatisticas,
    required this.perfil,
  });

  final Future<List<Estatistica>> estatisticas;
  final Perfil perfil;

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 32),
      children: [
        Text(
          'O que dizem os números',
          style: Theme.of(context).textTheme.headlineSmall,
        ),
        const SizedBox(height: 6),
        const Text(
          'Toque em “Fonte” para conferir cada dado na origem.',
          style: TextStyle(fontSize: 15, color: Cores.tintaFraca, height: 1.5),
        ),
        const SizedBox(height: 16),
        ListaEstatisticas(estatisticas: estatisticas, perfil: perfil),
      ],
    );
  }
}

/// Lista de [Estatistica] vinda do JSON embarcado; usada também na aba
/// "Orientar" do profissional.
class ListaEstatisticas extends StatelessWidget {
  const ListaEstatisticas({
    super.key,
    required this.estatisticas,
    required this.perfil,
  });

  final Future<List<Estatistica>> estatisticas;
  final Perfil perfil;

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<List<Estatistica>>(
      future: estatisticas,
      builder: (context, resposta) {
        if (resposta.hasError) {
          return const Text(
            'Não foi possível carregar os números.',
            style: TextStyle(color: Cores.tintaFraca),
          );
        }
        final itens = resposta.data;
        if (itens == null) {
          return const Padding(
            padding: EdgeInsets.symmetric(vertical: 24),
            child: Center(child: CircularProgressIndicator()),
          );
        }
        return Column(
          children: [
            for (final e in itens) _CartaoEstatistica(item: e, perfil: perfil),
          ],
        );
      },
    );
  }
}

class _CartaoEstatistica extends StatelessWidget {
  const _CartaoEstatistica({required this.item, required this.perfil});

  final Estatistica item;
  final Perfil perfil;

  @override
  Widget build(BuildContext context) {
    final estilo = Theme.of(context).textTheme;
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                item.titulo,
                style: estilo.titleMedium?.copyWith(
                  fontWeight: FontWeight.w800,
                  color: Cores.teal,
                ),
              ),
              const SizedBox(height: 6),
              Text(item.detalhePara(perfil), style: estilo.bodyMedium),
              const SizedBox(height: 8),
              Wrap(
                spacing: 8,
                crossAxisAlignment: WrapCrossAlignment.center,
                children: [
                  TextButton.icon(
                    style: TextButton.styleFrom(
                      padding: EdgeInsets.zero,
                      minimumSize: const Size(48, 48),
                      alignment: Alignment.centerLeft,
                    ),
                    onPressed: () => abrirLink(context, item.fonteUrl),
                    icon: const Icon(Icons.open_in_new, size: 16),
                    label: Text('Fonte: ${item.fonteNome}'),
                  ),
                  if (exibirSeloAConfirmar && !item.verificado)
                    const Chip(
                      avatar: Icon(Icons.rule, size: 16),
                      label: Text('a confirmar'),
                      visualDensity: VisualDensity.compact,
                    ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
