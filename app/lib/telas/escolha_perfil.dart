import 'package:flutter/material.dart';

import '../perfil.dart';
import '../tema.dart';

/// Primeira tela do aplicativo: quem vai usar?
///
/// Aparece a cada abertura de propósito. O mesmo aparelho pode ser usado por
/// uma profissional e por pacientes numa UBS, e o aplicativo não guarda nada.
class TelaEscolhaPerfil extends StatelessWidget {
  const TelaEscolhaPerfil({super.key, required this.aoEscolher});

  final ValueChanged<Perfil> aoEscolher;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 560),
            child: ListView(
              shrinkWrap: true,
              padding: const EdgeInsets.fromLTRB(20, 24, 20, 24),
              children: [
                const Icon(Icons.favorite, size: 48, color: Cores.teal),
                const SizedBox(height: 16),
                Text(
                  'Colo do útero\ncom inteligência artificial',
                  textAlign: TextAlign.center,
                  style: Theme.of(context).textTheme.headlineSmall,
                ),
                const SizedBox(height: 6),
                const Text(
                  'UFAM · SEMSA',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 14,
                    letterSpacing: 1.6,
                    color: Cores.tintaFraca,
                  ),
                ),
                const SizedBox(height: 28),
                Semantics(
                  header: true,
                  child: Text(
                    'Como você vai usar o aplicativo?',
                    textAlign: TextAlign.center,
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                ),
                const SizedBox(height: 16),
                _CartaoPerfil(
                  perfil: Perfil.paciente,
                  titulo: 'Sou paciente',
                  descricao: 'Quero entender o câncer do colo do útero, saber '
                      'como prevenir, conhecer meu perfil de risco e '
                      'entender o resultado do meu exame.',
                  aoToque: () => aoEscolher(Perfil.paciente),
                ),
                const SizedBox(height: 12),
                _CartaoPerfil(
                  perfil: Perfil.profissional,
                  titulo: 'Sou profissional de saúde',
                  descricao: 'Quero analisar campos de citologia com apoio de '
                      'IA, consultar a escala de Bethesda e ter material para '
                      'orientar pacientes.',
                  aoToque: () => aoEscolher(Perfil.profissional),
                ),
                const SizedBox(height: 24),
                const Text(
                  'Você pode trocar de perfil a qualquer momento, pelo botão '
                  'no topo da tela. Nada do que você informar sai do '
                  'aparelho.',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 13.5,
                    color: Cores.tintaFraca,
                    height: 1.45,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _CartaoPerfil extends StatelessWidget {
  const _CartaoPerfil({
    required this.perfil,
    required this.titulo,
    required this.descricao,
    required this.aoToque,
  });

  final Perfil perfil;
  final String titulo;
  final String descricao;
  final VoidCallback aoToque;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: true,
      label: '$titulo. $descricao',
      excludeSemantics: true,
      child: Card(
        clipBehavior: Clip.antiAlias,
        child: InkWell(
          onTap: aoToque,
          child: Padding(
            padding: const EdgeInsets.all(18),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                CircleAvatar(
                  radius: 24,
                  backgroundColor: Cores.tealClaro,
                  child: Icon(perfil.icone, color: Cores.teal, size: 26),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        titulo,
                        style: Theme.of(context).textTheme.titleMedium,
                      ),
                      const SizedBox(height: 4),
                      Text(
                        descricao,
                        style: Theme.of(context).textTheme.bodyMedium,
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 8),
                const Padding(
                  padding: EdgeInsets.only(top: 12),
                  child: Icon(Icons.chevron_right, color: Cores.tintaFraca),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
