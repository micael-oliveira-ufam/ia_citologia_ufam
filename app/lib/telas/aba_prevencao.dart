import 'package:flutter/material.dart';

import '../data/conteudo_prevencao.dart';
import '../data/estatisticas.dart';
import '../perfil.dart';
import '../tema.dart';
import '../widgets/conteudo.dart';
import 'aba_numeros.dart';

/// Prevenção em dois registros:
///
/// - paciente ("Prevenção"): linguagem simples;
/// - profissional ("Orientar"): o mesmo conteúdo com o detalhe técnico, mais
///   pontos para a conversa com a paciente e os números com fonte.
class AbaPrevencao extends StatelessWidget {
  const AbaPrevencao({super.key, required this.perfil, this.estatisticas});

  final Perfil perfil;

  /// Só usado no perfil profissional.
  final Future<List<Estatistica>>? estatisticas;

  @override
  Widget build(BuildContext context) {
    final estilo = Theme.of(context).textTheme;
    final pro = perfil.ehProfissional;
    final numeros = estatisticas;
    return ListView(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 32),
      children: [
        Text(
          pro ? 'Material para orientar' : 'Como prevenir',
          style: estilo.headlineSmall,
        ),
        const SizedBox(height: 6),
        Text(
          pro
              ? 'Três frentes se complementam: vacinar, rastrear e orientar '
                  'hábitos. O texto abaixo traz o detalhe técnico de cada uma.'
              : 'Três frentes se complementam: vacinar, rastrear e cuidar da '
                  'saúde.',
          style: const TextStyle(
            fontSize: 15,
            color: Cores.tintaFraca,
            height: 1.5,
          ),
        ),
        if (pro) ...[
          const TituloSecao('Pontos para a conversa'),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  for (final ponto in pontosParaConversa)
                    Padding(
                      padding: const EdgeInsets.only(bottom: 8),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Padding(
                            padding: EdgeInsets.only(top: 2, right: 10),
                            child: Icon(
                              Icons.check_circle_outline,
                              size: 18,
                              color: Cores.teal,
                            ),
                          ),
                          Expanded(
                            child: Text(ponto, style: estilo.bodyMedium),
                          ),
                        ],
                      ),
                    ),
                ],
              ),
            ),
          ),
        ],
        for (final grupo in prevencao) ...[
          TituloSecao(grupo.titulo),
          for (final t in grupo.topicos)
            CartaoInfo(
              icone: t.icone,
              titulo: t.titulo,
              texto: t.textoPara(perfil),
            ),
        ],
        if (pro && numeros != null) ...[
          const TituloSecao('Números e fontes'),
          ListaEstatisticas(estatisticas: numeros, perfil: perfil),
        ],
        const SizedBox(height: 12),
        AvisoEducativo(
          pro
              ? 'Conteúdo educativo, baseado em recomendações gerais e em '
                  'matérias jornalísticas sobre normas do Ministério da Saúde. '
                  'Confirme sempre as diretrizes vigentes no seu serviço.'
              : 'Conteúdo educativo, baseado em recomendações gerais. Siga '
                  'sempre a orientação do seu profissional de saúde.',
        ),
      ],
    );
  }
}
