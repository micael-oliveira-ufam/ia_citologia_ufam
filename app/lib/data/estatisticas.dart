import 'dart:convert';

import 'package:flutter/services.dart' show rootBundle;

import '../perfil.dart';

/// Um número ou fato de saúde pública, sempre com a fonte para conferência.
class Estatistica {
  const Estatistica({
    required this.titulo,
    required this.detalhePaciente,
    required this.detalheProfissional,
    required this.fonteNome,
    required this.fonteUrl,
    required this.verificado,
  });

  factory Estatistica.deJson(Map<String, dynamic> j) {
    final paciente = j['detalhe_paciente'] as String;
    return Estatistica(
      titulo: j['titulo'] as String,
      detalhePaciente: paciente,
      detalheProfissional: (j['detalhe_profissional'] as String?) ?? paciente,
      fonteNome: j['fonte_nome'] as String,
      fonteUrl: j['fonte_url'] as String,
      verificado: (j['verificado'] as bool?) ?? false,
    );
  }

  final String titulo;
  final String detalhePaciente;
  final String detalheProfissional;
  final String fonteNome;
  final String fonteUrl;

  /// false = número ainda não conferido na fonte (o app mostra "a confirmar").
  final bool verificado;

  String detalhePara(Perfil perfil) =>
      perfil.ehProfissional ? detalheProfissional : detalhePaciente;
}

Future<List<Estatistica>> carregarEstatisticas() async {
  final bruto = await rootBundle.loadString('assets/dados/estatisticas.json');
  final j = jsonDecode(bruto) as Map<String, dynamic>;
  return (j['itens'] as List)
      .map((e) => Estatistica.deJson(e as Map<String, dynamic>))
      .toList();
}
