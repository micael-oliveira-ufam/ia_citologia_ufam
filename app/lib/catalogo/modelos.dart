import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../config.dart';

/// Classificações de Bethesda do CitoAtlas (as mesmas oito do sistema web).
enum Classificacao {
  nilm('NILM', 'Negativo para Lesão ou Malignidade', Color(0xFF2E7D62)),
  ascUs(
    'ASC-US',
    'Células Escamosas Atípicas de Significado Indeterminado',
    Color(0xFF8A6A08),
  ),
  ascH(
    'ASC-H',
    'Células Escamosas Atípicas (Não se pode excluir HSIL)',
    Color(0xFFB05A14),
  ),
  lsil(
    'LSIL',
    'Lesão Intraepitelial Escamosa de Baixo Grau',
    Color(0xFF9A6A10),
  ),
  hsil(
    'HSIL',
    'Lesão Intraepitelial Escamosa de Alto Grau',
    Color(0xFFC0541F),
  ),
  scc('SCC', 'Carcinoma de Células Escamosas', Color(0xFFB23A3A)),
  adenocarcinoma('Adenocarcinoma', 'Adenocarcinoma', Color(0xFF7B3FA0)),
  outro('Outro', 'Outro preenchimento / Outras alterações', Color(0xFF55616B));

  const Classificacao(this.sigla, this.descricao, this.cor);

  /// Texto gravado no Firestore (igual ao enum ClassificationType do CitoAtlas).
  final String sigla;
  final String descricao;
  final Color cor;

  static Classificacao? deSigla(String? sigla) {
    for (final c in Classificacao.values) {
      if (c.sigla == sigla) return c;
    }
    return null;
  }
}

/// ID exibido de uma lâmina. O campo é texto livre no CitoAtlas: só IDs no
/// formato L-0001 aparecem como foram digitados; qualquer outra coisa pode ser
/// nome ou prontuário e vira um código curto derivado (o mesmo cálculo do site).
class IdExibido {
  const IdExibido(this.texto, {required this.oculto});

  final String texto;
  final bool oculto;
}

IdExibido idExibido(String? patientId) {
  final bruto = (patientId ?? '').trim();
  if (RegExp(r'^L-\d{1,6}$', caseSensitive: false).hasMatch(bruto)) {
    return IdExibido(bruto.toUpperCase(), oculto: false);
  }
  if (catalogoExibeIdLivre && bruto.isNotEmpty) {
    return IdExibido(
      bruto.length > 40 ? bruto.substring(0, 40) : bruto,
      oculto: false,
    );
  }
  var h = 0x811c9dc5;
  for (final unidade in bruto.codeUnits) {
    h ^= unidade;
    h = (h * 0x01000193) & 0xFFFFFFFF;
  }
  final hex = h.toRadixString(16);
  final curto = hex.length > 4 ? hex.substring(0, 4) : hex;
  return IdExibido('Lâmina ${curto.toUpperCase()}', oculto: true);
}

/// Uma lâmina do catálogo. Só guarda o que o aplicativo usa.
class Lamina {
  Lamina({
    required this.id,
    required this.patientId,
    required this.imageUrl,
    required this.criadaEm,
    this.data,
    this.classificacao,
    this.descricao,
    this.observacoes,
    this.criadorId,
  });

  final String id;
  final String patientId;
  final String imageUrl;
  final int criadaEm;

  /// "AAAA-MM-DD", como o CitoAtlas grava.
  final String? data;

  // Campos que o criador preencheu. Só são usados para o próprio criador editar
  // a lâmina; o catálogo nunca os exibe a outras pessoas (avaliação cega).
  final String? classificacao;
  final String? descricao;
  final String? observacoes;
  final String? criadorId;

  IdExibido get idPublico => idExibido(patientId);

  /// Bytes da imagem quando ela vem embutida (data URL); null se for um link.
  late final Uint8List? bytes = _decodificar(imageUrl);

  /// Link https da imagem quando ela está no Firebase Storage.
  String? get urlHttps => imageUrl.startsWith('https://') ? imageUrl : null;

  bool get temImagemUtilizavel => bytes != null || urlHttps != null;

  static Uint8List? _decodificar(String url) {
    final m = RegExp(r'^data:image/(jpeg|png|webp);base64,([A-Za-z0-9+/=]+)$')
        .firstMatch(url);
    if (m == null) return null;
    try {
      return base64Decode(m.group(2)!);
    } catch (_) {
      return null;
    }
  }

  DateTime get quando {
    final d = data;
    if (d != null && RegExp(r'^\d{4}-\d{2}-\d{2}$').hasMatch(d)) {
      final p = d.split('-').map(int.parse).toList();
      return DateTime(p[0], p[1], p[2]);
    }
    return DateTime.fromMillisecondsSinceEpoch(criadaEm);
  }
}

class Avaliacao {
  const Avaliacao({
    required this.slideId,
    required this.userId,
    required this.userName,
    required this.userEmail,
    required this.classificacao,
    required this.criadaEm,
    this.descricao = '',
    this.observacoes = '',
  });

  final String slideId;
  final String userId;
  final String userName;
  final String userEmail;
  final Classificacao classificacao;
  final String descricao;
  final String observacoes;
  final int criadaEm;

  /// ID do documento, igual ao do CitoAtlas: `lâmina_usuário`.
  String get id => '${slideId}_$userId';
}

String formatarData(DateTime d) {
  String dois(int n) => n.toString().padLeft(2, '0');
  return '${dois(d.day)}/${dois(d.month)}/${d.year}';
}
