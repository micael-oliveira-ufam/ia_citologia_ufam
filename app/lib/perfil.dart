import 'package:flutter/material.dart';

/// Quem está usando o aplicativo. A escolha é feita ao abrir e define quais
/// abas aparecem e com que nível de detalhe cada conteúdo é apresentado.
///
/// É uma escolha de conteúdo, não um controle de acesso: o aplicativo não tem
/// login e nada do que é informado sai do aparelho.
enum Perfil {
  paciente(
    rotulo: 'Paciente',
    modo: 'Modo paciente',
    icone: Icons.favorite_outline,
  ),
  profissional(
    rotulo: 'Profissional de saúde',
    modo: 'Modo profissional de saúde',
    icone: Icons.biotech_outlined,
  );

  const Perfil({
    required this.rotulo,
    required this.modo,
    required this.icone,
  });

  final String rotulo;
  final String modo;
  final IconData icone;

  bool get ehProfissional => this == Perfil.profissional;
}
