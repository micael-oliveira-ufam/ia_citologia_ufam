import 'dart:convert';
import 'dart:io';

import 'package:colo_ia/servicos/modelo_risco.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  late ModeloRisco modelo;

  setUpAll(() {
    final json = jsonDecode(
      File('assets/risco/modelo_risco.json').readAsStringSync(),
    ) as Map<String, dynamic>;
    modelo = ModeloRisco.deJson(json);
  });

  test('reproduz as probabilidades do scikit-learn (30 casos de referência)', () {
    final linhas = File('test/casos_referencia.csv').readAsLinesSync();
    final cabecalho = linhas.first.split(',');
    expect(cabecalho.last, 'prob');
    var casos = 0;
    for (final linha in linhas.skip(1).where((l) => l.trim().isNotEmpty)) {
      final celulas = linha.split(',');
      final respostas = <String, double?>{};
      for (var i = 0; i < cabecalho.length - 1; i++) {
        respostas[cabecalho[i]] =
            celulas[i].isEmpty ? null : double.parse(celulas[i]);
      }
      final esperado = double.parse(celulas.last);
      expect(
        modelo.prever(respostas),
        closeTo(esperado, 1e-9),
        reason: 'caso: $linha',
      );
      casos++;
    }
    expect(casos, 30);
  });

  test('cortes e faixas são consistentes', () {
    expect(modelo.corteModerado, lessThan(modelo.corteAlto));
    expect(modelo.faixas.length, 3);
    expect(modelo.faixas[0].taxa, lessThan(modelo.faixas[2].taxa));
    expect(modelo.prevalencia, closeTo(55 / 858, 1e-9));
  });

  test('o nível acompanha os cortes', () {
    // Mediana de tudo (nenhuma resposta) produz um escore fixo e válido.
    final r = modelo.avaliar(const {});
    final esperado = r.escore >= modelo.corteAlto
        ? NivelRisco.alto
        : (r.escore >= modelo.corteModerado
            ? NivelRisco.moderado
            : NivelRisco.baixo);
    expect(r.nivel, esperado);
    expect(r.observado, same(modelo.faixas[r.nivel.index]));
  });
}
