import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;

/// Faixas de risco relativo, da menor para a maior.
enum NivelRisco { baixo, moderado, alto }

/// Desempenho observado de uma faixa de risco nas previsões fora da amostra
/// (validação cruzada) do conjunto de treino.
class FaixaObservada {
  const FaixaObservada({
    required this.n,
    required this.positivos,
    required this.taxa,
  });

  final int n;
  final int positivos;
  final double taxa;
}

class ResultadoRisco {
  const ResultadoRisco({
    required this.escore,
    required this.nivel,
    required this.observado,
  });

  /// Saída bruta do modelo. Não é uma probabilidade clínica calibrada.
  final double escore;
  final NivelRisco nivel;
  final FaixaObservada observado;
}

class ArvoreDecisao {
  ArvoreDecisao(Map<String, dynamic> j)
      : esquerda = Int32List.fromList((j['left'] as List).cast<int>()),
        direita = Int32List.fromList((j['right'] as List).cast<int>()),
        variavel = Int32List.fromList((j['feature'] as List).cast<int>()),
        limiar = Float64List.fromList(
          (j['threshold'] as List).map((e) => (e as num).toDouble()).toList(),
        ),
        valor = Float64List.fromList(
          (j['value'] as List).map((e) => (e as num).toDouble()).toList(),
        );

  final Int32List esquerda;
  final Int32List direita;
  final Int32List variavel;
  final Float64List limiar;
  final Float64List valor;

  /// Mesma regra do scikit-learn: vai para a esquerda se x <= limiar; folhas
  /// têm filho esquerdo igual a -1.
  double folha(Float64List x) {
    var no = 0;
    while (esquerda[no] != -1) {
      no = x[variavel[no]] <= limiar[no] ? esquerda[no] : direita[no];
    }
    return valor[no];
  }
}

/// Modelo treinado em Python (ml/train.py) e exportado para
/// assets/risco/modelo_risco.json. Suporta os tipos que o script exporta:
/// "forest", "gbm" e "logistic".
class ModeloRisco {
  ModeloRisco._({
    required this.nome,
    required this.tipo,
    required this.variaveis,
    required this.medianas,
    required this.mediasEscala,
    required this.desviosEscala,
    required this.corteModerado,
    required this.corteAlto,
    required this.faixas,
    required this.validacao,
    this.arvores = const [],
    this.coeficientes = const [],
    this.intercepto = 0,
    this.inicial = 0,
    this.taxaAprendizado = 1,
  });

  factory ModeloRisco.deJson(Map<String, dynamic> j) {
    List<double> lista(String chave) =>
        (j[chave] as List).map((e) => (e as num).toDouble()).toList();

    final tiers = j['tiers'] as Map<String, dynamic>;
    final observadas = (tiers['observed'] as List).cast<Map<String, dynamic>>();
    FaixaObservada faixa(int i) => FaixaObservada(
          n: observadas[i]['n'] as int,
          positivos: observadas[i]['positives'] as int,
          taxa: (observadas[i]['rate'] as num).toDouble(),
        );

    final tipo = j['type'] as String;
    return ModeloRisco._(
      nome: j['name'] as String,
      tipo: tipo,
      variaveis: (j['features'] as List).cast<String>(),
      medianas: lista('impute_median'),
      mediasEscala: lista('scale_mean'),
      desviosEscala: lista('scale_std'),
      corteModerado: (tiers['cut_moderate'] as num).toDouble(),
      corteAlto: (tiers['cut_high'] as num).toDouble(),
      faixas: [faixa(0), faixa(1), faixa(2)],
      validacao: (j['validation'] as Map<String, dynamic>?) ?? const {},
      arvores: tipo == 'logistic'
          ? const []
          : (j['trees'] as List)
              .map((t) => ArvoreDecisao(t as Map<String, dynamic>))
              .toList(),
      coeficientes: tipo == 'logistic' ? lista('coef') : const [],
      intercepto: tipo == 'logistic' ? (j['intercept'] as num).toDouble() : 0,
      inicial: tipo == 'gbm' ? (j['init'] as num).toDouble() : 0,
      taxaAprendizado:
          tipo == 'gbm' ? (j['learning_rate'] as num).toDouble() : 1,
    );
  }

  static Future<ModeloRisco> carregar() async {
    final bruto = await rootBundle.loadString('assets/risco/modelo_risco.json');
    return ModeloRisco.deJson(jsonDecode(bruto) as Map<String, dynamic>);
  }

  final String nome;
  final String tipo;
  final List<String> variaveis;
  final List<double> medianas;
  final List<double> mediasEscala;
  final List<double> desviosEscala;
  final double corteModerado;
  final double corteAlto;

  /// Na ordem: baixo, moderado, alto.
  final List<FaixaObservada> faixas;

  /// Metadados de validação (n, positivos, AUC, teste de permutação...).
  final Map<String, dynamic> validacao;

  final List<ArvoreDecisao> arvores;
  final List<double> coeficientes;
  final double intercepto;
  final double inicial;
  final double taxaAprendizado;

  int get numeroDeArvores => arvores.length;

  /// Positivos / total no conjunto de treino (0 se não informado).
  double get prevalencia {
    final p = validacao['positives'];
    final n = validacao['n'];
    if (p is num && n is num && n > 0) return p / n;
    return 0;
  }

  double? get aucForaDaAmostra {
    final v = validacao['oof_auc'];
    return v is num ? v.toDouble() : null;
  }

  Map<String, dynamic> get permutacao =>
      (validacao['permutation'] as Map<String, dynamic>?) ?? const {};

  static final Float32List _f32 = Float32List(1);

  /// O scikit-learn arredonda a entrada das árvores para float32 e compara em
  /// dupla precisão. Repetimos isso para o resultado ser idêntico ao do Python.
  static double _paraF32(double v) {
    _f32[0] = v;
    return _f32[0];
  }

  /// [respostas] mapeia o nome de cada variável (ver [variaveis]) ao valor;
  /// `null` usa a mediana do treino.
  double prever(Map<String, double?> respostas) {
    final x = Float64List(variaveis.length);
    for (var i = 0; i < variaveis.length; i++) {
      final v = respostas[variaveis[i]] ?? medianas[i];
      final z = (v - mediasEscala[i]) / desviosEscala[i];
      x[i] = tipo == 'logistic' ? z : _paraF32(z);
    }
    switch (tipo) {
      case 'logistic':
        var logit = intercepto;
        for (var i = 0; i < x.length; i++) {
          logit += coeficientes[i] * x[i];
        }
        return _sigmoide(logit);
      case 'gbm':
        var bruto = inicial;
        for (final a in arvores) {
          bruto += taxaAprendizado * a.folha(x);
        }
        return _sigmoide(bruto);
      default: // forest
        var total = 0.0;
        for (final a in arvores) {
          total += a.folha(x);
        }
        return total / arvores.length;
    }
  }

  ResultadoRisco avaliar(Map<String, double?> respostas) {
    final e = prever(respostas);
    final nivel = e >= corteAlto
        ? NivelRisco.alto
        : (e >= corteModerado ? NivelRisco.moderado : NivelRisco.baixo);
    return ResultadoRisco(
      escore: e,
      nivel: nivel,
      observado: faixas[nivel.index],
    );
  }

  static double _sigmoide(double s) => 1.0 / (1.0 + math.exp(-s));
}
