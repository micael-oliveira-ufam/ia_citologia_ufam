import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

import '../data/bethesda.dart';
import 'imagem.dart';

const String caminhoModelo = 'assets/model/citologia_convnext.onnx';

// ---------------------------------------------------------------------------
// PARÂMETROS COMPARTILHADOS COM A PLATAFORMA WEB
// Espelham app.py. Ver PARAMETROS_COMPARTILHADOS.md. Mexer em um lado sem
// mexer no outro faz aparelho e web discordarem sobre a mesma imagem.
// ---------------------------------------------------------------------------

/// Abaixo disto a predição é tratada como indefinida.
const double limiarConfianca = 0.60;

/// Fração mínima das oito orientações que precisa concordar.
const double limiarConcordancia = 0.75;

/// Checkpoint de origem, exportado por tools/export_onnx.py.
const String checkpointOrigem =
    'convnext_liquid_based_citology_IARC_digital_atlas_01_08_26.pt';

/// Resultado de uma análise.
class Analise {
  Analise({
    required this.probabilidades,
    required this.indice,
    required this.pngOriginal,
    required this.pngSobreposto,
    required this.milissegundos,
    this.probabilidadesRobustez,
    this.concordancia,
  });

  /// Na ordem de [ordemDoModelo].
  final List<double> probabilidades;
  final int indice;
  final Uint8List pngOriginal;
  final Uint8List pngSobreposto;
  final int milissegundos;

  /// Média sobre as oito simetrias, quando a verificação foi pedida.
  final List<double>? probabilidadesRobustez;

  /// Fração das oito orientações que concordaram com a leitura direta.
  final double? concordancia;

  String get sigla => ordemDoModelo[indice];
  double get confianca => probabilidades[indice];

  bool get instavel =>
      confianca < limiarConfianca ||
      (concordancia != null && concordancia! < limiarConcordancia);

  String? get siglaRobustez {
    final p = probabilidadesRobustez;
    if (p == null) return null;
    var melhor = 0;
    for (var i = 1; i < p.length; i++) {
      if (p[i] > p[melhor]) melhor = i;
    }
    return ordemDoModelo[melhor];
  }

  double probabilidadeDe(String sigla) {
    final i = ordemDoModelo.indexOf(sigla);
    return i < 0 ? 0 : probabilidades[i];
  }
}

/// Carrega o modelo ONNX embarcado e roda a inferência dentro do aparelho.
/// Nenhuma imagem sai do dispositivo, e nada aqui exige conexão.
class Classificador {
  OrtSession? _sessao;
  String _entrada = 'imagem';
  bool _pronto = false;
  String? _erro;

  bool get pronto => _pronto;
  String? get erro => _erro;

  Future<void> iniciar() async {
    if (_pronto) return;
    try {
      final ort = OnnxRuntime();
      _sessao = await ort.createSessionFromAsset(
        caminhoModelo,
        options: OrtSessionOptions(
          intraOpNumThreads: 2,
          providers: [OrtProvider.CPU],
        ),
      );
      _entrada = _sessao!.inputNames.first;
      _pronto = true;
      _erro = null;
    } catch (e) {
      _erro = 'Não foi possível carregar o modelo embarcado.\n$e';
      _pronto = false;
      rethrow;
    }
  }

  Future<void> encerrar() async {
    await _sessao?.close();
    _sessao = null;
    _pronto = false;
  }

  /// Uma passagem direta. Devolve (probabilidades, mapa CAM da classe escolhida).
  Future<(List<double>, Float32List)> _rodar(Float32List tensor) async {
    final sessao = _sessao;
    if (sessao == null) throw StateError('Modelo não carregado.');

    final entrada = await OrtValue.fromList(
      tensor,
      [1, 3, ladoEntrada, ladoEntrada],
    );
    Map<String, OrtValue>? saidas;
    try {
      saidas = await sessao.run({_entrada: entrada});

      final logits = (await saidas['logits']!.asFlattenedList())
          .map((e) => (e as num).toDouble())
          .toList();
      final probabilidades = _softmax(logits);

      var melhor = 0;
      for (var i = 1; i < probabilidades.length; i++) {
        if (probabilidades[i] > probabilidades[melhor]) melhor = i;
      }

      final camPlano = (await saidas['cam']!.asFlattenedList())
          .map((e) => (e as num).toDouble())
          .toList();
      final area = ladoCam * ladoCam;
      final mapa = Float32List(area);
      for (var i = 0; i < area; i++) {
        mapa[i] = camPlano[melhor * area + i];
      }

      return (probabilidades, mapa);
    } finally {
      await entrada.dispose();
      if (saidas != null) {
        for (final t in saidas.values) {
          await t.dispose();
        }
      }
    }
  }

  /// Analisa os bytes de uma imagem.
  ///
  /// [verificarRobustez] roda a mesma imagem nas oito simetrias do quadrado.
  /// Custa oito vezes mais tempo, e revela quando a rede muda de opinião só
  /// porque o campo foi girado.
  Future<Analise> analisar(
    Uint8List bytes, {
    bool verificarRobustez = false,
  }) async {
    final relogio = Stopwatch()..start();

    final preparada = await compute(prepararImagem, bytes);
    final (probabilidades, mapa) = await _rodar(preparada.tensor);

    var melhor = 0;
    for (var i = 1; i < probabilidades.length; i++) {
      if (probabilidades[i] > probabilidades[melhor]) melhor = i;
    }

    List<double>? media;
    double? concordancia;
    if (verificarRobustez) {
      final soma = List<double>.filled(probabilidades.length, 0);
      var acordos = 0;
      for (var s = 0; s < 8; s++) {
        final variante = s == 0
            ? probabilidades
            : (await _rodar(aplicarSimetria(preparada.tensor, s))).$1;
        var topo = 0;
        for (var i = 1; i < variante.length; i++) {
          if (variante[i] > variante[topo]) topo = i;
        }
        if (topo == melhor) acordos++;
        for (var i = 0; i < soma.length; i++) {
          soma[i] += variante[i] / 8;
        }
      }
      media = soma;
      concordancia = acordos / 8;
    }

    final png = await compute(pngDoCampo, preparada.rgb);
    final sobreposto = await compute(
      gerarSobreposicao,
      PedidoSobreposicao(rgb: preparada.rgb, cam: mapa),
    );

    relogio.stop();
    return Analise(
      probabilidades: probabilidades,
      indice: melhor,
      pngOriginal: png,
      pngSobreposto: sobreposto,
      milissegundos: relogio.elapsedMilliseconds,
      probabilidadesRobustez: media,
      concordancia: concordancia,
    );
  }

  static List<double> _softmax(List<double> logits) {
    final maximo = logits.reduce(math.max);
    final exp = logits.map((v) => math.exp(v - maximo)).toList();
    final soma = exp.reduce((a, b) => a + b);
    return exp.map((v) => v / soma).toList();
  }
}
