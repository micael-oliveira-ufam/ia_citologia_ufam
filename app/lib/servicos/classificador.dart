import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

import '../data/bethesda.dart';
import 'gerenciador_modelo.dart';
import 'imagem.dart';

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
      confianca < 0.60 || (concordancia != null && concordancia! < 0.75);

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

/// O modelo não pôde ser carregado (ainda não foi baixado, arquivo corrompido
/// ou runtime incompatível). A mensagem é pensada para quem está usando o
/// aplicativo.
class ModeloIndisponivel implements Exception {
  ModeloIndisponivel(this.causa);

  final Object causa;

  @override
  String toString() {
    final c = causa;
    if (c is ErroModelo) return c.mensagem;
    return 'O modelo de citologia não pôde ser aberto neste aparelho. Remova o '
        'modelo na aba Analisar e baixe-o de novo; se continuar, reinstale o '
        'aplicativo.\n\nDetalhe técnico: $c';
  }
}

/// Abre o modelo ONNX que o [GerenciadorModelo] baixou e roda a inferência
/// dentro do aparelho. Nenhuma imagem sai do dispositivo: a rede só é usada
/// para baixar o modelo, e isso é feito pelo gerenciador, não por aqui.
class Classificador {
  Classificador(this.gerenciador);

  final GerenciadorModelo gerenciador;

  OrtSession? _sessao;
  String _entrada = 'imagem';
  bool _pronto = false;
  String? _erro;
  String? _shaCarregado;
  Future<void>? _carga;

  bool get pronto => _pronto;
  String? get erro => _erro;

  /// Carrega o modelo uma única vez, mesmo que várias telas peçam ao mesmo
  /// tempo, e só quando alguém precisa dele: no modo paciente o modelo nunca é
  /// carregado. Se o modelo instalado mudou (atualização), reabre a sessão.
  /// Se falhar, uma chamada posterior tenta de novo.
  Future<void> garantirPronto() async {
    if (_pronto && _shaCarregado != gerenciador.shaInstalado) {
      await encerrar();
    }
    await (_carga ??= _carregar());
  }

  Future<void> _carregar() async {
    try {
      await iniciar();
    } catch (e) {
      _carga = null;
      throw ModeloIndisponivel(e);
    }
  }

  Future<void> iniciar() async {
    if (_pronto) return;
    try {
      final sha = gerenciador.shaInstalado;
      final caminho = await gerenciador.caminhoDoModelo();
      final ort = OnnxRuntime();
      _sessao = await ort.createSession(
        caminho,
        options: OrtSessionOptions(
          intraOpNumThreads: 2,
          providers: [OrtProvider.CPU],
        ),
      );
      _entrada = _sessao!.inputNames.first;
      _shaCarregado = sha;
      _pronto = true;
      _erro = null;
    } catch (e) {
      _erro = 'Não foi possível carregar o modelo baixado.\n$e';
      _pronto = false;
      rethrow;
    }
  }

  Future<void> encerrar() async {
    await _sessao?.close();
    _sessao = null;
    _pronto = false;
    _shaCarregado = null;
    _carga = null;
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

    // O modelo é carregado sob demanda: na primeira análise do modo profissional.
    await garantirPronto();

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
