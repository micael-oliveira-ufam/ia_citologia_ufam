import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

import '../config.dart';

/// Falha ao obter, conferir ou localizar o modelo. A mensagem é para o usuário.
class ErroModelo implements Exception {
  ErroModelo(this.mensagem, [this.detalhe]);

  final String mensagem;
  final Object? detalhe;

  @override
  String toString() => mensagem;
}

/// O que o `manifesto.json` publicado no GitHub Pages declara sobre o modelo
/// (gerado por tools/publicar_modelo.py).
class ManifestoModelo {
  const ManifestoModelo({
    required this.versao,
    required this.arquivo,
    required this.bytes,
    required this.sha256,
    required this.url,
    this.partes = const [],
    this.arquitetura,
    this.quantizacao,
  });

  final String versao;
  final String arquivo;
  final int bytes;
  final String sha256;
  final Uri url;

  /// Quando o arquivo passa de 100 MB (limite do GitHub), ele é publicado em
  /// partes; o app as baixa em ordem, junta e confere o SHA-256 do todo.
  final List<({Uri url, int bytes})> partes;
  final String? arquitetura;
  final String? quantizacao;

  /// Lê e valida. Recusa manifesto que declare outra ordem de classes: usar um
  /// modelo com ordem diferente inverteria diagnósticos sem nenhum erro visível.
  factory ManifestoModelo.deJson(Map<String, dynamic> json, Uri urlManifesto) {
    ErroModelo invalido(String motivo) =>
        ErroModelo('Manifesto do modelo inválido: $motivo.');

    final versao = json['versao'];
    final arquivo = json['arquivo'];
    final bytes = json['bytes'];
    final hash = json['sha256'];
    if (versao is! String || versao.isEmpty) throw invalido('versão ausente');
    if (arquivo is! String || arquivo.isEmpty) {
      throw invalido('arquivo ausente');
    }
    if (bytes is! int || bytes <= 0 || bytes > 1500000000) {
      throw invalido('tamanho');
    }
    if (hash is! String || !RegExp(r'^[0-9a-f]{64}$').hasMatch(hash)) {
      throw invalido('SHA-256');
    }

    final classes = json['classes'];
    final mesmaOrdem = classes is List &&
        classes.length == classesEsperadas.length &&
        List.generate(classes.length, (i) => classes[i] == classesEsperadas[i])
            .every((b) => b);
    if (!mesmaOrdem) {
      throw ErroModelo(
        'A ordem das classes do modelo ($classes) não bate com a esperada '
        '($classesEsperadas). A análise foi bloqueada para não inverter '
        'diagnósticos.',
      );
    }

    final url = urlManifesto.resolve(arquivo);
    if (url.scheme != 'https' || url.host != urlManifesto.host) {
      throw invalido('o arquivo precisa estar no mesmo site, em HTTPS');
    }

    var partes = <({Uri url, int bytes})>[];
    final brutas = json['partes'];
    if (brutas != null) {
      if (brutas is! List || brutas.isEmpty || brutas.length > 20) {
        throw invalido('partes');
      }
      var soma = 0;
      for (final p in brutas) {
        if (p is! Map || p['arquivo'] is! String || p['bytes'] is! int ||
            (p['bytes'] as int) <= 0) {
          throw invalido('parte inválida');
        }
        final u = urlManifesto.resolve(p['arquivo'] as String);
        if (u.scheme != 'https' || u.host != urlManifesto.host) {
          throw invalido('as partes precisam estar no mesmo site, em HTTPS');
        }
        partes.add((url: u, bytes: p['bytes'] as int));
        soma += p['bytes'] as int;
      }
      if (soma != bytes) throw invalido('a soma das partes difere do tamanho');
    }

    final arquitetura = json['arquitetura'];
    final quantizacao = json['quantizacao'];
    return ManifestoModelo(
      versao: versao,
      arquivo: arquivo,
      bytes: bytes,
      sha256: hash,
      url: url,
      partes: partes,
      arquitetura: arquitetura is String ? arquitetura : null,
      quantizacao: quantizacao is String ? quantizacao : null,
    );
  }

  Map<String, Object?> paraJson() => {
        'versao': versao,
        'arquivo': arquivo,
        'bytes': bytes,
        'sha256': sha256,
        'arquitetura': arquitetura,
        'quantizacao': quantizacao,
      };

  List<({Uri url, int bytes})> get partesParaBaixar =>
      partes.isEmpty ? [(url: url, bytes: bytes)] : partes;

  String get tamanhoLegivel => formatarBytes(bytes);
}

String formatarBytes(int bytes) {
  if (bytes < 1024 * 1024) return '${(bytes / 1024).ceil()} KB';
  return '${(bytes / 1048576).toStringAsFixed(1).replaceAll('.', ',')} MB';
}

enum FaseModelo {
  /// Consultando o manifesto publicado (ou lendo o que já está no aparelho).
  consultando,

  /// O repositório ainda não publicou nenhum modelo (manifesto 404).
  naoPublicado,

  /// Sem conexão e sem modelo no aparelho.
  semRede,

  /// Há modelo publicado, mas ainda não está no aparelho.
  naoInstalado,

  baixando,

  /// Modelo conferido e guardado no aparelho.
  pronto,

  erro,
}

class _SorvedouroDigest implements Sink<Digest> {
  Digest? resultado;

  @override
  void add(Digest data) => resultado = data;

  @override
  void close() {}
}

class _Cancelado implements Exception {}

/// Baixa o modelo ONNX do GitHub Pages, confere tamanho e SHA-256 e o guarda
/// no armazenamento privado do aplicativo. Só a consulta e o download usam a
/// rede; a análise em si continua inteiramente no aparelho.
class GerenciadorModelo extends ChangeNotifier {
  GerenciadorModelo({http.Client? cliente, String? urlBase})
      : _cliente = cliente ?? http.Client(),
        _urlManifesto = _montarUrlManifesto(urlBase ?? urlBaseModelo),
        _inerte = false;

  /// Para testes de interface: não toca em rede nem em disco.
  @visibleForTesting
  GerenciadorModelo.inerte()
      : _cliente = http.Client(),
        _urlManifesto = Uri.parse('https://exemplo.invalid/manifesto.json'),
        _inerte = true {
    _fase = FaseModelo.naoPublicado;
  }

  final http.Client _cliente;
  final Uri _urlManifesto;
  final bool _inerte;

  FaseModelo _fase = FaseModelo.consultando;
  double _progresso = 0;
  String? _mensagem;
  ManifestoModelo? _publicado;
  ManifestoModelo? _instalado;
  bool _iniciou = false;
  bool _cancelar = false;

  FaseModelo get fase => _fase;
  double get progresso => _progresso;
  String? get mensagem => _mensagem;
  ManifestoModelo? get publicado => _publicado;
  ManifestoModelo? get instalado => _instalado;
  String? get shaInstalado => _instalado?.sha256;

  /// Há modelo no aparelho e nenhum download em curso.
  bool get prontoParaAnalisar =>
      _instalado != null && _fase != FaseModelo.baixando;

  bool get atualizacaoDisponivel =>
      _instalado != null &&
      _publicado != null &&
      _publicado!.sha256 != _instalado!.sha256;

  static Uri _montarUrlManifesto(String base) {
    final comBarra = base.endsWith('/') ? base : '$base/';
    return Uri.parse(comBarra).resolve('manifesto.json');
  }

  void _definir(FaseModelo fase, {String? mensagem}) {
    _fase = fase;
    _mensagem = mensagem;
    notifyListeners();
  }

  // ------------------------------------------------------------------ disco

  Future<Directory> _pasta() async {
    final base = await getApplicationSupportDirectory();
    final pasta = Directory('${base.path}${Platform.pathSeparator}modelo');
    if (!await pasta.exists()) await pasta.create(recursive: true);
    return pasta;
  }

  String _nomeLocal(ManifestoModelo m) => 'citologia_${m.sha256.substring(0, 16)}.onnx';

  Future<File> _arquivoDe(ManifestoModelo m) async =>
      File('${(await _pasta()).path}${Platform.pathSeparator}${_nomeLocal(m)}');

  Future<File> _marcador() async =>
      File('${(await _pasta()).path}${Platform.pathSeparator}instalado.json');

  /// Lê o que está instalado e confere que o arquivo existe com o tamanho certo.
  Future<ManifestoModelo?> _lerInstalado() async {
    try {
      final marcador = await _marcador();
      if (!await marcador.exists()) return null;
      final json = jsonDecode(await marcador.readAsString()) as Map<String, dynamic>;
      final m = ManifestoModelo(
        versao: json['versao'] as String,
        arquivo: json['arquivo'] as String,
        bytes: json['bytes'] as int,
        sha256: json['sha256'] as String,
        url: _urlManifesto.resolve(json['arquivo'] as String),
        arquitetura: json['arquitetura'] as String?,
        quantizacao: json['quantizacao'] as String?,
      );
      final arquivo = await _arquivoDe(m);
      if (!await arquivo.exists() || await arquivo.length() != m.bytes) {
        return null;
      }
      return m;
    } catch (_) {
      return null;
    }
  }

  Future<void> _limparAntigos(ManifestoModelo manter) async {
    final pasta = await _pasta();
    final nomeAtual = _nomeLocal(manter);
    await for (final e in pasta.list()) {
      final nome = e.path.split(Platform.pathSeparator).last;
      if (e is File &&
          nome != nomeAtual &&
          nome != 'instalado.json' &&
          (nome.endsWith('.onnx') || nome.endsWith('.parcial'))) {
        try {
          await e.delete();
        } catch (_) {/* melhor esforço */}
      }
    }
  }

  // ---------------------------------------------------------------- consulta

  /// Chame ao entrar no modo profissional. Pode ser chamado várias vezes.
  Future<void> iniciar() async {
    if (_inerte || _iniciou) return;
    _iniciou = true;
    _instalado = await _lerInstalado();
    if (_instalado != null) {
      _definir(FaseModelo.pronto); // já utilizável, mesmo sem rede
    }
    await consultar();
  }

  /// Consulta o manifesto publicado. Se já há modelo no aparelho, falhas de
  /// rede são silenciosas: o modelo instalado continua valendo.
  Future<void> consultar() async {
    if (_inerte || _fase == FaseModelo.baixando) return;
    if (_instalado == null) _definir(FaseModelo.consultando);
    try {
      final resp = await _cliente
          .get(_urlManifesto, headers: {'Cache-Control': 'no-cache'})
          .timeout(const Duration(seconds: 20));
      if (resp.statusCode == 404) {
        _publicado = null;
        if (_instalado == null) _definir(FaseModelo.naoPublicado);
        return;
      }
      if (resp.statusCode != 200) {
        throw ErroModelo(
          'O site do modelo respondeu com erro ${resp.statusCode}.',
        );
      }
      final json = jsonDecode(utf8.decode(resp.bodyBytes));
      if (json is! Map<String, dynamic>) {
        throw ErroModelo('O manifesto do modelo está ilegível.');
      }
      _publicado = ManifestoModelo.deJson(json, _urlManifesto);
      if (_instalado == null) {
        _definir(FaseModelo.naoInstalado);
      } else {
        notifyListeners(); // pode ter surgido uma atualização
      }
    } on ErroModelo catch (e) {
      if (_instalado == null) _definir(FaseModelo.erro, mensagem: e.mensagem);
    } on TimeoutException {
      if (_instalado == null) {
        _definir(
          FaseModelo.semRede,
          mensagem: 'O site do modelo não respondeu. Verifique a conexão.',
        );
      }
    } catch (e) {
      if (_instalado == null) {
        _definir(
          FaseModelo.semRede,
          mensagem: 'Sem conexão para consultar o modelo.',
        );
      }
    }
  }

  // ---------------------------------------------------------------- download

  void cancelar() => _cancelar = true;

  /// Baixa a versão publicada, confere e instala. Em caso de falha, descarta o
  /// arquivo parcial e mantém o modelo anterior, se houver.
  Future<void> baixar() async {
    final alvo = _publicado;
    if (_inerte || alvo == null || _fase == FaseModelo.baixando) return;

    final tinhaInstalado = _instalado != null;
    _cancelar = false;
    _progresso = 0;
    _definir(FaseModelo.baixando);

    File? parcial;
    IOSink? saida;
    try {
      final destino = await _arquivoDe(alvo);
      parcial = File('${destino.path}.parcial');
      if (await parcial.exists()) await parcial.delete();

      saida = parcial.openWrite();
      final sorvedouro = _SorvedouroDigest();
      final hash = sha256.startChunkedConversion(sorvedouro);
      var recebido = 0;
      var ultimoPercentual = -1;

      for (final parte in alvo.partesParaBaixar) {
        final pedido = http.Request('GET', parte.url);
        final resposta =
            await _cliente.send(pedido).timeout(const Duration(seconds: 30));
        if (resposta.statusCode != 200) {
          throw ErroModelo(
            'O arquivo do modelo não foi encontrado (erro ${resposta.statusCode}).',
          );
        }
        final limiteParte = recebido + parte.bytes;
        await for (final bloco
            in resposta.stream.timeout(const Duration(seconds: 30))) {
          if (_cancelar) throw _Cancelado();
          recebido += bloco.length;
          if (recebido > limiteParte) {
            throw ErroModelo(
              'O arquivo baixado é maior do que o manifesto declara. Descartado.',
            );
          }
          saida.add(bloco);
          hash.add(bloco);
          final percentual = (recebido * 100) ~/ alvo.bytes;
          if (percentual != ultimoPercentual) {
            ultimoPercentual = percentual;
            _progresso = recebido / alvo.bytes;
            notifyListeners();
          }
        }
        if (recebido != limiteParte) {
          throw ErroModelo('O download terminou incompleto. Tente de novo.');
        }
      }
      hash.close();
      await saida.flush();
      await saida.close();
      saida = null;

      if (recebido != alvo.bytes) {
        throw ErroModelo('O download terminou incompleto. Tente de novo.');
      }
      if (sorvedouro.resultado.toString() != alvo.sha256) {
        throw ErroModelo(
          'O arquivo baixado não confere com o SHA-256 do manifesto. Descartado.',
        );
      }

      if (await destino.exists()) await destino.delete();
      await parcial.rename(destino.path);
      await (await _marcador()).writeAsString(jsonEncode(alvo.paraJson()));
      await _limparAntigos(alvo);
      _instalado = alvo;
      _definir(FaseModelo.pronto);
    } on _Cancelado {
      await _descartar(saida, parcial);
      _definir(tinhaInstalado ? FaseModelo.pronto : FaseModelo.naoInstalado);
    } on ErroModelo catch (e) {
      await _descartar(saida, parcial);
      _definir(FaseModelo.erro, mensagem: e.mensagem);
    } on TimeoutException {
      await _descartar(saida, parcial);
      _definir(
        FaseModelo.erro,
        mensagem: 'A conexão ficou parada durante o download. Tente de novo.',
      );
    } catch (e) {
      await _descartar(saida, parcial);
      _definir(
        FaseModelo.erro,
        mensagem: 'O download do modelo falhou. Verifique a conexão e o '
            'espaço livre no aparelho. ($e)',
      );
    }
  }

  Future<void> _descartar(IOSink? saida, File? parcial) async {
    try {
      await saida?.close();
    } catch (_) {/* já fechado */}
    try {
      if (parcial != null && await parcial.exists()) await parcial.delete();
    } catch (_) {/* melhor esforço */}
  }

  /// Apaga o modelo do aparelho (libera espaço).
  Future<void> remover() async {
    if (_inerte || _fase == FaseModelo.baixando) return;
    try {
      final pasta = await _pasta();
      await for (final e in pasta.list()) {
        if (e is File) await e.delete();
      }
    } catch (_) {/* melhor esforço */}
    _instalado = null;
    _definir(_publicado != null ? FaseModelo.naoInstalado : FaseModelo.consultando);
    if (_publicado == null) await consultar();
  }

  /// Caminho do arquivo `.onnx` instalado, para o ONNX Runtime abrir.
  Future<String> caminhoDoModelo() async {
    final m = _instalado;
    if (m == null) {
      throw ErroModelo(
        'O modelo de citologia ainda não está no aparelho. Baixe-o na aba '
        'Analisar (uma única vez).',
      );
    }
    final arquivo = await _arquivoDe(m);
    if (!await arquivo.exists()) {
      _instalado = null;
      notifyListeners();
      throw ErroModelo(
        'O arquivo do modelo não foi encontrado no aparelho. Baixe-o de novo.',
      );
    }
    return arquivo.path;
  }

  @override
  void dispose() {
    _cliente.close();
    super.dispose();
  }
}
