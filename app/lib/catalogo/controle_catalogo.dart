import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;

import '../config.dart';
import 'firestore_rest.dart';
import 'modelos.dart';
import 'sessao_google.dart';

/// Reduz a foto para caber num documento do Firestore (limite de 1 MB), do
/// mesmo jeito do CitoAtlas: até 1024 px, JPEG, base64 abaixo de ~250 mil
/// caracteres. Pesado: rode em isolate (`compute`).
Uint8List comprimirParaCatalogo(Uint8List bytes) {
  final lida = img.decodeImage(bytes);
  if (lida == null) {
    throw const FormatException('Não foi possível ler esta imagem.');
  }
  final original = img.bakeOrientation(lida);

  img.Image redimensionar(img.Image origem, int lado) {
    final maior = origem.width > origem.height ? origem.width : origem.height;
    if (maior <= lado) return origem;
    return origem.width >= origem.height
        ? img.copyResize(origem, width: lado, interpolation: img.Interpolation.average)
        : img.copyResize(origem, height: lado, interpolation: img.Interpolation.average);
  }

  const limiteBase64 = 250000;
  var imagem = redimensionar(original, 1024);
  var jpeg = img.encodeJpg(imagem, quality: 78);
  if (base64Encode(jpeg).length > limiteBase64) {
    jpeg = img.encodeJpg(imagem, quality: 65);
  }
  if (base64Encode(jpeg).length > limiteBase64) {
    imagem = redimensionar(original, 800);
    jpeg = img.encodeJpg(imagem, quality: 65);
  }
  return jpeg;
}

/// Estado e operações do módulo de catálogo (CitoAtlas). Junta o acesso ao
/// banco ([CatalogoRemoto]) com o login ([SessaoGoogle]).
class ControleCatalogo extends ChangeNotifier {
  ControleCatalogo({SessaoGoogle? sessao}) : sessao = sessao ?? SessaoGoogle() {
    remoto = CatalogoRemoto(tokenDoUsuario: this.sessao.idToken);
    this.sessao.addListener(_aoMudarSessao);
  }

  final SessaoGoogle sessao;
  late final CatalogoRemoto remoto;

  final List<Lamina> laminas = [];
  Map<String, int>? _votos;
  final Map<String, Avaliacao> _minhas = {};
  String? _proximoToken;
  bool _fim = false;
  bool _carregando = false;
  bool _iniciou = false;
  ErroCatalogo? _erro;

  bool get carregando => _carregando;
  bool get fim => _fim;
  ErroCatalogo? get erro => _erro;

  /// Null enquanto a contagem não chegou (ou se ela falhou).
  bool get votosDisponiveis => _votos != null;

  UsuarioCatalogo? get usuario => sessao.usuario;

  int? votosDe(String slideId) => _votos == null ? null : (_votos![slideId] ?? 0);

  Avaliacao? minhaAvaliacaoDe(String slideId) => _minhas[slideId];

  bool ehCriador(Lamina l) =>
      usuario != null && l.criadorId != null && l.criadorId == usuario!.uid;

  // ---------------------------------------------------------------- leitura

  Future<void> iniciar() async {
    if (_iniciou) return;
    _iniciou = true;
    await Future.wait([carregarMais(), _carregarVotos()]);
  }

  Future<void> recarregar() async {
    laminas.clear();
    _proximoToken = null;
    _fim = false;
    _erro = null;
    _votos = null;
    _iniciou = true;
    notifyListeners();
    await Future.wait([carregarMais(), _carregarVotos(), _carregarMinhas()]);
  }

  Future<void> carregarMais() async {
    if (_carregando || _fim) return;
    _carregando = true;
    _erro = null;
    notifyListeners();
    try {
      final pagina = await remoto.listarLaminas(
        token: _proximoToken,
        tamanho: 24,
      );
      laminas.addAll(pagina.itens);
      _proximoToken = pagina.proximoToken;
      _fim = pagina.proximoToken == null;
    } on ErroCatalogo catch (e) {
      _erro = e;
    } catch (e) {
      _erro = ErroCatalogo(TipoErroCatalogo.inesperado, 'Erro inesperado: $e');
    } finally {
      _carregando = false;
      notifyListeners();
    }
  }

  Future<void> _carregarVotos() async {
    try {
      _votos = await remoto.contarVotos();
    } catch (_) {
      _votos = null; // a contagem é um complemento: sem ela, o catálogo segue
    }
    notifyListeners();
  }

  Future<void> _carregarMinhas() async {
    final u = usuario;
    _minhas.clear();
    if (u != null) {
      try {
        for (final a in await remoto.avaliacoesDoUsuario(u.uid)) {
          _minhas[a.slideId] = a;
        }
      } catch (_) {/* sem as avaliações, tudo aparece como pendente */}
    }
    notifyListeners();
  }

  void _aoMudarSessao() {
    _carregarMinhas();
  }

  // ------------------------------------------------------------------ escrita

  UsuarioCatalogo _exigirUsuario() {
    final u = usuario;
    if (u == null) {
      throw ErroCatalogo(
        TipoErroCatalogo.naoAutenticado,
        'Entre com sua conta Google para fazer isso.',
      );
    }
    return u;
  }

  Future<Avaliacao> _gravarAvaliacao(
    String slideId,
    Classificacao classificacao,
    String descricao,
    String observacoes,
  ) async {
    final u = _exigirUsuario();
    final a = Avaliacao(
      slideId: slideId,
      userId: u.uid,
      userName: u.nome,
      userEmail: u.email,
      classificacao: classificacao,
      descricao: descricao.trim(),
      observacoes: observacoes.trim(),
      criadaEm: DateTime.now().millisecondsSinceEpoch,
    );
    await remoto.salvarAvaliacao(a);
    return a;
  }

  /// Registra (ou refaz) a avaliação cega do usuário para a lâmina.
  Future<void> avaliar(
    Lamina l,
    Classificacao classificacao,
    String descricao,
    String observacoes,
  ) async {
    final jaAvaliada = _minhas.containsKey(l.id);
    final a = await _gravarAvaliacao(l.id, classificacao, descricao, observacoes);
    _minhas[l.id] = a;
    if (!jaAvaliada && _votos != null) {
      _votos![l.id] = (_votos![l.id] ?? 0) + 1;
    }
    notifyListeners();
  }

  /// Próximo ID livre no formato L-0001 (lê todos os IDs já usados).
  Future<String> proximoId() async {
    final ids = await remoto.listarIdsExistentes();
    var maior = 0;
    final padrao = RegExp(r'^L-(\d+)$', caseSensitive: false);
    for (final id in ids) {
      final m = padrao.firstMatch(id.trim());
      if (m != null) {
        final n = int.tryParse(m.group(1)!) ?? 0;
        if (n > maior) maior = n;
      }
    }
    return 'L-${(maior + 1).toString().padLeft(4, '0')}';
  }

  static String _dataIso(DateTime d) =>
      '${d.year.toString().padLeft(4, '0')}-${d.month.toString().padLeft(2, '0')}-${d.day.toString().padLeft(2, '0')}';

  /// Cadastra uma lâmina nova; o criador já a avalia, como no CitoAtlas.
  Future<void> criarLamina({
    required Uint8List jpeg,
    required String patientId,
    required Classificacao classificacao,
    required DateTime data,
    required String descricao,
    required String observacoes,
  }) async {
    final u = _exigirUsuario();
    final agora = DateTime.now().millisecondsSinceEpoch;
    final id = await remoto.criarLamina({
      'patientId': patientId.trim(),
      'imageUrl': 'data:image/jpeg;base64,${base64Encode(jpeg)}',
      'classification': classificacao.sigla,
      'date': _dataIso(data),
      'morphologyDescription': descricao.trim(),
      'notes': observacoes.trim(),
      'creatorId': u.uid,
      'creatorName': u.nome,
      'createdAt': agora,
      'updatedAt': agora,
    });
    final a = await _gravarAvaliacao(id, classificacao, descricao, observacoes);
    _minhas[id] = a;
    await recarregar();
  }

  /// Edita os dados de uma lâmina do próprio usuário. [jpeg] só vai se a foto
  /// foi trocada.
  Future<void> atualizarLamina(
    Lamina l, {
    Uint8List? jpeg,
    required String patientId,
    required Classificacao classificacao,
    required DateTime data,
    required String descricao,
    required String observacoes,
  }) async {
    _exigirUsuario();
    await remoto.atualizarLamina(l.id, {
      'patientId': patientId.trim(),
      if (jpeg != null) 'imageUrl': 'data:image/jpeg;base64,${base64Encode(jpeg)}',
      'classification': classificacao.sigla,
      'date': _dataIso(data),
      'morphologyDescription': descricao.trim(),
      'notes': observacoes.trim(),
      'updatedAt': DateTime.now().millisecondsSinceEpoch,
    });
    final a = await _gravarAvaliacao(l.id, classificacao, descricao, observacoes);
    _minhas[l.id] = a;
    await recarregar();
  }

  Future<void> excluirLamina(Lamina l) async {
    _exigirUsuario();
    await remoto.excluirLamina(l.id);
    _minhas.remove(l.id);
    await recarregar();
  }

  /// Quórum atingido para a lâmina, se a contagem estiver disponível.
  bool quorumAtingido(String slideId) =>
      (votosDe(slideId) ?? 0) >= catalogoVotosParaQuorum;

  @override
  void dispose() {
    sessao.removeListener(_aoMudarSessao);
    sessao.dispose();
    remoto.encerrar();
    super.dispose();
  }
}
