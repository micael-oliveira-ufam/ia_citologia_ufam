import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

import '../config.dart';
import 'modelos.dart';

enum TipoErroCatalogo { rede, permissao, chave, naoAutenticado, inesperado }

class ErroCatalogo implements Exception {
  ErroCatalogo(this.tipo, this.mensagem, [this.detalhe]);

  final TipoErroCatalogo tipo;
  final String mensagem;
  final String? detalhe;

  @override
  String toString() => mensagem;
}

class PaginaLaminas {
  const PaginaLaminas(this.itens, this.proximoToken);

  final List<Lamina> itens;
  final String? proximoToken;
}

/// Acesso ao Firestore do CitoAtlas pela API REST (sem o SDK do Firebase).
///
/// Leitura de lâminas e de contagem de votos não exige login (as regras do
/// projeto hoje permitem). Escrita exige o token de um usuário autenticado,
/// fornecido por [tokenDoUsuario].
class CatalogoRemoto {
  CatalogoRemoto({http.Client? cliente, this.tokenDoUsuario})
      : _cliente = cliente ?? http.Client();

  final http.Client _cliente;

  /// Devolve um ID token do Firebase válido, ou null se ninguém entrou.
  final Future<String?> Function()? tokenDoUsuario;

  static final String _raiz =
      'https://firestore.googleapis.com/v1/projects/$firebaseProjeto'
      '/databases/${Uri.encodeComponent(firebaseBanco)}/documents';

  // ------------------------------------------------------------------ valores

  static Object? _lerValor(Map<String, dynamic> v) {
    if (v.containsKey('stringValue')) return v['stringValue'];
    if (v.containsKey('integerValue')) {
      return int.tryParse('${v['integerValue']}') ?? 0;
    }
    if (v.containsKey('doubleValue')) return (v['doubleValue'] as num).toDouble();
    if (v.containsKey('booleanValue')) return v['booleanValue'];
    if (v.containsKey('timestampValue')) {
      return DateTime.tryParse('${v['timestampValue']}')?.millisecondsSinceEpoch;
    }
    return null;
  }

  static Map<String, Object?> _lerCampos(Map<String, dynamic>? campos) {
    final saida = <String, Object?>{};
    if (campos == null) return saida;
    for (final e in campos.entries) {
      final valor = e.value;
      if (valor is Map<String, dynamic>) saida[e.key] = _lerValor(valor);
    }
    return saida;
  }

  static Map<String, Object?> _escreverValor(Object? v) {
    if (v == null) return {'nullValue': null};
    if (v is String) return {'stringValue': v};
    if (v is bool) return {'booleanValue': v};
    if (v is int) return {'integerValue': v.toString()};
    if (v is double) return {'doubleValue': v};
    throw ArgumentError('Tipo não suportado: ${v.runtimeType}');
  }

  static Map<String, Object?> _corpo(Map<String, Object?> campos) => {
        'fields': {
          for (final e in campos.entries) e.key: _escreverValor(e.value),
        },
      };

  static String _idDoDocumento(Map<String, dynamic> doc) =>
      (doc['name'] as String).split('/').last;

  // --------------------------------------------------------------------- HTTP

  Future<Map<String, String>> _cabecalhos({required bool exigeLogin}) async {
    final token = await tokenDoUsuario?.call();
    if (exigeLogin && token == null) {
      throw ErroCatalogo(
        TipoErroCatalogo.naoAutenticado,
        'Entre com sua conta Google para fazer isso.',
      );
    }
    return {
      'Content-Type': 'application/json',
      if (token != null) 'Authorization': 'Bearer $token',
    };
  }

  Future<dynamic> _enviar(
    String metodo,
    Uri url, {
    Object? corpo,
    bool exigeLogin = false,
  }) async {
    final cabecalhos = await _cabecalhos(exigeLogin: exigeLogin);
    final pedido = http.Request(metodo, url)..headers.addAll(cabecalhos);
    if (corpo != null) pedido.body = jsonEncode(corpo);

    http.Response resposta;
    try {
      resposta = await http.Response.fromStream(
        await _cliente.send(pedido).timeout(const Duration(seconds: 30)),
      );
    } on TimeoutException {
      throw ErroCatalogo(
        TipoErroCatalogo.rede,
        'O banco de dados do catálogo não respondeu. Tente de novo.',
      );
    } catch (e) {
      throw ErroCatalogo(
        TipoErroCatalogo.rede,
        'Sem conexão com o banco de dados do catálogo.',
        '$e',
      );
    }

    if (resposta.statusCode >= 200 && resposta.statusCode < 300) {
      if (resposta.body.isEmpty) return null;
      return jsonDecode(utf8.decode(resposta.bodyBytes));
    }

    final texto = utf8.decode(resposta.bodyBytes, allowMalformed: true);
    final detalhe = texto.length > 300 ? texto.substring(0, 300) : texto;
    if (resposta.statusCode == 401) {
      throw ErroCatalogo(
        TipoErroCatalogo.naoAutenticado,
        'Sua sessão expirou. Entre de novo com o Google.',
        detalhe,
      );
    }
    if (resposta.statusCode == 403) {
      if (RegExp(r'referer|BLOCKED|API_KEY', caseSensitive: false)
          .hasMatch(texto)) {
        throw ErroCatalogo(
          TipoErroCatalogo.chave,
          'A chave de API do projeto bloqueia este aplicativo. No Google '
          'Cloud, em Credenciais, permita o uso da chave pelo aplicativo.',
          detalhe,
        );
      }
      throw ErroCatalogo(
        TipoErroCatalogo.permissao,
        'O banco de dados recusou a operação (permissão negada).',
        detalhe,
      );
    }
    throw ErroCatalogo(
      TipoErroCatalogo.inesperado,
      'O banco de dados respondeu com erro ${resposta.statusCode}.',
      detalhe,
    );
  }

  Uri _url(
    String caminho, {
    Map<String, String> consulta = const {},
    List<MapEntry<String, String>> repetidos = const [],
  }) {
    final partes = <String>[
      'key=${Uri.encodeQueryComponent(firebaseChaveApi)}',
      for (final e in consulta.entries)
        '${Uri.encodeQueryComponent(e.key)}=${Uri.encodeQueryComponent(e.value)}',
      for (final e in repetidos)
        '${Uri.encodeQueryComponent(e.key)}=${Uri.encodeQueryComponent(e.value)}',
    ];
    return Uri.parse('$_raiz$caminho?${partes.join('&')}');
  }

  // ------------------------------------------------------------------ leitura

  Lamina? _lamina(Map<String, dynamic> doc) {
    final c = _lerCampos(doc['fields'] as Map<String, dynamic>?);
    final imagem = c['imageUrl'];
    if (imagem is! String) return null;
    final l = Lamina(
      id: _idDoDocumento(doc),
      patientId: (c['patientId'] as String?) ?? '',
      imageUrl: imagem,
      criadaEm: (c['createdAt'] as num?)?.toInt() ?? 0,
      data: c['date'] as String?,
      classificacao: c['classification'] as String?,
      descricao: c['morphologyDescription'] as String?,
      observacoes: c['notes'] as String?,
      criadorId: c['creatorId'] as String?,
    );
    return l.temImagemUtilizavel ? l : null;
  }

  /// Uma página de lâminas (ordenada da mais nova para a mais antiga dentro da
  /// página; o banco não aceita orderBy na listagem). Documentos sem
  /// imagem utilizável são descartados.
  Future<PaginaLaminas> listarLaminas({String? token, int tamanho = 24}) async {
    final json = await _enviar(
      'GET',
      _url('/slides', consulta: {
        'pageSize': '$tamanho',
        if (token != null) 'pageToken': token,
      }),
    ) as Map<String, dynamic>;
    final docs = (json['documents'] as List<dynamic>?) ?? const [];
    final itens = <Lamina>[];
    for (final d in docs) {
      final l = _lamina(d as Map<String, dynamic>);
      if (l != null) itens.add(l);
    }
    // O banco não aceita orderBy na listagem: ordena aqui, dentro da página.
    itens.sort((a, b) => b.criadaEm.compareTo(a.criadaEm));
    return PaginaLaminas(itens, json['nextPageToken'] as String?);
  }

  /// Todos os patientId já usados (só esse campo vem do servidor), para
  /// calcular o próximo L-0000 livre.
  Future<List<String>> listarIdsExistentes({int maxPaginas = 40}) async {
    final ids = <String>[];
    String? token;
    for (var i = 0; i < maxPaginas; i++) {
      final json = await _enviar(
        'GET',
        _url(
          '/slides',
          consulta: {'pageSize': '300', if (token != null) 'pageToken': token},
          repetidos: [const MapEntry('mask.fieldPaths', 'patientId')],
        ),
      ) as Map<String, dynamic>;
      for (final d in (json['documents'] as List<dynamic>?) ?? const []) {
        final c = _lerCampos((d as Map<String, dynamic>)['fields'] as Map<String, dynamic>?);
        final id = c['patientId'];
        if (id is String) ids.add(id);
      }
      token = json['nextPageToken'] as String?;
      if (token == null) break;
    }
    return ids;
  }

  /// Número de avaliações por lâmina. Pede ao servidor só o campo slideId:
  /// os documentos de avaliação têm nome e e-mail dos avaliadores, que não
  /// precisam chegar ao aparelho.
  Future<Map<String, int>> contarVotos({int maxPaginas = 40}) async {
    final contagem = <String, int>{};
    String? token;
    for (var i = 0; i < maxPaginas; i++) {
      final json = await _enviar(
        'GET',
        _url(
          '/evaluations',
          consulta: {'pageSize': '300', if (token != null) 'pageToken': token},
          repetidos: [const MapEntry('mask.fieldPaths', 'slideId')],
        ),
      ) as Map<String, dynamic>;
      for (final d in (json['documents'] as List<dynamic>?) ?? const []) {
        final c = _lerCampos((d as Map<String, dynamic>)['fields'] as Map<String, dynamic>?);
        final id = c['slideId'];
        if (id is String) contagem[id] = (contagem[id] ?? 0) + 1;
      }
      token = json['nextPageToken'] as String?;
      if (token == null) break;
    }
    return contagem;
  }

  Avaliacao? _avaliacao(Map<String, dynamic> doc) {
    final c = _lerCampos(doc['fields'] as Map<String, dynamic>?);
    final classificacao = Classificacao.deSigla(c['classification'] as String?);
    final slideId = c['slideId'];
    final userId = c['userId'];
    if (classificacao == null || slideId is! String || userId is! String) {
      return null;
    }
    return Avaliacao(
      slideId: slideId,
      userId: userId,
      userName: (c['userName'] as String?) ?? '',
      userEmail: (c['userEmail'] as String?) ?? '',
      classificacao: classificacao,
      descricao: (c['morphologyDescription'] as String?) ?? '',
      observacoes: (c['notes'] as String?) ?? '',
      criadaEm: (c['createdAt'] as num?)?.toInt() ?? 0,
    );
  }

  Future<List<Map<String, dynamic>>> _consultarIgual(
    String colecao,
    String campo,
    String valor,
  ) async {
    final resposta = await _enviar(
      'POST',
      Uri.parse('$_raiz:runQuery?key=${Uri.encodeQueryComponent(firebaseChaveApi)}'),
      corpo: {
        'structuredQuery': {
          'from': [
            {'collectionId': colecao},
          ],
          'where': {
            'fieldFilter': {
              'field': {'fieldPath': campo},
              'op': 'EQUAL',
              'value': {'stringValue': valor},
            },
          },
        },
      },
    );
    final docs = <Map<String, dynamic>>[];
    for (final item in (resposta as List<dynamic>)) {
      final doc = (item as Map<String, dynamic>)['document'];
      if (doc is Map<String, dynamic>) docs.add(doc);
    }
    return docs;
  }

  /// Avaliações feitas por um usuário (as dele, e só as dele).
  Future<List<Avaliacao>> avaliacoesDoUsuario(String uid) async {
    final docs = await _consultarIgual('evaluations', 'userId', uid);
    return [
      for (final d in docs)
        if (_avaliacao(d) case final a?) a,
    ];
  }

  // ----------------------------------------------------------------- escrita

  /// Cria uma lâmina e devolve o ID do documento.
  Future<String> criarLamina(Map<String, Object?> campos) async {
    final json = await _enviar(
      'POST',
      _url('/slides'),
      corpo: _corpo(campos),
      exigeLogin: true,
    ) as Map<String, dynamic>;
    return _idDoDocumento(json);
  }

  Future<void> atualizarLamina(String id, Map<String, Object?> campos) async {
    await _enviar(
      'PATCH',
      _url(
        '/slides/${Uri.encodeComponent(id)}',
        repetidos: [
          for (final k in campos.keys) MapEntry('updateMask.fieldPaths', k),
        ],
      ),
      corpo: _corpo(campos),
      exigeLogin: true,
    );
  }

  /// Apaga a lâmina e, em melhor esforço, as avaliações dela.
  Future<void> excluirLamina(String id) async {
    await _enviar(
      'DELETE',
      _url('/slides/${Uri.encodeComponent(id)}'),
      exigeLogin: true,
    );
    try {
      for (final d in await _consultarIgual('evaluations', 'slideId', id)) {
        await _enviar(
          'DELETE',
          _url('/evaluations/${Uri.encodeComponent(_idDoDocumento(d))}'),
          exigeLogin: true,
        );
      }
    } on ErroCatalogo {
      // As regras podem não deixar apagar avaliações de outras pessoas.
    }
  }

  /// Grava (ou substitui) a avaliação do usuário para a lâmina.
  Future<void> salvarAvaliacao(Avaliacao a) async {
    await _enviar(
      'PATCH',
      _url('/evaluations/${Uri.encodeComponent(a.id)}'),
      corpo: _corpo({
        'id': a.id,
        'slideId': a.slideId,
        'userId': a.userId,
        'userName': a.userName,
        'userEmail': a.userEmail,
        'classification': a.classificacao.sigla,
        'morphologyDescription': a.descricao,
        'notes': a.observacoes,
        'createdAt': a.criadaEm,
      }),
      exigeLogin: true,
    );
  }

  void encerrar() => _cliente.close();
}
