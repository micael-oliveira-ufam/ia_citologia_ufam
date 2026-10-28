import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:http/http.dart' as http;

import '../config.dart';

class UsuarioCatalogo {
  const UsuarioCatalogo({
    required this.uid,
    required this.nome,
    required this.email,
  });

  final String uid;
  final String nome;
  final String email;
}

/// Login com Google para o catálogo.
///
/// O plugin do Google só entrega um token do Google; ele é trocado por um token
/// do Firebase (a API REST do Firebase Auth), que é o que o Firestore entende.
/// Assim o aplicativo não precisa do SDK nativo do Firebase nem do
/// google-services.json. O que o login exige no Google Cloud/Firebase está no
/// README (cliente OAuth Android com o SHA-1 do aplicativo; no iOS, o cliente
/// iOS).
///
/// Os tokens ficam só na memória: ao fechar o aplicativo, é preciso entrar de
/// novo.
class SessaoGoogle extends ChangeNotifier {
  SessaoGoogle({http.Client? cliente}) : _cliente = cliente ?? http.Client();

  final http.Client _cliente;

  UsuarioCatalogo? _usuario;
  String? _idToken;
  String? _tokenDeRenovacao;
  DateTime _expira = DateTime.fromMillisecondsSinceEpoch(0);
  bool _googleIniciado = false;
  bool _entrando = false;
  String? _erro;

  UsuarioCatalogo? get usuario => _usuario;
  bool get logado => _usuario != null;
  bool get entrando => _entrando;
  String? get erro => _erro;

  static String get _chave => Uri.encodeQueryComponent(firebaseChaveApi);

  Future<void> entrar() async {
    if (_entrando) return;
    _entrando = true;
    _erro = null;
    notifyListeners();
    try {
      if (!_googleIniciado) {
        await GoogleSignIn.instance.initialize(
          serverClientId: googleClienteWeb,
          clientId: googleClienteIos.isEmpty ? null : googleClienteIos,
        );
        _googleIniciado = true;
      }
      final conta = await GoogleSignIn.instance.authenticate();
      final tokenGoogle = conta.authentication.idToken;
      if (tokenGoogle == null || tokenGoogle.isEmpty) {
        throw StateError('O Google não devolveu o token de identificação.');
      }
      await _trocarPorTokenFirebase(tokenGoogle);
    } catch (e) {
      final texto = '$e'.toLowerCase();
      if (texto.contains('cancel')) {
        _erro = null; // a pessoa desistiu: não é erro
      } else {
        _erro = 'Não foi possível entrar com o Google. Confira se o login está '
            'configurado para este aplicativo (veja o README).\n$e';
      }
    } finally {
      _entrando = false;
      notifyListeners();
    }
  }

  Future<void> sair() async {
    try {
      await GoogleSignIn.instance.signOut();
    } catch (_) {/* já estava fora */}
    _usuario = null;
    _idToken = null;
    _tokenDeRenovacao = null;
    _expira = DateTime.fromMillisecondsSinceEpoch(0);
    notifyListeners();
  }

  /// Token do Firebase válido para as chamadas ao Firestore, renovando se
  /// estiver perto de vencer. Null se ninguém entrou.
  Future<String?> idToken() async {
    if (_usuario == null) return null;
    final margem = DateTime.now().add(const Duration(minutes: 2));
    if (_idToken != null && _expira.isAfter(margem)) return _idToken;
    try {
      await _renovar();
    } catch (_) {
      await sair(); // não deu para renovar: pede login de novo
      return null;
    }
    return _idToken;
  }

  Future<void> _trocarPorTokenFirebase(String tokenGoogle) async {
    final resposta = await _cliente.post(
      Uri.parse(
        'https://identitytoolkit.googleapis.com/v1/accounts:signInWithIdp?key=$_chave',
      ),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'postBody': 'id_token=${Uri.encodeQueryComponent(tokenGoogle)}&providerId=google.com',
        'requestUri': 'http://localhost',
        'returnIdpCredential': true,
        'returnSecureToken': true,
      }),
    ).timeout(const Duration(seconds: 30));
    if (resposta.statusCode != 200) {
      throw StateError(
        'O Firebase recusou o login (${resposta.statusCode}). Confira se o '
        'provedor Google está ativado no Firebase Authentication.',
      );
    }
    final json = jsonDecode(utf8.decode(resposta.bodyBytes)) as Map<String, dynamic>;
    final email = (json['email'] as String?) ?? '';
    final nome = (json['displayName'] as String?)?.trim();
    _usuario = UsuarioCatalogo(
      uid: json['localId'] as String,
      nome: (nome != null && nome.isNotEmpty)
          ? nome
          : (email.contains('@') ? email.split('@').first : 'Citopatologista'),
      email: email,
    );
    _guardar(
      json['idToken'] as String,
      json['refreshToken'] as String,
      int.tryParse('${json['expiresIn']}') ?? 3600,
    );
  }

  Future<void> _renovar() async {
    final renovacao = _tokenDeRenovacao;
    if (renovacao == null) throw StateError('Sem token de renovação.');
    final resposta = await _cliente.post(
      Uri.parse('https://securetoken.googleapis.com/v1/token?key=$_chave'),
      headers: {'Content-Type': 'application/x-www-form-urlencoded'},
      body: 'grant_type=refresh_token&refresh_token=${Uri.encodeQueryComponent(renovacao)}',
    ).timeout(const Duration(seconds: 30));
    if (resposta.statusCode != 200) {
      throw StateError('Renovação recusada (${resposta.statusCode}).');
    }
    final json = jsonDecode(utf8.decode(resposta.bodyBytes)) as Map<String, dynamic>;
    _guardar(
      json['id_token'] as String,
      json['refresh_token'] as String,
      int.tryParse('${json['expires_in']}') ?? 3600,
    );
  }

  void _guardar(String idToken, String renovacao, int segundos) {
    _idToken = idToken;
    _tokenDeRenovacao = renovacao;
    _expira = DateTime.now().add(Duration(seconds: segundos));
  }

  @override
  void dispose() {
    _cliente.close();
    super.dispose();
  }
}
