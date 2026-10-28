import 'package:flutter/material.dart';

import 'casca.dart';
import 'data/estatisticas.dart';
import 'perfil.dart';
import 'servicos/classificador.dart';
import 'servicos/gerenciador_modelo.dart';
import 'servicos/modelo_risco.dart';
import 'telas/escolha_perfil.dart';
import 'tema.dart';

void main() {
  runApp(const ColoIaApp());
}

class ColoIaApp extends StatelessWidget {
  /// [gerenciadorModelo] só é informado em testes (um gerenciador inerte, que
  /// não usa rede nem disco).
  const ColoIaApp({super.key, this.gerenciadorModelo});

  final GerenciadorModelo? gerenciadorModelo;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Colo do Útero com IA',
      debugShowCheckedModeBanner: false,
      theme: construirTema(),
      home: Raiz(gerenciadorModelo: gerenciadorModelo),
    );
  }
}

/// Guarda o perfil escolhido e os serviços compartilhados. A troca de perfil
/// acontece aqui dentro (e não no `home` do MaterialApp), que é o jeito
/// confiável de alternar entre a escolha e o aplicativo.
class Raiz extends StatefulWidget {
  const Raiz({super.key, this.gerenciadorModelo});

  final GerenciadorModelo? gerenciadorModelo;

  @override
  State<Raiz> createState() => _RaizState();
}

class _RaizState extends State<Raiz> {
  /// Baixa e guarda o modelo de citologia. Só é acionado no modo profissional.
  late final GerenciadorModelo _gerenciador =
      widget.gerenciadorModelo ?? GerenciadorModelo();

  /// O modelo ONNX só é aberto quando o profissional faz a primeira análise.
  late final Classificador _classificador = Classificador(_gerenciador);

  // Carregados sob demanda (late): só quando alguém entra num perfil, e uma
  // única vez, mesmo que o perfil seja trocado.
  late final Future<ModeloRisco> _modeloRisco = ModeloRisco.carregar();
  late final Future<List<Estatistica>> _estatisticas = carregarEstatisticas();

  Perfil? _perfil;

  @override
  void dispose() {
    _classificador.encerrar();
    _gerenciador.dispose();
    super.dispose();
  }

  void _escolherPerfil(Perfil p) {
    if (p.ehProfissional) _gerenciador.iniciar(); // consulta o modelo publicado
    setState(() => _perfil = p);
  }

  @override
  Widget build(BuildContext context) {
    final perfil = _perfil;
    if (perfil == null) {
      return TelaEscolhaPerfil(
        aoEscolher: _escolherPerfil,
      );
    }
    return Casca(
      key: ValueKey(perfil),
      perfil: perfil,
      classificador: _classificador,
      modeloRisco: _modeloRisco,
      estatisticas: _estatisticas,
      aoTrocarPerfil: () => setState(() => _perfil = null),
    );
  }
}
