import 'package:flutter/material.dart';

import 'servicos/classificador.dart';
import 'telas/tela_abertura.dart';
import 'telas/tela_inicial.dart';
import 'tema.dart';

void main() {
  runApp(const CitoPred());
}

class CitoPred extends StatefulWidget {
  const CitoPred({super.key});

  @override
  State<CitoPred> createState() => _CitoPredState();
}

class _CitoPredState extends State<CitoPred> {
  final _classificador = Classificador();
  late Future<void> _carga;
  bool _pronto = false;
  Object? _erro;

  @override
  void initState() {
    super.initState();
    _carga = _classificador.iniciar();
  }

  @override
  void dispose() {
    _classificador.encerrar();
    super.dispose();
  }

  void _tentarNovamente() {
    setState(() {
      _erro = null;
      _pronto = false;
      _carga = _classificador.iniciar();
    });
  }

  @override
  Widget build(BuildContext context) {
    Widget conteudo;
    if (_erro != null) {
      conteudo = _FalhaNoModelo(
        mensagem: _erro.toString(),
        aoTentar: _tentarNovamente,
      );
    } else if (_pronto) {
      conteudo = TelaInicial(classificador: _classificador);
    } else {
      conteudo = TelaAbertura(
        // A chave força a reconstrução da abertura quando o usuário tenta de novo.
        key: ValueKey(_carga),
        carga: _carga,
        aoConcluir: () => setState(() => _pronto = true),
        aoFalhar: (erro) => setState(() => _erro = erro),
      );
    }

    return MaterialApp(
      title: 'CitoPred',
      debugShowCheckedModeBanner: false,
      theme: construirTema(),
      home: conteudo,
    );
  }
}

class _FalhaNoModelo extends StatelessWidget {
  const _FalhaNoModelo({required this.mensagem, required this.aoTentar});

  final String mensagem;
  final VoidCallback aoTentar;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(28),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.cloud_off_outlined, size: 48, color: Cores.tintaFraca),
              const SizedBox(height: 18),
              Text(
                'O modelo não pôde ser carregado',
                style: Theme.of(context).textTheme.titleMedium,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 10),
              const Text(
                'O arquivo do modelo deveria estar embutido no aplicativo. '
                'Reinstale a partir do APK oficial do projeto.',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 15, color: Cores.tintaFraca, height: 1.5),
              ),
              const SizedBox(height: 16),
              Text(
                mensagem,
                textAlign: TextAlign.center,
                style: const TextStyle(fontSize: 12, color: Cores.tintaFraca),
              ),
              const SizedBox(height: 28),
              FilledButton(onPressed: aoTentar, child: const Text('Tentar de novo')),
            ],
          ),
        ),
      ),
    );
  }
}
