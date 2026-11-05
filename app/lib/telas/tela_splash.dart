import 'dart:async';

import 'package:flutter/material.dart';

import '../tema.dart';
import '../versao.dart';

/// Tela de abertura: marca, nome e versão. Some sozinha e mostra [proxima].
///
/// Respeita "reduzir movimento" do sistema: nesse caso aparece já pronta.
class TelaSplash extends StatefulWidget {
  const TelaSplash({
    super.key,
    required this.proxima,
    this.duracao = const Duration(milliseconds: 2400),
  });

  final Widget proxima;
  final Duration duracao;

  @override
  State<TelaSplash> createState() => _TelaSplashState();
}

class _TelaSplashState extends State<TelaSplash>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controle;
  late final Animation<double> _escala;
  late final Animation<double> _opacidade;
  Timer? _timer;
  bool _concluido = false;

  @override
  void initState() {
    super.initState();
    _controle = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    );
    _escala = Tween<double>(begin: 0.7, end: 1).animate(
      CurvedAnimation(parent: _controle, curve: Curves.easeOutBack),
    );
    _opacidade = CurvedAnimation(parent: _controle, curve: Curves.easeOut);
    _controle.forward();
    _timer = Timer(widget.duracao, () {
      if (mounted) setState(() => _concluido = true);
    });
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (MediaQuery.disableAnimationsOf(context)) _controle.value = 1;
  }

  @override
  void dispose() {
    _timer?.cancel();
    _controle.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedSwitcher(
      duration: const Duration(milliseconds: 450),
      child: _concluido
          ? KeyedSubtree(key: const ValueKey('app'), child: widget.proxima)
          : KeyedSubtree(
              key: const ValueKey('splash'),
              child: _conteudo(context),
            ),
    );
  }

  Widget _conteudo(BuildContext context) {
    return Scaffold(
      body: Container(
        width: double.infinity,
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [Cores.teal, Cores.tealProfundo],
          ),
        ),
        child: SafeArea(
          child: Column(
            children: [
              const Spacer(),
              FadeTransition(
                opacity: _opacidade,
                child: ScaleTransition(
                  scale: _escala,
                  child: Column(
                    children: [
                      Container(
                        width: 120,
                        height: 120,
                        decoration: BoxDecoration(
                          color: Colors.white.withValues(alpha: 0.16),
                          shape: BoxShape.circle,
                          border: Border.all(
                            color: Colors.white.withValues(alpha: 0.35),
                            width: 2,
                          ),
                        ),
                        child: const Icon(
                          Icons.biotech_outlined,
                          size: 60,
                          color: Colors.white,
                          semanticLabel: 'Logotipo do CitoIlumina',
                        ),
                      ),
                      const SizedBox(height: 28),
                      const Text(
                        'CitoIlumina',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 36,
                          fontWeight: FontWeight.w800,
                          letterSpacing: -0.5,
                        ),
                      ),
                      const SizedBox(height: 10),
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 40),
                        child: Text(
                          'Citologia cervical com inteligência artificial',
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            color: Colors.white.withValues(alpha: 0.9),
                            fontSize: 16,
                            height: 1.4,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const Spacer(),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 48),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(4),
                  child: LinearProgressIndicator(
                    minHeight: 3,
                    backgroundColor: Colors.white.withValues(alpha: 0.2),
                    color: Colors.white,
                  ),
                ),
              ),
              const SizedBox(height: 28),
              Text(
                'UFAM · SEMSA · v$versaoNome ($versaoBuild)',
                style: TextStyle(
                  color: Colors.white.withValues(alpha: 0.75),
                  fontSize: 12.5,
                  letterSpacing: 0.6,
                ),
              ),
              const SizedBox(height: 28),
            ],
          ),
        ),
      ),
    );
  }
}
