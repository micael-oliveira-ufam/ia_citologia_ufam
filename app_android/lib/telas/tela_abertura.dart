import 'package:flutter/material.dart';

import '../data/projeto.dart';
import '../tema.dart';
import '../widgets/comuns.dart';

/// Tela de abertura do CitoPred.
///
/// Fica visível enquanto o modelo é carregado, e por no mínimo [duracaoMinima]
/// para que dê tempo de ler os créditos. Se o carregamento terminar antes, a
/// tela espera; se demorar mais, ela permanece até terminar.
class TelaAbertura extends StatefulWidget {
  const TelaAbertura({
    super.key,
    required this.carga,
    required this.aoConcluir,
    required this.aoFalhar,
  });

  final Future<void> carga;
  final VoidCallback aoConcluir;
  final ValueChanged<Object> aoFalhar;

  static const Duration duracaoMinima = Duration(milliseconds: 2600);

  @override
  State<TelaAbertura> createState() => _TelaAberturaState();
}

class _TelaAberturaState extends State<TelaAbertura>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controle;

  @override
  void initState() {
    super.initState();
    _controle = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    )..forward();
    _aguardar();
  }

  Future<void> _aguardar() async {
    final relogio = Future<void>.delayed(TelaAbertura.duracaoMinima);
    try {
      await Future.wait([widget.carga, relogio]);
      if (mounted) widget.aoConcluir();
    } catch (erro) {
      if (mounted) widget.aoFalhar(erro);
    }
  }

  @override
  void dispose() {
    _controle.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final aparecer = CurvedAnimation(parent: _controle, curve: Curves.easeOut);

    return Scaffold(
      backgroundColor: Cores.porcelana,
      body: SafeArea(
        child: FadeTransition(
          opacity: aparecer,
          child: LayoutBuilder(
            builder: (context, restricoes) {
              return SingleChildScrollView(
                child: ConstrainedBox(
                  constraints: BoxConstraints(minHeight: restricoes.maxHeight),
                  child: Padding(
                    padding: const EdgeInsets.fromLTRB(24, 28, 24, 20),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        const _Marca(),
                        const _Participantes(),
                        const _Instituicoes(),
                        const _Rodape(),
                      ],
                    ),
                  ),
                ),
              );
            },
          ),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------

class _Marca extends StatelessWidget {
  const _Marca();

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Container(
          width: 96,
          height: 96,
          decoration: BoxDecoration(
            color: Cores.superficie,
            borderRadius: BorderRadius.circular(24),
            border: Border.all(color: Cores.linha),
          ),
          clipBehavior: Clip.antiAlias,
          child: Image.asset(
            'assets/icone/icone_1024.png',
            fit: BoxFit.cover,
            errorBuilder: (_, __, ___) => const Icon(
              Icons.biotech_outlined,
              size: 52,
              color: Cores.teal,
            ),
          ),
        ),
        const SizedBox(height: 18),
        const Text(
          'CitoPred',
          textAlign: TextAlign.center,
          style: TextStyle(
            fontSize: 34,
            fontWeight: FontWeight.w800,
            color: Cores.tinta,
            letterSpacing: -0.5,
          ),
        ),
        const SizedBox(height: 6),
        const Text(
          'Apoio ao diagnóstico citológico\nem meio líquido',
          textAlign: TextAlign.center,
          style: TextStyle(fontSize: 15.5, color: Cores.tintaFraca, height: 1.4),
        ),
        const SizedBox(height: 14),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
          decoration: BoxDecoration(
            color: Cores.tealClaro,
            borderRadius: BorderRadius.circular(20),
          ),
          child: const Text(
            'ANÁLISE LOCAL, SEM ENVIO DE IMAGENS',
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 10.5,
              fontWeight: FontWeight.w700,
              color: Cores.teal,
              letterSpacing: 0.9,
            ),
          ),
        ),
      ],
    );
  }
}

// ---------------------------------------------------------------------------

class _Participantes extends StatelessWidget {
  const _Participantes();

  @override
  Widget build(BuildContext context) {
    // Agrupa por papel para a lista não virar um bloco indistinto de nomes.
    final grupos = <String, List<Pessoa>>{};
    for (final p in equipe) {
      grupos.putIfAbsent(p.vinculo.contains('SEMSA') ? 'SEMSA Manaus'
          : p.vinculo.contains('IComp') ? 'Instituto de Computação, UFAM'
          : 'Faculdade de Ciências Farmacêuticas, UFAM', () => []).add(p);
    }

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 24),
      child: Column(
        children: [
          for (final entrada in grupos.entries) ...[
            Text(
              entrada.key.toUpperCase(),
              textAlign: TextAlign.center,
              style: const TextStyle(
                fontSize: 10.5,
                fontWeight: FontWeight.w700,
                color: Cores.teal,
                letterSpacing: 1.1,
              ),
            ),
            const SizedBox(height: 6),
            for (final pessoa in entrada.value)
              Padding(
                padding: const EdgeInsets.only(bottom: 4),
                child: Column(
                  children: [
                    Text(
                      pessoa.nome,
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.w600,
                        color: Cores.tinta,
                      ),
                    ),
                    Text(
                      pessoa.papel,
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        fontSize: 12.5,
                        color: Cores.tintaFraca,
                      ),
                    ),
                  ],
                ),
              ),
            const SizedBox(height: 16),
          ],
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------

class _Instituicoes extends StatelessWidget {
  const _Instituicoes();

  static const _logos = [
    ('assets/logos/logo_ufam.png', 58.0),
    ('assets/logos/logo-icomp.png', 92.0),
    ('assets/logos/semsa.png', 84.0),
  ];

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 12),
      decoration: BoxDecoration(
        color: Cores.superficie,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Cores.linha),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          for (final (caminho, largura) in _logos)
            Flexible(
              child: Image.asset(
                caminho,
                width: largura,
                fit: BoxFit.contain,
                excludeFromSemantics: true,
                errorBuilder: (_, __, ___) => const SizedBox.shrink(),
              ),
            ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------

class _Rodape extends StatelessWidget {
  const _Rodape();

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(top: 22),
      child: Column(
        children: [
          const SizedBox(
            width: 28,
            height: 28,
            child: CircularProgressIndicator(strokeWidth: 2.6),
          ),
          const SizedBox(height: 14),
          const Text(
            'Preparando o modelo',
            style: TextStyle(fontSize: 13.5, color: Cores.tintaFraca),
          ),
          const SizedBox(height: 16),
          const AvisoEtico(compacto: true),
          const SizedBox(height: 12),
          Text(
            creditoFerramentas,
            textAlign: TextAlign.center,
            style: const TextStyle(
              fontSize: 11,
              color: Cores.tintaFraca,
              height: 1.4,
            ),
          ),
        ],
      ),
    );
  }
}
