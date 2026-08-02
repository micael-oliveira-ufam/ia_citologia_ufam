import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/semantics.dart';

import '../data/bethesda.dart';
import '../servicos/classificador.dart';
import '../tema.dart';
import '../widgets/comuns.dart';

class TelaResultado extends StatefulWidget {
  const TelaResultado({
    super.key,
    required this.classificador,
    required this.bytes,
    required this.origem,
    required this.verificarRobustez,
    this.referencia,
  });

  final Classificador classificador;
  final Uint8List bytes;
  final String origem;
  final bool verificarRobustez;

  /// Classe declarada no IARC Digital Atlas, quando o campo é um exemplo.
  final String? referencia;

  @override
  State<TelaResultado> createState() => _TelaResultadoState();
}

class _TelaResultadoState extends State<TelaResultado> {
  Analise? _analise;
  String? _erro;
  bool _mostrarMapa = true;

  @override
  void initState() {
    super.initState();
    _analisar();
  }

  Future<void> _analisar() async {
    try {
      final resultado = await widget.classificador.analisar(
        widget.bytes,
        verificarRobustez: widget.verificarRobustez,
      );
      if (!mounted) return;
      setState(() => _analise = resultado);

      // Leitores de tela não percebem a troca de conteúdo sozinhos.
      final c = categoriaDe(resultado.sigla);
      SemanticsService.announce(
        'Análise concluída. Sugestão do modelo: ${c.sigla}, ${c.nome}. '
        'Confiança de ${(resultado.confianca * 100).toStringAsFixed(0)} por cento.',
        Directionality.of(context),
      );
    } catch (e) {
      if (!mounted) return;
      setState(() => _erro = e.toString());
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(widget.origem, overflow: TextOverflow.ellipsis)),
      body: SafeArea(
        child: _erro != null
            ? _Falha(mensagem: _erro!, aoTentar: () {
                setState(() => _erro = null);
                _analisar();
              })
            : _analise == null
                ? const _Carregando()
                : _Resultado(
                    analise: _analise!,
                    referencia: widget.referencia,
                    mostrarMapa: _mostrarMapa,
                    aoAlternarMapa: (v) => setState(() => _mostrarMapa = v),
                  ),
      ),
    );
  }
}

class _Carregando extends StatelessWidget {
  const _Carregando();

  @override
  Widget build(BuildContext context) {
    return Semantics(
      label: 'Analisando o campo. Aguarde.',
      liveRegion: true,
      child: const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            SizedBox(
              width: 44,
              height: 44,
              child: CircularProgressIndicator(strokeWidth: 3),
            ),
            SizedBox(height: 20),
            Text(
              'Analisando o campo…',
              style: TextStyle(fontSize: 16, color: Cores.tintaFraca),
            ),
          ],
        ),
      ),
    );
  }
}

class _Falha extends StatelessWidget {
  const _Falha({required this.mensagem, required this.aoTentar});

  final String mensagem;
  final VoidCallback aoTentar;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.error_outline, size: 44, color: Cores.tintaFraca),
          const SizedBox(height: 16),
          const Text(
            'A análise não foi concluída',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 8),
          Text(
            mensagem,
            textAlign: TextAlign.center,
            style: const TextStyle(fontSize: 14, color: Cores.tintaFraca, height: 1.5),
          ),
          const SizedBox(height: 24),
          FilledButton(onPressed: aoTentar, child: const Text('Tentar de novo')),
        ],
      ),
    );
  }
}

class _Resultado extends StatelessWidget {
  const _Resultado({
    required this.analise,
    required this.referencia,
    required this.mostrarMapa,
    required this.aoAlternarMapa,
  });

  final Analise analise;
  final String? referencia;
  final bool mostrarMapa;
  final ValueChanged<bool> aoAlternarMapa;

  @override
  Widget build(BuildContext context) {
    final c = categoriaDe(analise.sigla);

    return ListView(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 32),
      children: [
        // --- Veredito ---------------------------------------------------
        Semantics(
          label: 'Sugestão do modelo: ${c.sigla}, ${c.nome}. '
              'Confiança de ${(analise.confianca * 100).toStringAsFixed(1)} por cento.',
          excludeSemantics: true,
          child: Card(
            child: Container(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(16),
                border: Border(left: BorderSide(color: c.cor, width: 5)),
              ),
              padding: const EdgeInsets.all(18),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'SUGESTÃO DO MODELO',
                    style: TextStyle(
                      fontSize: 12,
                      letterSpacing: 1.2,
                      fontWeight: FontWeight.w700,
                      color: Cores.tintaFraca,
                    ),
                  ),
                  const SizedBox(height: 10),
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Icon(c.icone, color: c.cor, size: 30),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              c.sigla,
                              style: TextStyle(
                                fontSize: 26,
                                fontWeight: FontWeight.w800,
                                color: c.cor,
                              ),
                            ),
                            const SizedBox(height: 2),
                            Text(
                              c.nome,
                              style: const TextStyle(
                                fontSize: 15,
                                color: Cores.tinta,
                                height: 1.35,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 14),
                  Text(
                    'Confiança: ${(analise.confianca * 100).toStringAsFixed(1)}%'
                    '${analise.concordancia != null ? '  ·  Concordância entre orientações: ${(analise.concordancia! * 100).toStringAsFixed(0)}%' : ''}',
                    style: const TextStyle(fontSize: 14.5, color: Cores.tintaFraca),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    c.explicacao,
                    style: const TextStyle(
                      fontSize: 14.5,
                      color: Cores.tintaFraca,
                      height: 1.5,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),

        // --- Alertas de instabilidade -----------------------------------
        if (analise.instavel) ...[
          const SizedBox(height: 12),
          const _Alerta(
            texto: 'Predição instável: a rede muda de opinião quando o campo é '
                'girado ou espelhado. Trate este resultado como indefinido.',
          ),
        ] else if (analise.siglaRobustez != null &&
            analise.siglaRobustez != analise.sigla) ...[
          const SizedBox(height: 12),
          _Alerta(
            texto: 'A média das oito orientações aponta ${analise.siglaRobustez}, '
                'diferente da leitura direta. Vale repetir com outra imagem do '
                'mesmo campo.',
          ),
        ],

        if (referencia != null) ...[
          const SizedBox(height: 12),
          _Conferencia(
            acertou: referencia == analise.sigla,
            referencia: referencia!,
            predita: analise.sigla,
          ),
        ],

        const SizedBox(height: 20),
        ReguaBethesda(ativa: analise.sigla),

        // --- Imagem e mapa de calor -------------------------------------
        const SizedBox(height: 24),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    const Expanded(
                      child: Text(
                        'Onde a rede olhou',
                        style: TextStyle(fontSize: 17, fontWeight: FontWeight.w600),
                      ),
                    ),
                    Semantics(
                      label: 'Mostrar mapa de calor sobre o campo',
                      child: Switch.adaptive(
                        value: mostrarMapa,
                        onChanged: aoAlternarMapa,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: AspectRatio(
                    aspectRatio: 1,
                    child: Image.memory(
                      mostrarMapa ? analise.pngSobreposto : analise.pngOriginal,
                      fit: BoxFit.cover,
                      gaplessPlayback: true,
                      semanticLabel: mostrarMapa
                          ? 'Campo analisado com mapa de calor sobreposto. '
                              'Regiões claras foram as mais influentes na decisão.'
                          : 'Campo analisado, sem sobreposição.',
                    ),
                  ),
                ),
                const SizedBox(height: 12),
                const Text(
                  'Regiões claras marcam maior influência na classe escolhida. Se o '
                  'destaque cair em fundo, muco ou artefato de preparo, desconfie '
                  'da predição.',
                  style: TextStyle(fontSize: 14, color: Cores.tintaFraca, height: 1.5),
                ),
              ],
            ),
          ),
        ),

        // --- Probabilidades ---------------------------------------------
        const SizedBox(height: 16),
        CartaoSecao(
          titulo: 'Confiança por categoria',
          icone: Icons.bar_chart_outlined,
          child: BarrasDeProbabilidade(
            probabilidades: {
              for (final s in ordemDoModelo) s: analise.probabilidadeDe(s),
            },
            destaque: analise.sigla,
          ),
        ),

        // --- Robustez ----------------------------------------------------
        if (analise.probabilidadesRobustez != null) ...[
          const SizedBox(height: 16),
          CartaoSecao(
            titulo: 'Verificação de robustez',
            icone: Icons.rotate_90_degrees_ccw_outlined,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'O campo foi analisado nas oito orientações possíveis (giros e '
                  'rotações de 90°). Como o treinamento usou essas mesmas '
                  'transformações, a predição não deveria mudar entre elas.',
                  style: TextStyle(fontSize: 14, color: Cores.tintaFraca, height: 1.5),
                ),
                const SizedBox(height: 16),
                for (var i = 0; i < ordemDoModelo.length; i++)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 8),
                    child: Row(
                      children: [
                        SizedBox(
                          width: 60,
                          child: Text(
                            ordemDoModelo[i],
                            style: const TextStyle(fontWeight: FontWeight.w600),
                          ),
                        ),
                        Expanded(
                          child: Text(
                            'direta ${(analise.probabilidades[i] * 100).toStringAsFixed(1)}%'
                            '   ·   média ${(analise.probabilidadesRobustez![i] * 100).toStringAsFixed(1)}%',
                            style: const TextStyle(
                              fontSize: 14,
                              color: Cores.tintaFraca,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
              ],
            ),
          ),
        ],

        const SizedBox(height: 16),
        Text(
          'Processado no aparelho em ${(analise.milissegundos / 1000).toStringAsFixed(1)} s, '
          'sem conexão com servidor.',
          style: const TextStyle(fontSize: 13, color: Cores.tintaFraca),
        ),
        const SizedBox(height: 16),
        const AvisoEtico(compacto: true),
      ],
    );
  }
}

class _Alerta extends StatelessWidget {
  const _Alerta({required this.texto});

  final String texto;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFFFDECEA),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFFF3C6C0)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.priority_high, size: 20, color: Color(0xFF8C2F22)),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              texto,
              style: const TextStyle(
                fontSize: 14,
                color: Color(0xFF6B241A),
                height: 1.45,
              ),
            ),
          ),
        ],
      ),
    );
  }
}


/// Comparação com o gabarito do atlas, exibida só nos campos de exemplo.
class _Conferencia extends StatelessWidget {
  const _Conferencia({
    required this.acertou,
    required this.referencia,
    required this.predita,
  });

  final bool acertou;
  final String referencia;
  final String predita;

  @override
  Widget build(BuildContext context) {
    final cor = acertou ? const Color(0xFF2E7D62) : const Color(0xFF8C2F22);
    final fundo = acertou ? const Color(0xFFEAF5F0) : const Color(0xFFFDECEA);
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: fundo,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: cor.withValues(alpha: 0.3)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(acertou ? Icons.check_circle_outline : Icons.close, size: 20, color: cor),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              acertou
                  ? 'A predição coincide com a referência do atlas ($referencia).'
                  : 'A predição ($predita) diverge da referência do atlas '
                      '($referencia). Divergências assim são o material mais útil '
                      'para melhorar o modelo.',
              style: TextStyle(fontSize: 14, color: cor, height: 1.45),
            ),
          ),
        ],
      ),
    );
  }
}
