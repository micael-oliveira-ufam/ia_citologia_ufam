import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

import '../tema.dart';

/// Título de seção dentro de uma aba rolável.
class TituloSecao extends StatelessWidget {
  const TituloSecao(this.texto, {super.key});

  final String texto;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(0, 24, 0, 10),
      child: Semantics(
        header: true,
        child: Text(
          texto,
          style: Theme.of(context).textTheme.titleLarge?.copyWith(
                fontWeight: FontWeight.w700,
                color: Cores.tinta,
              ),
        ),
      ),
    );
  }
}

/// Cartão com ícone, título e texto corrido.
class CartaoInfo extends StatelessWidget {
  const CartaoInfo({
    super.key,
    required this.icone,
    required this.titulo,
    required this.texto,
    this.cor = Cores.teal,
  });

  final IconData icone;
  final String titulo;
  final String texto;
  final Color cor;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(icone, color: cor, size: 26),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      titulo,
                      style: Theme.of(context).textTheme.titleMedium,
                    ),
                    const SizedBox(height: 4),
                    Text(
                      texto,
                      style: Theme.of(context).textTheme.bodyMedium,
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// Aviso de que o conteúdo é educativo. Em tom mais calmo que o [AvisoEtico],
/// que fica reservado à ferramenta de análise de lâminas.
class AvisoEducativo extends StatelessWidget {
  const AvisoEducativo(this.texto, {super.key});

  final String texto;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      label: 'Aviso. $texto',
      excludeSemantics: true,
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.fromLTRB(14, 12, 14, 12),
        decoration: BoxDecoration(
          color: Cores.tealClaro,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Cores.linha),
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Icon(Icons.info_outline, size: 20, color: Cores.teal),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                texto,
                style: const TextStyle(
                  color: Cores.tinta,
                  fontSize: 14,
                  height: 1.45,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

Future<void> abrirLink(BuildContext context, String url) async {
  final aviso = ScaffoldMessenger.of(context);
  const falha = SnackBar(content: Text('Não foi possível abrir o link.'));
  try {
    final abriu = await launchUrl(
      Uri.parse(url),
      mode: LaunchMode.externalApplication,
    );
    if (!abriu) aviso.showSnackBar(falha);
  } catch (_) {
    aviso.showSnackBar(falha);
  }
}

/// 0,1234 -> "12,3%".
String percentual(double fracao, {int casas = 1}) =>
    '${(fracao * 100).toStringAsFixed(casas).replaceAll('.', ',')}%';

/// 0.4123 -> "0,41".
String numero(double valor, {int casas = 2}) =>
    valor.toStringAsFixed(casas).replaceAll('.', ',');
