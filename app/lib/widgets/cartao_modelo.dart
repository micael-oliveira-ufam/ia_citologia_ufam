import 'package:flutter/material.dart';

import '../servicos/gerenciador_modelo.dart';
import '../tema.dart';

/// Estado do modelo de citologia no aparelho: consulta, download com progresso,
/// atualização e remoção. A análise só é liberada com o modelo conferido.
class CartaoModelo extends StatelessWidget {
  const CartaoModelo({super.key, required this.gerenciador});

  final GerenciadorModelo gerenciador;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: gerenciador,
      builder: (context, _) {
        return Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Row(
                  children: [
                    Icon(Icons.memory_outlined, color: Cores.teal),
                    SizedBox(width: 10),
                    Text(
                      'Modelo de análise',
                      style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
                    ),
                  ],
                ),
                const SizedBox(height: 10),
                ..._conteudo(context),
              ],
            ),
          ),
        );
      },
    );
  }

  List<Widget> _conteudo(BuildContext context) {
    final g = gerenciador;
    const estiloTexto = TextStyle(
      fontSize: 14.5,
      color: Cores.tintaFraca,
      height: 1.45,
    );

    switch (g.fase) {
      case FaseModelo.consultando:
        return const [
          Row(
            children: [
              SizedBox(
                width: 18,
                height: 18,
                child: CircularProgressIndicator(strokeWidth: 2.5),
              ),
              SizedBox(width: 12),
              Expanded(
                child: Text('Consultando o modelo publicado…', style: estiloTexto),
              ),
            ],
          ),
        ];

      case FaseModelo.naoPublicado:
        return [
          const Text(
            'O modelo de citologia ainda não foi publicado no repositório do '
            'projeto. Quando for, ele poderá ser baixado aqui.',
            style: estiloTexto,
          ),
          const SizedBox(height: 10),
          OutlinedButton.icon(
            onPressed: g.consultar,
            icon: const Icon(Icons.refresh),
            label: const Text('Consultar de novo'),
          ),
        ];

      case FaseModelo.semRede:
      case FaseModelo.erro:
        return [
          Text(
            g.mensagem ?? 'Não foi possível obter o modelo.',
            style: const TextStyle(
              fontSize: 14.5,
              color: Color(0xFF8A2E2E),
              height: 1.45,
            ),
          ),
          const SizedBox(height: 10),
          OutlinedButton.icon(
            onPressed: g.publicado != null ? g.baixar : g.consultar,
            icon: const Icon(Icons.refresh),
            label: Text(g.publicado != null ? 'Baixar de novo' : 'Tentar de novo'),
          ),
        ];

      case FaseModelo.naoInstalado:
        final m = g.publicado!;
        return [
          Text(
            'O modelo (versão ${m.versao}, ${m.tamanhoLegivel}) ainda não está '
            'neste aparelho. O download acontece uma única vez; prefira o Wi-Fi. '
            'Depois disso, a análise funciona sem internet e a imagem nunca '
            'sai do aparelho.',
            style: estiloTexto,
          ),
          const SizedBox(height: 12),
          FilledButton.icon(
            onPressed: g.baixar,
            icon: const Icon(Icons.download_outlined),
            label: Text('Baixar o modelo (${m.tamanhoLegivel})'),
          ),
        ];

      case FaseModelo.baixando:
        final pct = (g.progresso * 100).clamp(0, 100).toStringAsFixed(0);
        final total = g.publicado?.tamanhoLegivel ?? '';
        return [
          Semantics(
            label: 'Baixando o modelo, $pct por cento',
            liveRegion: true,
            child: LinearProgressIndicator(
              value: g.progresso <= 0 ? null : g.progresso,
              minHeight: 8,
              borderRadius: BorderRadius.circular(4),
            ),
          ),
          const SizedBox(height: 8),
          Text('Baixando… $pct% de $total', style: estiloTexto),
          const SizedBox(height: 4),
          Align(
            alignment: Alignment.centerLeft,
            child: TextButton(
              onPressed: g.cancelar,
              child: const Text('Cancelar'),
            ),
          ),
        ];

      case FaseModelo.pronto:
        final m = g.instalado!;
        final detalhes = [
          'versão ${m.versao}',
          if (m.arquitetura != null) m.arquitetura!,
          if (m.quantizacao != null) m.quantizacao!,
          m.tamanhoLegivel,
        ].join(' · ');
        return [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Icon(Icons.check_circle_outline, color: Color(0xFF2E7D62)),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  'Pronto no aparelho ($detalhes). Conferido por SHA-256.',
                  style: estiloTexto,
                ),
              ),
            ],
          ),
          if (g.atualizacaoDisponivel) ...[
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Cores.alertaFundo,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: const Color(0xFFF0D6A8)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Há uma versão nova do modelo (${g.publicado!.versao}, '
                    '${g.publicado!.tamanhoLegivel}).',
                    style: const TextStyle(
                      fontSize: 14.5,
                      color: Color(0xFF5A4415),
                      height: 1.4,
                    ),
                  ),
                  const SizedBox(height: 8),
                  FilledButton.tonalIcon(
                    onPressed: g.baixar,
                    icon: const Icon(Icons.system_update_alt),
                    label: const Text('Atualizar o modelo'),
                  ),
                ],
              ),
            ),
          ],
          const SizedBox(height: 6),
          Align(
            alignment: Alignment.centerLeft,
            child: TextButton.icon(
              onPressed: () => _confirmarRemocao(context),
              icon: const Icon(Icons.delete_outline),
              label: const Text('Remover do aparelho'),
            ),
          ),
        ];
    }
  }

  Future<void> _confirmarRemocao(BuildContext context) async {
    final confirmou = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Remover o modelo?'),
        content: const Text(
          'O arquivo será apagado deste aparelho. Para analisar de novo, será '
          'preciso baixá-lo outra vez.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: const Text('Cancelar'),
          ),
          FilledButton(
            style: FilledButton.styleFrom(minimumSize: const Size(0, 44)),
            onPressed: () => Navigator.of(ctx).pop(true),
            child: const Text('Remover'),
          ),
        ],
      ),
    );
    if (confirmou == true) await gerenciador.remover();
  }
}
