import 'package:flutter/material.dart';

import '../data/bethesda.dart';
import '../data/projeto.dart';
import '../tema.dart';
import '../widgets/comuns.dart';

class TelaSobre extends StatelessWidget {
  const TelaSobre({super.key});

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 4,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Sobre o projeto'),
          bottom: const TabBar(
            isScrollable: true,
            tabAlignment: TabAlignment.start,
            labelColor: Cores.teal,
            unselectedLabelColor: Cores.tintaFraca,
            indicatorColor: Cores.teal,
            tabs: [
              Tab(text: 'O projeto'),
              Tab(text: 'Equipe'),
              Tab(text: 'Desempenho'),
              Tab(text: 'Limitações'),
            ],
          ),
        ),
        body: const SafeArea(
          child: TabBarView(
            children: [
              _AbaProjeto(),
              _AbaEquipe(),
              _AbaDesempenho(),
              _AbaLimitacoes(),
            ],
          ),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------

class _AbaProjeto extends StatelessWidget {
  const _AbaProjeto();

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 32),
      children: [
        const CartaoSecao(
          titulo: 'Citologia em meio líquido',
          icone: Icons.biotech_outlined,
          child: Text(
            textoSobreCitopatologia,
            style: TextStyle(fontSize: 15, height: 1.55, color: Cores.tintaFraca),
          ),
        ),
        const SizedBox(height: 14),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(18),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'As quatro categorias',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 14),
                for (final c in categoriasPorGravidade) ...[
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Icon(c.icone, color: c.cor, size: 22),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              '${c.sigla}. ${c.nome}',
                              style: TextStyle(
                                fontWeight: FontWeight.w700,
                                fontSize: 15,
                                color: c.cor,
                              ),
                            ),
                            const SizedBox(height: 4),
                            Text(
                              c.explicacao,
                              style: const TextStyle(
                                fontSize: 14.5,
                                height: 1.5,
                                color: Cores.tintaFraca,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                  if (c != categoriasPorGravidade.last)
                    const Padding(
                      padding: EdgeInsets.symmetric(vertical: 14),
                      child: Divider(height: 1),
                    ),
                ],
              ],
            ),
          ),
        ),
        const SizedBox(height: 14),
        const CartaoSecao(
          titulo: 'Como a inteligência artificial funciona',
          icone: Icons.memory_outlined,
          child: Text(
            textoSobreIA,
            style: TextStyle(fontSize: 15, height: 1.55, color: Cores.tintaFraca),
          ),
        ),
        const SizedBox(height: 14),
        const CartaoSecao(
          titulo: 'O mapa de calor',
          icone: Icons.blur_on_outlined,
          child: Text(
            textoSobreMapaDeCalor,
            style: TextStyle(fontSize: 15, height: 1.55, color: Cores.tintaFraca),
          ),
        ),
        const SizedBox(height: 14),
        const CartaoSecao(
          titulo: 'Privacidade',
          icone: Icons.lock_outline,
          child: Text(
            'O modelo está embutido no aplicativo e roda no processador do '
            'próprio aparelho. Nenhuma imagem é enviada para servidor, não há '
            'coleta de dados de uso e o aplicativo funciona sem conexão.',
            style: TextStyle(fontSize: 15, height: 1.55, color: Cores.tintaFraca),
          ),
        ),
        const SizedBox(height: 20),
        const Center(
          child: Text(
            versaoModelo,
            style: TextStyle(fontSize: 13, color: Cores.tintaFraca),
          ),
        ),
      ],
    );
  }
}

// ---------------------------------------------------------------------------

class _AbaEquipe extends StatelessWidget {
  const _AbaEquipe();

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 32),
      children: [
        Text('Quem faz', style: Theme.of(context).textTheme.headlineSmall),
        const SizedBox(height: 14),
        Card(
          child: Column(
            children: [
              for (var i = 0; i < equipe.length; i++) ...[
                Padding(
                  padding: const EdgeInsets.all(18),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      CircleAvatar(
                        radius: 22,
                        backgroundColor: Cores.tealClaro,
                        child: Text(
                          _iniciais(equipe[i].nome),
                          style: const TextStyle(
                            color: Cores.teal,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ),
                      const SizedBox(width: 14),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              equipe[i].nome,
                              style: const TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                            const SizedBox(height: 4),
                            Text(
                              equipe[i].papel,
                              style: const TextStyle(
                                fontSize: 14.5,
                                color: Cores.teal,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                            const SizedBox(height: 4),
                            Text(
                              equipe[i].vinculo,
                              style: const TextStyle(
                                fontSize: 14,
                                color: Cores.tintaFraca,
                                height: 1.4,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
                if (i < equipe.length - 1) const Divider(height: 1),
              ],
            ],
          ),
        ),
        const SizedBox(height: 24),
        Text(
          'Instituições',
          style: Theme.of(context).textTheme.headlineSmall,
        ),
        const SizedBox(height: 14),
        for (final inst in instituicoes) ...[
          Card(
            child: Padding(
              padding: const EdgeInsets.all(18),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                    decoration: BoxDecoration(
                      color: Cores.tealClaro,
                      borderRadius: BorderRadius.circular(6),
                    ),
                    child: Text(
                      inst.sigla,
                      style: const TextStyle(
                        color: Cores.teal,
                        fontWeight: FontWeight.w700,
                        fontSize: 13,
                        letterSpacing: 0.5,
                      ),
                    ),
                  ),
                  const SizedBox(height: 10),
                  Text(
                    inst.nome,
                    style: const TextStyle(fontSize: 15.5, fontWeight: FontWeight.w600),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    inst.papel,
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
          const SizedBox(height: 12),
        ],
        const SizedBox(height: 8),
        CartaoSecao(
          titulo: 'Conjuntos de dados',
          icone: Icons.dataset_outlined,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'O modelo foi treinado e validado sobre dois acervos públicos de '
                'citologia em meio líquido. Nenhuma imagem de paciente atendido '
                'pela rede municipal foi utilizada.',
                style: TextStyle(fontSize: 15, height: 1.55, color: Cores.tintaFraca),
              ),
              const SizedBox(height: 16),
              for (final ds in datasets) ...[
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: Cores.tealClaro,
                    borderRadius: BorderRadius.circular(6),
                  ),
                  child: Text(
                    ds.sigla,
                    style: const TextStyle(
                      color: Cores.teal,
                      fontWeight: FontWeight.w700,
                      fontSize: 12.5,
                    ),
                  ),
                ),
                const SizedBox(height: 6),
                Text(
                  ds.nome,
                  style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w600),
                ),
                const SizedBox(height: 4),
                SelectableText(
                  ds.url,
                  style: const TextStyle(fontSize: 13, color: Cores.teal),
                ),
                if (ds.referencia != null) ...[
                  const SizedBox(height: 8),
                  const Text(
                    'Como citar:',
                    style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
                  ),
                  const SizedBox(height: 2),
                  SelectableText(
                    ds.referencia!,
                    style: const TextStyle(
                      fontSize: 13,
                      height: 1.5,
                      color: Cores.tintaFraca,
                    ),
                  ),
                ],
                const SizedBox(height: 18),
              ],
            ],
          ),
        ),
        const SizedBox(height: 14),
        const CartaoSecao(
          titulo: 'Ferramentas de apoio ao desenvolvimento',
          icone: Icons.build_outlined,
          child: Text(
            creditoFerramentas,
            style: TextStyle(fontSize: 15, height: 1.55, color: Cores.tintaFraca),
          ),
        ),
      ],
    );
  }

  static String _iniciais(String nome) {
    final partes = nome
        .replaceAll(RegExp(r'Prof\.|Dr[a]?\.'), '')
        .trim()
        .split(RegExp(r'\s+'))
        .where((p) => p.length > 2)
        .toList();
    if (partes.isEmpty) return '?';
    if (partes.length == 1) return partes.first.substring(0, 1).toUpperCase();
    return (partes.first.substring(0, 1) + partes.last.substring(0, 1))
        .toUpperCase();
  }
}

// ---------------------------------------------------------------------------

class _AbaDesempenho extends StatelessWidget {
  const _AbaDesempenho();

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 32),
      children: [
        Text(
          'Na partição de validação',
          style: Theme.of(context).textTheme.headlineSmall,
        ),
        const SizedBox(height: 4),
        Text(
          '${Metricas.protocolo}. ${Metricas.acertos} acertos em '
          '${Metricas.imagens} imagens do IARC Digital Atlas.',
          style: const TextStyle(fontSize: 15, color: Cores.tintaFraca, height: 1.5),
        ),
        const SizedBox(height: 16),
        Row(
          children: [
            Expanded(child: _Numero(rotulo: 'Acurácia', valor: Metricas.acuracia)),
            const SizedBox(width: 12),
            Expanded(child: _Numero(rotulo: 'F1 macro', valor: Metricas.f1Macro)),
          ],
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: _Numero(rotulo: 'Precisão macro', valor: Metricas.precisaoMacro),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _Numero(rotulo: 'Recall macro', valor: Metricas.recallMacro),
            ),
          ],
        ),
        const SizedBox(height: 20),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(18),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Por categoria',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 14),
                for (final c in categoriasPorGravidade) ...[
                  _LinhaClasse(sigla: c.sigla, cor: c.cor),
                  if (c != categoriasPorGravidade.last)
                    const Padding(
                      padding: EdgeInsets.symmetric(vertical: 12),
                      child: Divider(height: 1),
                    ),
                ],
                const SizedBox(height: 16),
                const Text(
                  'Em rastreio, o recall de HSIL e SCC é o número que mais importa: '
                  'mede quantos casos graves escapariam sem sinalização alguma. '
                  'Nesta validação, nenhum foi lido como normal.',
                  style: TextStyle(fontSize: 14, color: Cores.tintaFraca, height: 1.5),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 14),
        const CartaoSecao(
          titulo: 'A direção dos erros',
          icone: Icons.compare_arrows_outlined,
          child: Text(
            'Numa ferramenta de triagem, o que importa não é o total de erros e '
            'sim para onde eles apontam.\n\n'
            'Nesta validação, nenhuma das 312 imagens de HSIL ou carcinoma foi '
            'classificada como normal. O erro mais comum foi o oposto, sete '
            'carcinomas lidos como HSIL, o que mantém a paciente na via de '
            'investigação. Seis campos de LSIL foram para NILM: é o achado que '
            'merece mais atenção, porque adia um seguimento.',
            style: TextStyle(fontSize: 15, height: 1.55, color: Cores.tintaFraca),
          ),
        ),
      ],
    );
  }
}

class _Numero extends StatelessWidget {
  const _Numero({required this.rotulo, required this.valor});

  final String rotulo;
  final double valor;

  @override
  Widget build(BuildContext context) {
    final texto = '${(valor * 100).toStringAsFixed(2)}%';
    return Semantics(
      label: '$rotulo: $texto',
      excludeSemantics: true,
      child: Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                rotulo.toUpperCase(),
                style: const TextStyle(
                  fontSize: 11.5,
                  letterSpacing: 0.9,
                  color: Cores.tintaFraca,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(height: 6),
              Text(
                texto,
                style: const TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.w700,
                  color: Cores.tinta,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _LinhaClasse extends StatelessWidget {
  const _LinhaClasse({required this.sigla, required this.cor});

  final String sigla;
  final Color cor;

  @override
  Widget build(BuildContext context) {
    final m = Metricas.porClasse[sigla];
    if (m == null) return const SizedBox.shrink();
    String pct(double v) => '${(v * 100).toStringAsFixed(1)}%';

    return Semantics(
      label: '$sigla. Precisão ${pct(m[0])}, recall ${pct(m[1])}, '
          'F1 ${pct(m[2])}, ${m[3].toInt()} imagens.',
      excludeSemantics: true,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(width: 10, height: 10, decoration: BoxDecoration(
                color: cor,
                shape: BoxShape.circle,
              )),
              const SizedBox(width: 8),
              Text(
                sigla,
                style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 15),
              ),
              const Spacer(),
              Text(
                '${m[3].toInt()} imagens',
                style: const TextStyle(fontSize: 13, color: Cores.tintaFraca),
              ),
            ],
          ),
          const SizedBox(height: 6),
          Text(
            'precisão ${pct(m[0])}   ·   recall ${pct(m[1])}   ·   F1 ${pct(m[2])}',
            style: const TextStyle(fontSize: 14, color: Cores.tintaFraca),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------

class _AbaLimitacoes extends StatelessWidget {
  const _AbaLimitacoes();

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 32),
      children: const [
        AvisoEtico(),
        SizedBox(height: 16),
        CartaoSecao(
          titulo: 'O que este aplicativo ainda não é',
          icone: Icons.gavel_outlined,
          child: Text(
            textoLimitacoes,
            style: TextStyle(fontSize: 15, height: 1.6, color: Cores.tintaFraca),
          ),
        ),
        SizedBox(height: 14),
        CartaoSecao(
          titulo: 'Próximos passos',
          icone: Icons.timeline_outlined,
          child: Text(
            '1. Submissão do protocolo ao CEP/UFAM.\n\n'
            '2. Validação externa com lâminas do Laboratório Sebastião Marinho, '
            'medindo desempenho por lâmina e por paciente.\n\n'
            '3. Concordância entre a IA e a dupla leitura humana.\n\n'
            '4. Estudo de impacto no tempo de fila e na taxa de detecção de '
            'HSIL ou pior.',
            style: TextStyle(fontSize: 15, height: 1.6, color: Cores.tintaFraca),
          ),
        ),
      ],
    );
  }
}
