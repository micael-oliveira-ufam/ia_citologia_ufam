import 'package:flutter/material.dart';

import '../data/bethesda.dart';
import '../data/projeto.dart';
import '../perfil.dart';
import '../servicos/modelo_risco.dart';
import '../tema.dart';
import '../widgets/comuns.dart';
import '../widgets/conteudo.dart';
import 'aba_modelo_risco.dart';

/// "Sobre o projeto". O paciente vê uma versão curta e sem jargão; o
/// profissional vê também o desempenho do modelo de citologia, o modelo de
/// risco com seus gráficos e as limitações de cada um.
class TelaSobre extends StatelessWidget {
  const TelaSobre({
    super.key,
    required this.perfil,
    required this.modeloRisco,
  });

  final Perfil perfil;
  final Future<ModeloRisco> modeloRisco;

  @override
  Widget build(BuildContext context) {
    final pro = perfil.ehProfissional;
    final titulos = pro
        ? const ['O projeto', 'Equipe', 'Desempenho', 'Modelo de risco', 'Limitações']
        : const ['O aplicativo', 'Equipe', 'Limites'];
    final paginas = pro
        ? <Widget>[
            const _AbaProjeto(),
            const _AbaEquipe(comBaseDeImagens: true),
            const _AbaDesempenho(),
            AbaModeloRisco(modelo: modeloRisco),
            const _AbaLimitacoes(),
          ]
        : const <Widget>[
            _AbaAplicativo(),
            _AbaEquipe(comBaseDeImagens: false),
            _AbaLimites(),
          ];

    return DefaultTabController(
      length: titulos.length,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Sobre o projeto'),
          bottom: TabBar(
            isScrollable: true,
            tabAlignment: TabAlignment.start,
            labelColor: Cores.teal,
            unselectedLabelColor: Cores.tintaFraca,
            indicatorColor: Cores.teal,
            tabs: [for (final t in titulos) Tab(text: t)],
          ),
        ),
        body: SafeArea(child: TabBarView(children: paginas)),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Profissional
// ---------------------------------------------------------------------------

class _AbaProjeto extends StatelessWidget {
  const _AbaProjeto();

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 32),
      children: [
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
            'Os modelos estão embutidos no aplicativo e rodam no processador do '
            'próprio aparelho. Nenhuma imagem nem resposta de questionário é '
            'enviada para servidor, não há coleta de dados de uso e o '
            'aplicativo funciona sem conexão.',
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
  const _AbaEquipe({required this.comBaseDeImagens});

  /// A base de imagens (IARC) só interessa a quem usa a ferramenta de citologia.
  final bool comBaseDeImagens;

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
        if (comBaseDeImagens)
          const CartaoSecao(
            titulo: 'Base de dados',
            icone: Icons.dataset_outlined,
            child: Text(
              'IARC Digital Atlas of Cervical Cytology — acervo público de '
              'imagens da Agência Internacional de Pesquisa em Câncer, '
              'vinculada à Organização Mundial da Saúde. Foi a única fonte de '
              'imagens usada no treinamento e na validação.',
              style: TextStyle(
                fontSize: 15,
                height: 1.55,
                color: Cores.tintaFraca,
              ),
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
          '${Metricas.acertos} acertos em ${Metricas.imagens} imagens do IARC '
          'Digital Atlas.',
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
                  'mede quantos casos graves escapariam. Nesta partição, nenhum.',
                  style: TextStyle(fontSize: 14, color: Cores.tintaFraca, height: 1.5),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 14),
        const CartaoSecao(
          titulo: 'O único erro',
          icone: Icons.error_outline,
          child: Text(
            'Um campo normal foi classificado como carcinoma. É um falso '
            'positivo: gera revisão desnecessária, e não um caso perdido. A '
            'assimetria entre os dois tipos de erro é o que importa numa '
            'ferramenta de triagem.',
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


// ---------------------------------------------------------------------------
// Paciente
// ---------------------------------------------------------------------------

class _AbaAplicativo extends StatelessWidget {
  const _AbaAplicativo();

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 32),
      children: [
        const CartaoInfo(
          icone: Icons.favorite_outline,
          titulo: 'Para que serve',
          texto: 'Reunir informação sobre o câncer do colo do útero (o que é, '
              'como prevenir e o que significam os resultados dos exames) e '
              'permitir uma estimativa educativa de perfil de risco. É um '
              'projeto acadêmico da Universidade Federal do Amazonas.',
        ),
        const CartaoInfo(
          icone: Icons.lock_outline,
          titulo: 'Privacidade',
          texto: 'As respostas do questionário ficam só no aparelho. O '
              'aplicativo não pede nome nem outros dados pessoais e não envia o '
              'que você digita. A internet só é usada para baixar o modelo de '
              'citologia (modo profissional) e para o catálogo de lâminas.',
        ),
        const CartaoInfo(
          icone: Icons.memory_outlined,
          titulo: 'Como a estimativa é feita',
          texto: 'Um modelo de inteligência artificial aprendeu, em dados '
              'públicos de 858 pacientes de um hospital de Caracas '
              '(Venezuela), quais respostas aparecem mais entre as mulheres '
              'com biópsia positiva. Ele compara o seu perfil com esses '
              'padrões e indica uma faixa de risco relativo. A capacidade do '
              'modelo de separar mulheres com e sem a doença é fraca: ele '
              'serve para conscientizar, não para diagnosticar.',
        ),
        Builder(
          builder: (context) => Align(
            alignment: Alignment.centerLeft,
            child: TextButton.icon(
              onPressed: () =>
                  abrirLink(context, 'https://archive.ics.uci.edu/dataset/383'),
              icon: const Icon(Icons.open_in_new, size: 16),
              label: const Text('Conjunto de dados usado (UCI)'),
            ),
          ),
        ),
      ],
    );
  }
}

class _AbaLimites extends StatelessWidget {
  const _AbaLimites();

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 32),
      children: const [
        AvisoEducativo(
          'Este aplicativo não faz diagnóstico, não emite laudo e não '
          'substitui consulta nem exame.',
        ),
        SizedBox(height: 16),
        CartaoInfo(
          icone: Icons.groups_outlined,
          titulo: 'Poucos casos, um só hospital',
          texto: 'O modelo aprendeu com apenas 55 mulheres com biópsia '
              'positiva, todas de um hospital. O resultado pode não valer para '
              'outras populações, como as de comunidades da Amazônia.',
        ),
        CartaoInfo(
          icone: Icons.ssid_chart,
          titulo: 'Resposta nem sempre previsível',
          texto: 'Com tão poucos casos, o modelo pode reagir de forma '
              'irregular a combinações raras de respostas: um fator de risco a '
              'mais nem sempre eleva a faixa. Ele reflete padrões desse '
              'conjunto de dados, não uma regra médica.',
        ),
        CartaoInfo(
          icone: Icons.event_available_outlined,
          titulo: 'O exame preventivo é o que protege',
          texto: 'Nenhuma estimativa, de risco menor ou maior, substitui o '
              'exame preventivo nem a avaliação de um profissional de saúde.',
        ),
      ],
    );
  }
}
