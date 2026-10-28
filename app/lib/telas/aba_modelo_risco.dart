import 'package:flutter/material.dart';

import '../servicos/modelo_risco.dart';
import '../tema.dart';
import '../widgets/comuns.dart';
import '../widgets/conteudo.dart';

/// Aba do profissional em "Sobre o projeto": como o modelo de risco foi
/// treinado, o que ele consegue (e não consegue) fazer, e os gráficos que
/// comparam os algoritmos testados.
class AbaModeloRisco extends StatelessWidget {
  const AbaModeloRisco({super.key, required this.modelo});

  final Future<ModeloRisco> modelo;

  @override
  Widget build(BuildContext context) {
    final estilo = Theme.of(context).textTheme;
    return FutureBuilder<ModeloRisco>(
      future: modelo,
      builder: (context, resposta) {
        final m = resposta.data;
        final perm = m?.permutacao ?? const <String, dynamic>{};
        final auc = m?.aucForaDaAmostra;
        final nuloMedia = perm['null_mean'];
        final nuloDesvio = perm['null_std'];
        final pValor = perm['p_value'];
        final sufixoP =
            pValor is num ? '; p ≈ ${numero(pValor.toDouble())}' : '';
        return ListView(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 32),
          children: [
            Text('Modelo de estimativa de risco', style: estilo.headlineSmall),
            const SizedBox(height: 14),
            const CartaoSecao(
              titulo: 'Os dados',
              icone: Icons.dataset_outlined,
              child: Text(
                'Conjunto público “Cervical Cancer (Risk Factors)”, do '
                'repositório UCI, com 858 pacientes do Hospital Universitário '
                'de Caracas (Venezuela). Só 55 (6,4%) tiveram biópsia '
                'positiva, que é o desfecho estimado. Foram usadas apenas '
                'variáveis respondíveis em questionário; resultados de exames '
                '(Hinselmann, Schiller, citologia) ficaram de fora para não '
                '“colar” a resposta. O conjunto não tem renda, escolaridade '
                'nem local de moradia.',
                style: TextStyle(
                  fontSize: 15,
                  height: 1.55,
                  color: Cores.tintaFraca,
                ),
              ),
            ),
            const SizedBox(height: 14),
            const CartaoSecao(
              titulo: 'Os algoritmos',
              icone: Icons.compare_arrows,
              child: Text(
                'Nove algoritmos foram comparados: regressão logística, árvore '
                'de decisão, Random Forest, Extra Trees, Gradient Boosting, '
                'SVM, KNN, rede neural (MLP) e Naive Bayes, mais um modelo de '
                'referência que nunca prevê risco. Todos usaram imputação por '
                'mediana, padronização, ponderação das classes e validação '
                'cruzada estratificada 5×5.',
                style: TextStyle(
                  fontSize: 15,
                  height: 1.55,
                  color: Cores.tintaFraca,
                ),
              ),
            ),
            if (m != null) ...[
              const SizedBox(height: 14),
              CartaoSecao(
                titulo: 'Modelo usado no aplicativo: ${m.nome}',
                icone: Icons.verified_outlined,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    if (auc != null)
                      Text(
                        'AUC fora da amostra: ${numero(auc)} '
                        '(0,50 = acaso; 1,00 = perfeito).',
                        style: estilo.bodyMedium,
                      ),
                    if (nuloMedia is num && nuloDesvio is num)
                      Text(
                        'Com rótulos embaralhados, o mesmo método dá '
                        '${numero(nuloMedia.toDouble())} ± '
                        '${numero(nuloDesvio.toDouble())}$sufixoP.',
                        style: estilo.bodyMedium,
                      ),
                    const SizedBox(height: 6),
                    Text(
                      'Há sinal, mas fraco: serve para ilustrar fatores de '
                      'risco e iniciar a conversa, não para decidir condutas.',
                      style: estilo.bodyMedium,
                    ),
                  ],
                ),
              ),
            ],
            const TituloSecao('Gráficos de desempenho'),
            const _Grafico(
              'assets/graficos/01_acuracia.png',
              'Acurácia por algoritmo',
              'Um modelo que sempre diz “sem risco” já acerta cerca de 94%. '
                  'Por isso a acurácia sozinha engana.',
            ),
            const _Grafico(
              'assets/graficos/02_acuracia_balanceada_recall.png',
              'Acurácia, acurácia balanceada e sensibilidade',
              'A acurácia balanceada e a sensibilidade mostram o quanto cada '
                  'algoritmo realmente encontra os casos positivos.',
            ),
            const _Grafico(
              'assets/graficos/03_roc_auc.png',
              'ROC-AUC por algoritmo',
              'Capacidade de ordenar mulheres com e sem biópsia positiva, '
                  'medida em previsões fora da amostra.',
            ),
            const _Grafico(
              'assets/graficos/04_curvas_roc.png',
              'Curvas ROC',
              'Quanto mais acima da diagonal, melhor. Aqui todas ficam perto '
                  'dela: o sinal nesses dados é modesto.',
            ),
            const _Grafico(
              'assets/graficos/06_importancia.png',
              'Variáveis mais influentes',
              'Importância relativa de cada pergunta no modelo usado.',
            ),
            const TituloSecao('Limitações'),
            const CartaoInfo(
              icone: Icons.groups_outlined,
              titulo: 'Amostra pequena e de um só hospital',
              texto: 'Só 55 casos positivos. Os resultados podem não se repetir '
                  'em outras populações, como a de comunidades amazônicas.',
            ),
            const CartaoInfo(
              icone: Icons.ssid_chart,
              titulo: 'Resposta nem sempre monótona',
              texto: 'Com tão poucos casos positivos, o modelo pode reagir de '
                  'forma irregular a combinações raras de respostas: um fator '
                  'de risco a mais nem sempre eleva a faixa. Ele reflete '
                  'padrões desse conjunto de dados, não uma regra clínica.',
            ),
            const CartaoInfo(
              icone: Icons.rule,
              titulo: 'Não substitui rastreamento',
              texto: 'Nenhum questionário substitui o exame preventivo nem a '
                  'avaliação de um profissional.',
            ),
            const TituloSecao('Referência'),
            const Text(
              'Fernandes, K.; Cardoso, J. S.; Fernandes, J. Transfer Learning '
              'with Partial Observability Applied to Cervical Cancer Screening. '
              'Iberian Conference on Pattern Recognition and Image Analysis, '
              '2017. Dados: UCI Machine Learning Repository.',
              style: TextStyle(fontSize: 14, color: Cores.tintaFraca, height: 1.5),
            ),
            const SizedBox(height: 8),
            Align(
              alignment: Alignment.centerLeft,
              child: TextButton.icon(
                onPressed: () =>
                    abrirLink(context, 'https://archive.ics.uci.edu/dataset/383'),
                icon: const Icon(Icons.open_in_new, size: 16),
                label: const Text('Conjunto de dados (UCI)'),
              ),
            ),
          ],
        );
      },
    );
  }
}

class _Grafico extends StatelessWidget {
  const _Grafico(this.arquivo, this.titulo, this.legenda);

  final String arquivo;
  final String titulo;
  final String legenda;

  @override
  Widget build(BuildContext context) {
    final estilo = Theme.of(context).textTheme;
    return Padding(
      padding: const EdgeInsets.only(bottom: 18),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(titulo, style: estilo.titleMedium),
          const SizedBox(height: 6),
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: Image.asset(
              arquivo,
              fit: BoxFit.fitWidth,
              semanticLabel: '$titulo. $legenda',
            ),
          ),
          const SizedBox(height: 6),
          Text(legenda, style: estilo.bodySmall),
        ],
      ),
    );
  }
}
