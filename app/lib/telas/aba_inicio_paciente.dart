import 'package:flutter/material.dart';

import '../tema.dart';
import '../widgets/conteudo.dart';

/// Primeira aba do paciente: o que é a doença e para onde ir em seguida.
class AbaInicioPaciente extends StatelessWidget {
  const AbaInicioPaciente({super.key, required this.aoIrPara});

  /// Troca a aba da casca: 1 Prevenção, 2 Meu risco, 3 Seu exame, 4 Números.
  final void Function(int indice) aoIrPara;

  @override
  Widget build(BuildContext context) {
    final texto = Theme.of(context).textTheme;
    return ListView(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 32),
      children: [
        Container(
          padding: const EdgeInsets.all(22),
          decoration: BoxDecoration(
            color: Cores.tealClaro,
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: Cores.linha),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Icon(Icons.favorite, color: Cores.teal, size: 32),
              const SizedBox(height: 12),
              Text(
                'Câncer do colo do útero:\nprevenir é possível',
                style: texto.headlineSmall,
              ),
              const SizedBox(height: 10),
              Text(
                'É um dos poucos cânceres que podem ser evitados com vacina e '
                'encontrados ainda como lesão, antes de virar câncer, por um '
                'exame simples.',
                style: texto.bodyLarge,
              ),
            ],
          ),
        ),
        const TituloSecao('O que é'),
        const CartaoInfo(
          icone: Icons.biotech_outlined,
          titulo: 'Começa no colo do útero',
          texto: 'O colo do útero é a parte de baixo do útero, que se abre na '
              'vagina. O câncer se desenvolve quando células dessa região '
              'sofrem alterações persistentes.',
        ),
        const CartaoInfo(
          icone: Icons.coronavirus_outlined,
          titulo: 'O papel do HPV',
          texto: 'A infecção persistente por tipos de alto risco do HPV '
              '(papilomavírus humano), transmitido por contato sexual, é a '
              'principal causa. A maioria das infecções some sozinha; algumas '
              'evoluem ao longo de anos.',
        ),
        const CartaoInfo(
          icone: Icons.hourglass_bottom,
          titulo: 'Evolução lenta, tempo para agir',
          texto: 'Entre a infecção e o câncer costumam passar muitos anos. '
              'Nesse intervalo, lesões pré-cancerosas podem ser encontradas no '
              'exame preventivo e tratadas.',
        ),
        const CartaoInfo(
          icone: Icons.visibility_off_outlined,
          titulo: 'Muitas vezes não dá sintoma',
          texto: 'No início, geralmente não há sinais. Por isso o exame '
              'preventivo deve ser feito mesmo se você se sente bem.',
        ),
        const TituloSecao('Explore'),
        _Atalho(
          icone: Icons.health_and_safety,
          titulo: 'Como prevenir',
          subtitulo: 'Vacina, exame preventivo e hábitos',
          aoToque: () => aoIrPara(1),
        ),
        _Atalho(
          icone: Icons.fact_check,
          titulo: 'Conhecer meu perfil de risco',
          subtitulo: 'Questionário rápido; suas respostas ficam só no aparelho',
          aoToque: () => aoIrPara(2),
        ),
        _Atalho(
          icone: Icons.description,
          titulo: 'Entender o resultado do meu exame',
          subtitulo: 'O que significam NILM, LSIL, HSIL e carcinoma',
          aoToque: () => aoIrPara(3),
        ),
        _Atalho(
          icone: Icons.bar_chart,
          titulo: 'Números e estatísticas',
          subtitulo: 'Brasil e mundo, com a fonte de cada dado',
          aoToque: () => aoIrPara(4),
        ),
        const SizedBox(height: 16),
        const AvisoEducativo(
          'Este aplicativo é educativo e não faz diagnóstico nem substitui '
          'consulta ou exame. Em caso de dúvida, procure a Unidade Básica de '
          'Saúde mais próxima.',
        ),
      ],
    );
  }
}

class _Atalho extends StatelessWidget {
  const _Atalho({
    required this.icone,
    required this.titulo,
    required this.subtitulo,
    required this.aoToque,
  });

  final IconData icone;
  final String titulo;
  final String subtitulo;
  final VoidCallback aoToque;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Card(
        clipBehavior: Clip.antiAlias,
        child: ListTile(
          leading: Icon(icone),
          title: Text(
            titulo,
            style: const TextStyle(fontWeight: FontWeight.w700),
          ),
          subtitle: Text(subtitulo),
          trailing: const Icon(Icons.chevron_right),
          onTap: aoToque,
        ),
      ),
    );
  }
}
