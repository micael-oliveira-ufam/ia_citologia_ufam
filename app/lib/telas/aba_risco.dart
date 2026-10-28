import 'package:flutter/material.dart';

import '../perfil.dart';
import '../servicos/modelo_risco.dart';
import '../tema.dart';
import '../widgets/conteudo.dart';

/// Questionário de perfil de risco. As respostas são processadas no aparelho
/// por um modelo treinado em dados públicos (ml/train.py).
///
/// O paciente recebe a faixa e uma orientação em linguagem simples; o
/// profissional recebe, além disso, o escore, os cortes e as taxas observadas.
class AbaRisco extends StatefulWidget {
  const AbaRisco({super.key, required this.modelo, required this.perfil});

  final Future<ModeloRisco> modelo;
  final Perfil perfil;

  @override
  State<AbaRisco> createState() => _AbaRiscoState();
}

class _AbaRiscoState extends State<AbaRisco> {
  final _formulario = GlobalKey<FormState>();
  final _rolagem = ScrollController();
  final _idade = TextEditingController();
  final _parceiros = TextEditingController();
  final _primeiraRelacao = TextEditingController();
  final _gestacoes = TextEditingController();
  final _anosFumando = TextEditingController();
  final _anosHormonal = TextEditingController();
  final _anosDiu = TextEditingController();
  final _numeroDeIsts = TextEditingController();

  bool _fuma = false;
  bool _hormonal = false;
  bool _diu = false;
  bool _ists = false;
  bool _ocupado = false;
  ModeloRisco? _modelo;
  ResultadoRisco? _resultado;

  @override
  void dispose() {
    for (final c in [
      _idade,
      _parceiros,
      _primeiraRelacao,
      _gestacoes,
      _anosFumando,
      _anosHormonal,
      _anosDiu,
      _numeroDeIsts,
    ]) {
      c.dispose();
    }
    _rolagem.dispose();
    super.dispose();
  }

  static double? _ler(String? s) =>
      double.tryParse((s ?? '').trim().replaceAll(',', '.'));

  static String _formatar(double v) =>
      v == v.roundToDouble() ? v.toInt().toString() : v.toString();

  /// Os limites são o intervalo observado no treino: fora dele a estimativa
  /// não é confiável.
  String? Function(String?) _faixa(
    double min,
    double max, {
    bool inteiro = false,
  }) {
    return (v) {
      final n = _ler(v);
      if (n == null) return 'Informe um número';
      if (inteiro && n != n.roundToDouble()) return 'Use um número inteiro';
      if (n < min || n > max) {
        return 'Valor entre ${_formatar(min)} e ${_formatar(max)}';
      }
      return null;
    };
  }

  String? _validarPrimeiraRelacao(String? v) {
    final base = _faixa(10, 32, inteiro: true)(v);
    if (base != null) return base;
    final idade = _ler(_idade.text);
    if (idade != null && _ler(v)! > idade) {
      return 'Não pode ser maior que a idade';
    }
    return null;
  }

  Future<void> _calcular() async {
    if (!(_formulario.currentState?.validate() ?? false)) return;
    setState(() => _ocupado = true);
    try {
      final modelo = await widget.modelo;
      final respostas = <String, double?>{
        'age': _ler(_idade.text),
        'partners': _ler(_parceiros.text),
        'first_intercourse': _ler(_primeiraRelacao.text),
        'pregnancies': _ler(_gestacoes.text),
        'smokes': _fuma ? 1.0 : 0.0,
        'smokes_years': _fuma ? _ler(_anosFumando.text) : 0.0,
        'hormonal': _hormonal ? 1.0 : 0.0,
        'hormonal_years': _hormonal ? _ler(_anosHormonal.text) : 0.0,
        'iud': _diu ? 1.0 : 0.0,
        'iud_years': _diu ? _ler(_anosDiu.text) : 0.0,
        'stds': _ists ? 1.0 : 0.0,
        'stds_number': _ists ? _ler(_numeroDeIsts.text) : 0.0,
      };
      final resultado = modelo.avaliar(respostas);
      if (!mounted) return;
      setState(() {
        _modelo = modelo;
        _resultado = resultado;
        _ocupado = false;
      });
      if (_rolagem.hasClients) _rolagem.jumpTo(0);
    } catch (_) {
      if (!mounted) return;
      setState(() => _ocupado = false);
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Não foi possível calcular a estimativa.'),
        ),
      );
    }
  }

  void _refazer() => setState(() => _resultado = null);

  @override
  Widget build(BuildContext context) {
    final resultado = _resultado;
    final modelo = _modelo;
    return ListView(
      controller: _rolagem,
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 32),
      children: [
        if (resultado != null && modelo != null)
          _VistaResultado(
            resultado: resultado,
            modelo: modelo,
            perfil: widget.perfil,
            aoRefazer: _refazer,
          )
        else
          _formularioDeRisco(context),
      ],
    );
  }

  Widget _formularioDeRisco(BuildContext context) {
    final pro = widget.perfil.ehProfissional;
    return Form(
      key: _formulario,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            pro ? 'Estimativa de perfil de risco' : 'Meu perfil de risco',
            style: Theme.of(context).textTheme.headlineSmall,
          ),
          const SizedBox(height: 8),
          Text(
            pro
                ? 'Modelo de triagem educativo treinado com dados públicos '
                    '(UCI, 858 pacientes). Use para ilustrar fatores de risco '
                    'na conversa com a paciente, não para decidir condutas. '
                    'As respostas são processadas no aparelho e não são '
                    'guardadas.'
                : 'Estimativa educativa feita por um modelo de IA treinado com '
                    'um conjunto de dados público. Suas respostas são '
                    'processadas só no aparelho e não são enviadas nem '
                    'guardadas.',
            style: const TextStyle(
              fontSize: 15,
              color: Cores.tintaFraca,
              height: 1.5,
            ),
          ),
          const SizedBox(height: 12),
          const AvisoEducativo(
            'Vale para mulheres que já tiveram vida sexual. O modelo usa idade, '
            'hábitos e histórico sexual e reprodutivo. O conjunto de dados não '
            'inclui renda nem escolaridade.',
          ),
          const TituloSecao('Perfil'),
          _campo('Idade (anos)', _idade, _faixa(13, 84, inteiro: true)),
          _campo(
            'Idade na primeira relação sexual',
            _primeiraRelacao,
            _validarPrimeiraRelacao,
          ),
          _campo(
            'Número de parceiros sexuais na vida',
            _parceiros,
            _faixa(1, 28, inteiro: true),
          ),
          _campo(
            'Número de gestações',
            _gestacoes,
            _faixa(0, 11, inteiro: true),
          ),
          const TituloSecao('Hábitos e histórico'),
          _chave('Fuma ou já fumou regularmente', _fuma,
              (v) => setState(() => _fuma = v)),
          if (_fuma)
            _campo('Há quantos anos fuma (pode usar decimais)', _anosFumando,
                _faixa(0, 37)),
          _chave('Usa ou já usou anticoncepcional hormonal', _hormonal,
              (v) => setState(() => _hormonal = v)),
          if (_hormonal)
            _campo('Por quantos anos (pode usar decimais)', _anosHormonal,
                _faixa(0, 30)),
          _chave('Usa ou já usou DIU', _diu, (v) => setState(() => _diu = v)),
          if (_diu)
            _campo('Por quantos anos (pode usar decimais)', _anosDiu,
                _faixa(0, 19)),
          _chave(
            'Já teve alguma infecção sexualmente transmissível (IST)',
            _ists,
            (v) => setState(() => _ists = v),
          ),
          if (_ists)
            _campo('Quantas ISTs diferentes', _numeroDeIsts,
                _faixa(1, 4, inteiro: true)),
          const SizedBox(height: 20),
          FilledButton.icon(
            onPressed: _ocupado ? null : _calcular,
            icon: _ocupado
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.analytics_outlined),
            label: const Text('Ver estimativa'),
          ),
        ],
      ),
    );
  }

  Widget _campo(
    String rotulo,
    TextEditingController controlador,
    String? Function(String?) validador,
  ) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: TextFormField(
        controller: controlador,
        keyboardType: const TextInputType.numberWithOptions(decimal: true),
        decoration: InputDecoration(
          labelText: rotulo,
          border: const OutlineInputBorder(),
        ),
        validator: validador,
      ),
    );
  }

  Widget _chave(String rotulo, bool valor, ValueChanged<bool> aoMudar) {
    return SwitchListTile(
      contentPadding: EdgeInsets.zero,
      title: Text(rotulo),
      value: valor,
      onChanged: aoMudar,
    );
  }
}

class _EstiloNivel {
  const _EstiloNivel(this.rotulo, this.cor, this.icone, this.resumo);

  final String rotulo;
  final Color cor;
  final IconData icone;
  final String resumo;
}

_EstiloNivel _estiloDe(NivelRisco nivel) {
  switch (nivel) {
    case NivelRisco.baixo:
      return const _EstiloNivel(
        'Risco relativo: MENOR',
        Color(0xFF2E7D32),
        Icons.check_circle_outline,
        'O perfil ficou na faixa de menor risco relativo entre as mulheres do '
            'conjunto de dados. Isso não significa ausência de risco.',
      );
    case NivelRisco.moderado:
      return const _EstiloNivel(
        'Risco relativo: INTERMEDIÁRIO',
        Color(0xFFB45309),
        Icons.warning_amber_rounded,
        'O perfil ficou na faixa intermediária de risco relativo entre as '
            'mulheres do conjunto de dados.',
      );
    case NivelRisco.alto:
      return const _EstiloNivel(
        'Risco relativo: MAIOR',
        Color(0xFFC62828),
        Icons.error_outline,
        'O perfil ficou na faixa de maior risco relativo entre as mulheres do '
            'conjunto de dados. Vale conversar com a equipe de saúde.',
      );
  }
}

class _VistaResultado extends StatelessWidget {
  const _VistaResultado({
    required this.resultado,
    required this.modelo,
    required this.perfil,
    required this.aoRefazer,
  });

  final ResultadoRisco resultado;
  final ModeloRisco modelo;
  final Perfil perfil;
  final VoidCallback aoRefazer;

  @override
  Widget build(BuildContext context) {
    final estilo = _estiloDe(resultado.nivel);
    final texto = Theme.of(context).textTheme;
    final o = resultado.observado;
    final base = modelo.prevalencia;
    final contraTotal =
        base > 0 ? ', contra ${percentual(base)} no total' : '';
    final totalTreino = modelo.validacao['n'] ?? 858;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Semantics(
          liveRegion: true,
          label: '${estilo.rotulo}. ${estilo.resumo}',
          excludeSemantics: true,
          child: Container(
            width: double.infinity,
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
              color: estilo.cor,
              borderRadius: BorderRadius.circular(16),
            ),
            child: Row(
              children: [
                Icon(estilo.icone, color: Colors.white, size: 36),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    estilo.rotulo,
                    style: texto.titleLarge?.copyWith(
                      color: Colors.white,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 14),
        Text(estilo.resumo, style: texto.bodyLarge),
        const SizedBox(height: 10),
        Text(
          'No estudo, entre as mulheres que caíram nesta faixa nas previsões '
          'feitas fora da amostra, ${percentual(o.taxa)} (${o.positivos} de '
          '${o.n}) tiveram biópsia positiva$contraTotal.',
          style: texto.bodyMedium,
        ),
        if (perfil.ehProfissional) ...[
          const TituloSecao('Detalhes técnicos'),
          _DetalhesTecnicos(resultado: resultado, modelo: modelo),
        ],
        const TituloSecao('O que isso significa'),
        const CartaoInfo(
          icone: Icons.event_available_outlined,
          titulo: 'O exame preventivo continua necessário',
          texto: 'Qualquer resultado, inclusive “menor”, não dispensa o '
              'rastreamento. Se você nunca fez o exame ou faz muito tempo, '
              'procure a UBS para agendar.',
        ),
        CartaoInfo(
          icone: Icons.rule,
          titulo: 'Não é diagnóstico',
          texto: 'A estimativa tem poder de discriminação limitado: muitas '
              'mulheres com biópsia positiva ficam na faixa menor e muitas na '
              'maior não têm a doença. Pequenas mudanças nas respostas podem '
              'mudar a faixa. O modelo foi treinado com $totalTreino '
              'pacientes de um único hospital (Caracas, Venezuela), que pode '
              'não representar a sua realidade. Veja detalhes em “Sobre o '
              'projeto”.',
        ),
        const SizedBox(height: 8),
        OutlinedButton.icon(
          onPressed: aoRefazer,
          icon: const Icon(Icons.refresh),
          label: const Text('Refazer questionário'),
        ),
      ],
    );
  }
}

class _DetalhesTecnicos extends StatelessWidget {
  const _DetalhesTecnicos({required this.resultado, required this.modelo});

  final ResultadoRisco resultado;
  final ModeloRisco modelo;

  @override
  Widget build(BuildContext context) {
    final estilo = Theme.of(context).textTheme;
    final perm = modelo.permutacao;
    final auc = modelo.aucForaDaAmostra;
    final nuloMedia = perm['null_mean'];
    final nuloDesvio = perm['null_std'];
    final pValor = perm['p_value'];
    final arvores = modelo.numeroDeArvores > 0
        ? ' (${modelo.numeroDeArvores} árvores)'
        : '';
    const nomes = ['menor', 'intermediária', 'maior'];

    Widget linha(String rotulo, String valor) => Padding(
          padding: const EdgeInsets.only(bottom: 6),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                flex: 4,
                child: Text(rotulo, style: estilo.bodyMedium),
              ),
              Expanded(
                flex: 5,
                child: Text(
                  valor,
                  style: estilo.bodyMedium?.copyWith(
                    color: Cores.tinta,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ],
          ),
        );

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            linha('Modelo', '${modelo.nome}$arvores'),
            linha(
              'Escore',
              '${numero(resultado.escore, casas: 3)}  (saída do modelo, não é '
                  'probabilidade calibrada)',
            ),
            linha(
              'Cortes',
              'intermediária ≥ ${numero(modelo.corteModerado, casas: 3)}; '
                  'maior ≥ ${numero(modelo.corteAlto, casas: 3)}',
            ),
            const Divider(height: 20),
            Text(
              'Biópsia positiva por faixa, em previsões fora da amostra',
              style: estilo.titleMedium,
            ),
            const SizedBox(height: 8),
            for (var i = 0; i < modelo.faixas.length; i++)
              linha(
                'Faixa ${nomes[i]}',
                '${percentual(modelo.faixas[i].taxa)} '
                    '(${modelo.faixas[i].positivos} de ${modelo.faixas[i].n})',
              ),
            const Divider(height: 20),
            if (auc != null)
              linha('AUC fora da amostra', numero(auc)),
            if (nuloMedia is num && nuloDesvio is num)
              linha(
                'AUC com rótulos embaralhados',
                '${numero(nuloMedia.toDouble())} ± '
                    '${numero(nuloDesvio.toDouble())}',
              ),
            if (pValor is num)
              linha('Valor de p (permutação)', numero(pValor.toDouble())),
            const SizedBox(height: 4),
            const Text(
              'Sinal fraco: serve para ilustrar fatores de risco, não para '
              'estratificar condutas.',
              style: TextStyle(
                fontSize: 13.5,
                color: Cores.tintaFraca,
                height: 1.4,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
