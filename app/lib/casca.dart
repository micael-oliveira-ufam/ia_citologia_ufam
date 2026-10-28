import 'package:flutter/material.dart';

import 'data/estatisticas.dart';
import 'perfil.dart';
import 'servicos/classificador.dart';
import 'servicos/modelo_risco.dart';
import 'telas/aba_analisar.dart';
import 'catalogo/controle_catalogo.dart';
import 'telas/aba_bethesda.dart';
import 'telas/aba_catalogo.dart';
import 'telas/aba_inicio_paciente.dart';
import 'telas/aba_numeros.dart';
import 'telas/aba_prevencao.dart';
import 'telas/aba_risco.dart';
import 'telas/aba_seu_exame.dart';
import 'telas/tela_sobre.dart';
import 'tema.dart';

class _Aba {
  const _Aba({
    required this.titulo,
    required this.rotulo,
    required this.icone,
    required this.iconeAtivo,
    required this.pagina,
  });

  final String titulo;
  final String rotulo;
  final IconData icone;
  final IconData iconeAtivo;
  final Widget pagina;
}

/// Estrutura de navegação comum aos dois perfis. O que muda é a lista de abas:
///
/// - paciente: Início, Prevenção, Meu risco, Seu exame, Números;
/// - profissional: Analisar (IA de citologia), Bethesda, Orientar, Risco.
class Casca extends StatefulWidget {
  const Casca({
    super.key,
    required this.perfil,
    required this.classificador,
    required this.modeloRisco,
    required this.estatisticas,
    required this.aoTrocarPerfil,
  });

  final Perfil perfil;
  final Classificador classificador;
  final Future<ModeloRisco> modeloRisco;
  final Future<List<Estatistica>> estatisticas;
  final VoidCallback aoTrocarPerfil;

  @override
  State<Casca> createState() => _CascaState();
}

class _CascaState extends State<Casca> {
  int _indice = 0;

  /// Catálogo de lâminas: só existe no perfil profissional e só é criado (e só
  /// consulta a rede) quando a pessoa abre a aba pela primeira vez.
  ControleCatalogo? _catalogo;
  ControleCatalogo get _controleCatalogo => _catalogo ??= ControleCatalogo();

  @override
  void dispose() {
    _catalogo?.dispose();
    super.dispose();
  }

  late final List<_Aba> _abas = widget.perfil.ehProfissional
      ? _abasProfissional()
      : _abasPaciente();

  void _irPara(int indice) => setState(() => _indice = indice);

  List<_Aba> _abasPaciente() => [
        _Aba(
          titulo: 'Colo do útero',
          rotulo: 'Início',
          icone: Icons.home_outlined,
          iconeAtivo: Icons.home,
          pagina: AbaInicioPaciente(aoIrPara: _irPara),
        ),
        const _Aba(
          titulo: 'Como prevenir',
          rotulo: 'Prevenção',
          icone: Icons.health_and_safety_outlined,
          iconeAtivo: Icons.health_and_safety,
          pagina: AbaPrevencao(perfil: Perfil.paciente),
        ),
        _Aba(
          titulo: 'Meu perfil de risco',
          rotulo: 'Meu risco',
          icone: Icons.fact_check_outlined,
          iconeAtivo: Icons.fact_check,
          pagina: AbaRisco(modelo: widget.modeloRisco, perfil: widget.perfil),
        ),
        const _Aba(
          titulo: 'Entenda seu exame',
          rotulo: 'Seu exame',
          icone: Icons.description_outlined,
          iconeAtivo: Icons.description,
          pagina: AbaSeuExame(),
        ),
        _Aba(
          titulo: 'Números e estatísticas',
          rotulo: 'Números',
          icone: Icons.bar_chart_outlined,
          iconeAtivo: Icons.bar_chart,
          pagina: AbaNumeros(
            estatisticas: widget.estatisticas,
            perfil: widget.perfil,
          ),
        ),
      ];

  List<_Aba> _abasProfissional() => [
        _Aba(
          titulo: 'Analisar campo',
          rotulo: 'Analisar',
          icone: Icons.biotech_outlined,
          iconeAtivo: Icons.biotech,
          pagina: AbaAnalisar(classificador: widget.classificador),
        ),
        const _Aba(
          titulo: 'Sistema Bethesda',
          rotulo: 'Bethesda',
          icone: Icons.menu_book_outlined,
          iconeAtivo: Icons.menu_book,
          pagina: AbaBethesda(),
        ),
        _Aba(
          titulo: 'Catálogo de lâminas',
          rotulo: 'Catálogo',
          icone: Icons.collections_outlined,
          iconeAtivo: Icons.collections,
          pagina: _CatalogoPreguicoso(
            ativa: () => _indice == 2,
            criar: () => _controleCatalogo,
          ),
        ),
        _Aba(
          titulo: 'Orientar pacientes',
          rotulo: 'Orientar',
          icone: Icons.record_voice_over_outlined,
          iconeAtivo: Icons.record_voice_over,
          pagina: AbaPrevencao(
            perfil: Perfil.profissional,
            estatisticas: widget.estatisticas,
          ),
        ),
        _Aba(
          titulo: 'Estimativa de risco',
          rotulo: 'Risco',
          icone: Icons.fact_check_outlined,
          iconeAtivo: Icons.fact_check,
          pagina: AbaRisco(modelo: widget.modeloRisco, perfil: widget.perfil),
        ),
      ];

  @override
  Widget build(BuildContext context) {
    final aba = _abas[_indice];
    return Scaffold(
      appBar: AppBar(
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(aba.titulo, overflow: TextOverflow.ellipsis),
            Text(
              widget.perfil.modo,
              style: const TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w500,
                color: Cores.tintaFraca,
              ),
            ),
          ],
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.info_outline),
            tooltip: 'Sobre o projeto',
            onPressed: () => Navigator.of(context).push(
              MaterialPageRoute(
                builder: (_) => TelaSobre(
                  perfil: widget.perfil,
                  modeloRisco: widget.modeloRisco,
                ),
              ),
            ),
          ),
          IconButton(
            icon: const Icon(Icons.swap_horiz),
            tooltip: 'Trocar perfil',
            onPressed: widget.aoTrocarPerfil,
          ),
        ],
      ),
      body: SafeArea(
        child: IndexedStack(
          index: _indice,
          children: [for (final a in _abas) a.pagina],
        ),
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _indice,
        onDestinationSelected: _irPara,
        destinations: [
          for (final a in _abas)
            NavigationDestination(
              icon: Icon(a.icone),
              selectedIcon: Icon(a.iconeAtivo),
              label: a.rotulo,
            ),
        ],
      ),
    );
  }
}

/// Só monta [AbaCatalogo] (e portanto só acessa a rede) depois que a aba foi
/// aberta uma vez; depois disso ela fica montada no IndexedStack.
class _CatalogoPreguicoso extends StatefulWidget {
  const _CatalogoPreguicoso({required this.ativa, required this.criar});

  final bool Function() ativa;
  final ControleCatalogo Function() criar;

  @override
  State<_CatalogoPreguicoso> createState() => _CatalogoPreguicosoState();
}

class _CatalogoPreguicosoState extends State<_CatalogoPreguicoso> {
  bool _aberta = false;

  @override
  Widget build(BuildContext context) {
    if (!_aberta && widget.ativa()) _aberta = true;
    if (!_aberta) return const SizedBox.shrink();
    return AbaCatalogo(controle: widget.criar());
  }
}
