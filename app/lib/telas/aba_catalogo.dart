import 'package:flutter/material.dart';

import '../catalogo/controle_catalogo.dart';
import '../catalogo/firestore_rest.dart';
import '../catalogo/modelos.dart';
import '../config.dart';
import '../tema.dart';
import 'form_lamina.dart';
import 'tela_lamina.dart';

/// Foto de uma lâmina: embutida (base64) ou link do Firebase Storage.
class FotoLamina extends StatelessWidget {
  const FotoLamina({
    super.key,
    required this.lamina,
    this.fit = BoxFit.cover,
    this.larguraDecodificada,
  });

  final Lamina lamina;
  final BoxFit fit;

  /// Reduz a decodificação nas miniaturas para poupar memória.
  final int? larguraDecodificada;

  @override
  Widget build(BuildContext context) {
    final erro = ColoredBox(
      color: const Color(0xFF0D1A22),
      child: const Center(
        child: Icon(Icons.broken_image_outlined, color: Colors.white54),
      ),
    );
    final bytes = lamina.bytes;
    if (bytes != null) {
      return Image.memory(
        bytes,
        fit: fit,
        cacheWidth: larguraDecodificada,
        gaplessPlayback: true,
        excludeFromSemantics: true,
        errorBuilder: (_, __, ___) => erro,
      );
    }
    final url = lamina.urlHttps;
    if (url != null) {
      return Image.network(
        url,
        fit: fit,
        cacheWidth: larguraDecodificada,
        excludeFromSemantics: true,
        errorBuilder: (_, __, ___) => erro,
      );
    }
    return erro;
  }
}

/// Módulo de catálogo de lâminas (CitoAtlas), no modo profissional.
///
/// Protocolo cego: quem ainda não avaliou uma lâmina não vê classificação,
/// descrição nem observação de ninguém; vê só a foto, o ID, a data e quantos
/// votos ela tem. Depois de avaliar, vê a própria avaliação.
class AbaCatalogo extends StatefulWidget {
  const AbaCatalogo({super.key, required this.controle});

  final ControleCatalogo controle;

  @override
  State<AbaCatalogo> createState() => _AbaCatalogoState();
}

class _AbaCatalogoState extends State<AbaCatalogo> {
  final _busca = TextEditingController();
  String _termo = '';

  @override
  void initState() {
    super.initState();
    widget.controle.iniciar();
  }

  @override
  void dispose() {
    _busca.dispose();
    super.dispose();
  }

  List<Lamina> _filtradas() {
    final t = _termo.trim().toLowerCase();
    if (t.isEmpty) return widget.controle.laminas;
    return [
      for (final l in widget.controle.laminas)
        if (l.idPublico.texto.toLowerCase().contains(t)) l,
    ];
  }

  Future<void> _entrar() async {
    final sessao = widget.controle.sessao;
    await sessao.entrar();
    if (!mounted) return;
    final erro = sessao.erro;
    if (erro != null) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(erro), duration: const Duration(seconds: 8)),
      );
    }
  }

  Future<void> _novaLamina() async {
    final c = widget.controle;
    if (!c.sessao.logado) {
      final entrar = await showDialog<bool>(
        context: context,
        builder: (ctx) => AlertDialog(
          title: const Text('Entre com o Google'),
          content: const Text(
            'Para cadastrar lâminas, é preciso estar autenticado com uma conta '
            'Google. Isso garante a autoria científica e a rastreabilidade dos '
            'registros.',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(ctx).pop(false),
              child: const Text('Agora não'),
            ),
            FilledButton(
              style: FilledButton.styleFrom(minimumSize: const Size(0, 44)),
              onPressed: () => Navigator.of(ctx).pop(true),
              child: const Text('Entrar'),
            ),
          ],
        ),
      );
      if (entrar != true) return;
      await _entrar();
      if (!c.sessao.logado || !mounted) return;
    }
    await Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => FormLamina(controle: c)),
    );
  }

  @override
  Widget build(BuildContext context) {
    final c = widget.controle;
    return ListenableBuilder(
      listenable: c,
      builder: (context, _) {
        final lista = _filtradas();
        return Scaffold(
          backgroundColor: Colors.transparent,
          floatingActionButton: FloatingActionButton.extended(
            onPressed: _novaLamina,
            icon: const Icon(Icons.add_a_photo_outlined),
            label: const Text('Nova lâmina'),
          ),
          body: RefreshIndicator(
            onRefresh: c.recarregar,
            child: CustomScrollView(
              slivers: [
                SliverPadding(
                  padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
                  sliver: SliverList(
                    delegate: SliverChildListDelegate([
                      const _AvisoCego(),
                      const SizedBox(height: 12),
                      _LinhaLogin(controle: c, aoEntrar: _entrar),
                      const SizedBox(height: 12),
                      TextField(
                        controller: _busca,
                        onChanged: (v) => setState(() => _termo = v),
                        textInputAction: TextInputAction.search,
                        decoration: InputDecoration(
                          hintText: 'Buscar pelo ID (ex.: L-0012)',
                          prefixIcon: const Icon(Icons.search),
                          suffixIcon: _termo.isEmpty
                              ? null
                              : IconButton(
                                  tooltip: 'Limpar busca',
                                  icon: const Icon(Icons.close),
                                  onPressed: () {
                                    _busca.clear();
                                    setState(() => _termo = '');
                                  },
                                ),
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                          filled: true,
                          fillColor: Colors.white,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        '${lista.length} de ${c.laminas.length} lâminas carregadas',
                        style: const TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                          color: Cores.tintaFraca,
                        ),
                      ),
                      const SizedBox(height: 12),
                    ]),
                  ),
                ),
                SliverPadding(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  sliver: SliverGrid(
                    gridDelegate:
                        const SliverGridDelegateWithMaxCrossAxisExtent(
                      maxCrossAxisExtent: 230,
                      mainAxisSpacing: 12,
                      crossAxisSpacing: 12,
                      childAspectRatio: 0.82,
                    ),
                    delegate: SliverChildBuilderDelegate(
                      (context, i) => _CartaoLamina(
                        lamina: lista[i],
                        controle: c,
                        aoAbrir: () => Navigator.of(context).push(
                          MaterialPageRoute(
                            builder: (_) => TelaLamina(
                              controle: c,
                              lamina: lista[i],
                            ),
                          ),
                        ),
                      ),
                      childCount: lista.length,
                    ),
                  ),
                ),
                SliverToBoxAdapter(
                  child: Padding(
                    padding: const EdgeInsets.fromLTRB(16, 16, 16, 96),
                    child: _Rodape(controle: c, vazia: lista.isEmpty),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}

class _AvisoCego extends StatelessWidget {
  const _AvisoCego();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Cores.tealClaro,
        borderRadius: BorderRadius.circular(12),
      ),
      child: const Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.visibility_off_outlined, color: Cores.teal, size: 22),
          SizedBox(width: 10),
          Expanded(
            child: Text(
              'Avaliação cega. Ao avaliar uma lâmina, seu resultado fica sob '
              'sigilo e não altera a visão dos demais analistas. O quórum '
              'oficial exige a concordância de pelo menos '
              '$catalogoVotosParaQuorum citopatologistas.',
              style: TextStyle(fontSize: 14, height: 1.45, color: Cores.tinta),
            ),
          ),
        ],
      ),
    );
  }
}

class _LinhaLogin extends StatelessWidget {
  const _LinhaLogin({required this.controle, required this.aoEntrar});

  final ControleCatalogo controle;
  final VoidCallback aoEntrar;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: controle.sessao,
      builder: (context, _) {
        final s = controle.sessao;
        if (s.entrando) {
          return const Row(
            children: [
              SizedBox(
                width: 18,
                height: 18,
                child: CircularProgressIndicator(strokeWidth: 2.5),
              ),
              SizedBox(width: 12),
              Text('Entrando…'),
            ],
          );
        }
        final u = s.usuario;
        if (u == null) {
          return Row(
            children: [
              const Expanded(
                child: Text(
                  'Para avaliar e cadastrar lâminas, entre com o Google.',
                  style: TextStyle(fontSize: 14, color: Cores.tintaFraca),
                ),
              ),
              const SizedBox(width: 8),
              OutlinedButton.icon(
                style: OutlinedButton.styleFrom(minimumSize: const Size(0, 44)),
                onPressed: aoEntrar,
                icon: const Icon(Icons.login),
                label: const Text('Entrar'),
              ),
            ],
          );
        }
        return Row(
          children: [
            const Icon(Icons.verified_user_outlined, color: Cores.teal),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                u.nome,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(fontWeight: FontWeight.w700),
              ),
            ),
            TextButton(onPressed: s.sair, child: const Text('Sair')),
          ],
        );
      },
    );
  }
}

class _CartaoLamina extends StatelessWidget {
  const _CartaoLamina({
    required this.lamina,
    required this.controle,
    required this.aoAbrir,
  });

  final Lamina lamina;
  final ControleCatalogo controle;
  final VoidCallback aoAbrir;

  @override
  Widget build(BuildContext context) {
    final id = lamina.idPublico;
    final minha = controle.minhaAvaliacaoDe(lamina.id);
    final votos = controle.votosDe(lamina.id);
    final quorum = votos != null && votos >= catalogoVotosParaQuorum;
    final textoVotos = votos == null
        ? 'votos indisponíveis'
        : votos == 0
            ? 'Sem votos ainda'
            : quorum
                ? '$votos votos · quórum atingido'
                : '$votos/$catalogoVotosParaQuorum avaliações';
    final rotuloSelo = minha != null
        ? 'sua avaliação: ${minha.classificacao.sigla}'
        : 'pendente de sua avaliação';

    return Semantics(
      button: true,
      label: '${id.texto}, ${formatarData(lamina.quando)}, $textoVotos, $rotuloSelo',
      excludeSemantics: true,
      child: Card(
        clipBehavior: Clip.antiAlias,
        child: InkWell(
          onTap: aoAbrir,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Expanded(
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    FotoLamina(lamina: lamina, larguraDecodificada: 480),
                    Positioned(
                      top: 8,
                      right: 8,
                      child: _Selo(
                        texto: minha != null
                            ? minha.classificacao.sigla
                            : 'Pendente',
                        icone: minha != null
                            ? Icons.check_circle_outline
                            : Icons.lock_outline,
                        cor: minha?.classificacao.cor,
                      ),
                    ),
                  ],
                ),
              ),
              Padding(
                padding: const EdgeInsets.fromLTRB(12, 10, 12, 12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Flexible(
                          child: Text(
                            id.texto,
                            overflow: TextOverflow.ellipsis,
                            style: const TextStyle(
                              fontWeight: FontWeight.w800,
                              fontSize: 14.5,
                            ),
                          ),
                        ),
                        Text(
                          formatarData(lamina.quando),
                          style: const TextStyle(
                            fontSize: 12.5,
                            color: Cores.tintaFraca,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 4),
                    Text(
                      textoVotos,
                      style: TextStyle(
                        fontSize: 12.5,
                        fontWeight: FontWeight.w700,
                        color: quorum ? const Color(0xFF2E7D62) : Cores.tintaFraca,
                      ),
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

class _Selo extends StatelessWidget {
  const _Selo({required this.texto, required this.icone, this.cor});

  final String texto;
  final IconData icone;
  final Color? cor;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 4),
      decoration: BoxDecoration(
        color: (cor ?? const Color(0xFF0D1A22)).withValues(alpha: 0.9),
        borderRadius: BorderRadius.circular(999),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icone, size: 13, color: Colors.white),
          const SizedBox(width: 4),
          Text(
            texto,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 12,
              fontWeight: FontWeight.w700,
            ),
          ),
        ],
      ),
    );
  }
}

class _Rodape extends StatelessWidget {
  const _Rodape({required this.controle, required this.vazia});

  final ControleCatalogo controle;
  final bool vazia;

  @override
  Widget build(BuildContext context) {
    final c = controle;
    if (c.carregando) {
      return const Center(
        child: Padding(
          padding: EdgeInsets.all(12),
          child: CircularProgressIndicator(),
        ),
      );
    }
    final erro = c.erro;
    if (erro != null) {
      return _CartaoErro(erro: erro, aoTentar: c.carregarMais);
    }
    return Column(
      children: [
        if (vazia)
          const Padding(
            padding: EdgeInsets.symmetric(vertical: 24),
            child: Text(
              'Nenhuma lâmina encontrada.',
              style: TextStyle(color: Cores.tintaFraca),
            ),
          ),
        if (!c.fim)
          OutlinedButton.icon(
            style: OutlinedButton.styleFrom(minimumSize: const Size(0, 48)),
            onPressed: c.carregarMais,
            icon: const Icon(Icons.expand_more),
            label: const Text('Carregar mais lâminas'),
          ),
        if (!c.votosDisponiveis && c.laminas.isNotEmpty)
          const Padding(
            padding: EdgeInsets.only(top: 12),
            child: Text(
              'A contagem de votos não está disponível agora.',
              style: TextStyle(fontSize: 13, color: Cores.tintaFraca),
            ),
          ),
      ],
    );
  }
}

class _CartaoErro extends StatelessWidget {
  const _CartaoErro({required this.erro, required this.aoTentar});

  final ErroCatalogo erro;
  final VoidCallback aoTentar;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFFFDECEC),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFFF0B8B8)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Não foi possível carregar o catálogo.',
            style: TextStyle(fontWeight: FontWeight.w700, color: Color(0xFF6B1F1F)),
          ),
          const SizedBox(height: 6),
          Text(
            erro.mensagem,
            style: const TextStyle(color: Color(0xFF6B1F1F), height: 1.4),
          ),
          const SizedBox(height: 10),
          OutlinedButton.icon(
            onPressed: aoTentar,
            icon: const Icon(Icons.refresh),
            label: const Text('Tentar de novo'),
          ),
        ],
      ),
    );
  }
}
