import 'package:flutter/material.dart';

import '../catalogo/controle_catalogo.dart';
import '../catalogo/modelos.dart';
import '../config.dart';
import '../tema.dart';
import 'aba_catalogo.dart' show FotoLamina;
import 'form_lamina.dart';

/// Detalhe de uma lâmina: foto com zoom e avaliação cega.
class TelaLamina extends StatefulWidget {
  const TelaLamina({super.key, required this.controle, required this.lamina});

  final ControleCatalogo controle;
  final Lamina lamina;

  @override
  State<TelaLamina> createState() => _TelaLaminaState();
}

class _TelaLaminaState extends State<TelaLamina> {
  Classificacao? _escolha;
  final _descricao = TextEditingController();
  final _observacoes = TextEditingController();
  bool _salvando = false;
  bool _refazendo = false;

  ControleCatalogo get _c => widget.controle;
  Lamina get _l => widget.lamina;

  @override
  void dispose() {
    _descricao.dispose();
    _observacoes.dispose();
    super.dispose();
  }

  void _aviso(String texto) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(texto)));
  }

  Future<void> _entrar() async {
    await _c.sessao.entrar();
    if (!mounted) return;
    final erro = _c.sessao.erro;
    if (erro != null) _aviso(erro);
  }

  void _comecarRefazer(Avaliacao a) {
    setState(() {
      _refazendo = true;
      _escolha = a.classificacao;
      _descricao.text = a.descricao;
      _observacoes.text = a.observacoes;
    });
  }

  Future<void> _salvar() async {
    final escolha = _escolha;
    if (escolha == null) {
      _aviso('Escolha uma classificação.');
      return;
    }
    setState(() => _salvando = true);
    try {
      await _c.avaliar(_l, escolha, _descricao.text, _observacoes.text);
      if (!mounted) return;
      setState(() => _refazendo = false);
      _aviso('Avaliação registrada. Ela fica visível só para você.');
    } catch (e) {
      if (mounted) _aviso('Não foi possível salvar: $e');
    } finally {
      if (mounted) setState(() => _salvando = false);
    }
  }

  Future<void> _excluir() async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Excluir esta lâmina?'),
        content: const Text(
          'A lâmina sai do catálogo para todos. As avaliações já feitas por '
          'outras pessoas permanecem no banco.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: const Text('Cancelar'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(ctx).pop(true),
            child: const Text('Excluir'),
          ),
        ],
      ),
    );
    if (ok != true) return;
    try {
      await _c.excluirLamina(_l);
      if (mounted) Navigator.of(context).pop();
    } catch (e) {
      if (mounted) _aviso('Não foi possível excluir: $e');
    }
  }

  void _ampliar() {
    Navigator.of(context).push(
      MaterialPageRoute(
        fullscreenDialog: true,
        builder: (_) => Scaffold(
          backgroundColor: Colors.black,
          appBar: AppBar(
            backgroundColor: Colors.black,
            foregroundColor: Colors.white,
            title: Text(_l.idPublico.texto),
          ),
          body: InteractiveViewer(
            minScale: 1,
            maxScale: 8,
            child: Center(
              child: FotoLamina(lamina: _l, fit: BoxFit.contain),
            ),
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: Listenable.merge([_c, _c.sessao]),
      builder: (context, _) {
        final minha = _c.minhaAvaliacaoDe(_l.id);
        final votos = _c.votosDe(_l.id);
        final logado = _c.sessao.logado;
        final mostrarFormulario = logado && (minha == null || _refazendo);
        return Scaffold(
          appBar: AppBar(
            title: Text(_l.idPublico.texto),
            actions: [
              if (_c.ehCriador(_l)) ...[
                IconButton(
                  tooltip: 'Editar lâmina',
                  icon: const Icon(Icons.edit_outlined),
                  onPressed: () => Navigator.of(context).push(
                    MaterialPageRoute(
                      builder: (_) =>
                          FormLamina(controle: _c, existente: _l),
                    ),
                  ),
                ),
                IconButton(
                  tooltip: 'Excluir lâmina',
                  icon: const Icon(Icons.delete_outline),
                  onPressed: _excluir,
                ),
              ],
            ],
          ),
          body: ListView(
            padding: const EdgeInsets.all(16),
            children: [
              Semantics(
                button: true,
                label: 'Ampliar a foto da lâmina',
                child: GestureDetector(
                  onTap: _ampliar,
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(12),
                    child: AspectRatio(
                      aspectRatio: 4 / 3,
                      child: FotoLamina(lamina: _l, fit: BoxFit.contain),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 6),
              const Text(
                'Toque na foto para ampliar e dar zoom.',
                style: TextStyle(fontSize: 13, color: Cores.tintaFraca),
              ),
              const SizedBox(height: 12),
              Text(
                '${formatarData(_l.quando)} · '
                '${votos == null ? 'votos indisponíveis' : '$votos de $catalogoVotosParaQuorum votos para o quórum'}',
                style: const TextStyle(fontWeight: FontWeight.w700),
              ),
              if (_l.idPublico.oculto)
                const Padding(
                  padding: EdgeInsets.only(top: 6),
                  child: Text(
                    'O identificador original foi ocultado por privacidade.',
                    style: TextStyle(fontSize: 13, color: Cores.tintaFraca),
                  ),
                ),
              const SizedBox(height: 16),
              if (!logado)
                _Caixa(
                  icone: Icons.lock_outline,
                  texto: 'Entre com o Google para avaliar esta lâmina às cegas. '
                      'Classificações e descrições de outras pessoas nunca são '
                      'exibidas.',
                  acao: OutlinedButton.icon(
                    onPressed: _c.sessao.entrando ? null : _entrar,
                    icon: const Icon(Icons.login),
                    label: const Text('Entrar com o Google'),
                  ),
                ),
              if (logado && minha != null && !_refazendo)
                _MinhaAvaliacao(
                  avaliacao: minha,
                  aoRefazer: () => _comecarRefazer(minha),
                ),
              if (mostrarFormulario) ...[
                const Text(
                  'Sua avaliação cega',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.w800),
                ),
                const SizedBox(height: 8),
                Column(
                  children: [
                    for (final c in Classificacao.values)
                      Semantics(
                        inMutuallyExclusiveGroup: true,
                        selected: _escolha == c,
                        child: ListTile(
                          contentPadding: EdgeInsets.zero,
                          minVerticalPadding: 8,
                          onTap: () => setState(() => _escolha = c),
                          leading: Icon(
                            _escolha == c
                                ? Icons.radio_button_checked
                                : Icons.radio_button_unchecked,
                            color: _escolha == c ? c.cor : Cores.tintaFraca,
                          ),
                          title: Text(
                            c.sigla,
                            style: const TextStyle(fontWeight: FontWeight.w700),
                          ),
                          subtitle: Text(c.descricao),
                        ),
                      ),
                  ],
                ),
                const SizedBox(height: 8),
                TextField(
                  controller: _descricao,
                  minLines: 2,
                  maxLines: 5,
                  maxLength: 2000,
                  decoration: const InputDecoration(
                    labelText: 'Descrição morfológica (opcional)',
                    border: OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 8),
                TextField(
                  controller: _observacoes,
                  minLines: 1,
                  maxLines: 4,
                  maxLength: 2000,
                  decoration: const InputDecoration(
                    labelText: 'Observações (opcional)',
                    border: OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    if (_refazendo)
                      TextButton(
                        onPressed: _salvando
                            ? null
                            : () => setState(() => _refazendo = false),
                        child: const Text('Cancelar'),
                      ),
                    const Spacer(),
                    FilledButton.icon(
                      style: FilledButton.styleFrom(
                        minimumSize: const Size(0, 48),
                      ),
                      onPressed: _salvando ? null : _salvar,
                      icon: _salvando
                          ? const SizedBox(
                              width: 18,
                              height: 18,
                              child: CircularProgressIndicator(strokeWidth: 2.5),
                            )
                          : const Icon(Icons.check),
                      label: const Text('Registrar avaliação'),
                    ),
                  ],
                ),
              ],
              const SizedBox(height: 24),
              const Text(
                'Uso científico e educacional. A classificação por IA não é '
                'aplicada ao catálogo e nada aqui substitui o laudo do '
                'citopatologista responsável.',
                style: TextStyle(fontSize: 13, color: Cores.tintaFraca, height: 1.4),
              ),
            ],
          ),
        );
      },
    );
  }
}

class _Caixa extends StatelessWidget {
  const _Caixa({required this.icone, required this.texto, this.acao});

  final IconData icone;
  final String texto;
  final Widget? acao;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Cores.tealClaro,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(icone, color: Cores.teal),
              const SizedBox(width: 10),
              Expanded(child: Text(texto, style: const TextStyle(height: 1.45))),
            ],
          ),
          if (acao != null) ...[const SizedBox(height: 10), acao!],
        ],
      ),
    );
  }
}

class _MinhaAvaliacao extends StatelessWidget {
  const _MinhaAvaliacao({required this.avaliacao, required this.aoRefazer});

  final Avaliacao avaliacao;
  final VoidCallback aoRefazer;

  @override
  Widget build(BuildContext context) {
    final a = avaliacao;
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Cores.linha),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Sua avaliação (visível só para você)',
            style: TextStyle(fontSize: 13, color: Cores.tintaFraca),
          ),
          const SizedBox(height: 6),
          Text(
            a.classificacao.sigla,
            style: TextStyle(
              fontSize: 22,
              fontWeight: FontWeight.w800,
              color: a.classificacao.cor,
            ),
          ),
          Text(a.classificacao.descricao),
          if (a.descricao.isNotEmpty) ...[
            const SizedBox(height: 10),
            Text(a.descricao, style: const TextStyle(height: 1.45)),
          ],
          if (a.observacoes.isNotEmpty) ...[
            const SizedBox(height: 8),
            Text(
              a.observacoes,
              style: const TextStyle(height: 1.45, color: Cores.tintaFraca),
            ),
          ],
          const SizedBox(height: 8),
          TextButton.icon(
            onPressed: aoRefazer,
            icon: const Icon(Icons.edit_outlined),
            label: const Text('Refazer minha avaliação'),
          ),
        ],
      ),
    );
  }
}
