import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../catalogo/controle_catalogo.dart';
import '../catalogo/modelos.dart';
import '../tema.dart';

/// Cadastro ou edição de uma lâmina. O ID segue o padrão L-0001 e é sugerido
/// automaticamente; não digite nome, prontuário ou qualquer dado que
/// identifique a paciente.
class FormLamina extends StatefulWidget {
  const FormLamina({super.key, required this.controle, this.existente});

  final ControleCatalogo controle;
  final Lamina? existente;

  @override
  State<FormLamina> createState() => _FormLaminaState();
}

class _FormLaminaState extends State<FormLamina> {
  final _chave = GlobalKey<FormState>();
  final _id = TextEditingController();
  final _descricao = TextEditingController();
  final _observacoes = TextEditingController();
  final _seletor = ImagePicker();

  Classificacao? _classificacao;
  DateTime _data = DateTime.now();
  Uint8List? _jpeg; // foto nova já comprimida
  bool _preparandoFoto = false;
  bool _salvando = false;
  String? _erro;

  Lamina? get _existente => widget.existente;

  @override
  void initState() {
    super.initState();
    final e = _existente;
    if (e != null) {
      _id.text = e.patientId;
      _classificacao = Classificacao.deSigla(e.classificacao);
      _descricao.text = e.descricao ?? '';
      _observacoes.text = e.observacoes ?? '';
      _data = e.quando;
    } else {
      _sugerirId();
    }
  }

  @override
  void dispose() {
    _id.dispose();
    _descricao.dispose();
    _observacoes.dispose();
    super.dispose();
  }

  Future<void> _sugerirId() async {
    try {
      final id = await widget.controle.proximoId();
      if (mounted && _id.text.isEmpty) _id.text = id;
    } catch (_) {/* a pessoa digita */}
  }

  Future<void> _escolherFoto(ImageSource origem) async {
    try {
      final arquivo = await _seletor.pickImage(
        source: origem,
        maxWidth: 2048,
        maxHeight: 2048,
        imageQuality: 92,
      );
      if (arquivo == null) return;
      setState(() {
        _preparandoFoto = true;
        _erro = null;
      });
      final bruto = await arquivo.readAsBytes();
      final jpeg = await compute(comprimirParaCatalogo, bruto);
      if (mounted) setState(() => _jpeg = jpeg);
    } on FormatException catch (e) {
      if (mounted) setState(() => _erro = e.message);
    } catch (e) {
      if (mounted) setState(() => _erro = 'Não foi possível usar esta foto: $e');
    } finally {
      if (mounted) setState(() => _preparandoFoto = false);
    }
  }

  Future<void> _escolherData() async {
    final agora = DateTime.now();
    final d = await showDatePicker(
      context: context,
      initialDate: _data.isAfter(agora) ? agora : _data,
      firstDate: DateTime(2000),
      lastDate: agora,
    );
    if (d != null) setState(() => _data = d);
  }

  Future<void> _salvar() async {
    if (!_chave.currentState!.validate()) return;
    final classificacao = _classificacao;
    if (classificacao == null) {
      setState(() => _erro = 'Escolha a classificação.');
      return;
    }
    if (_existente == null && _jpeg == null) {
      setState(() => _erro = 'Adicione a foto da lâmina.');
      return;
    }
    setState(() {
      _salvando = true;
      _erro = null;
    });
    try {
      final e = _existente;
      if (e == null) {
        await widget.controle.criarLamina(
          jpeg: _jpeg!,
          patientId: _id.text,
          classificacao: classificacao,
          data: _data,
          descricao: _descricao.text,
          observacoes: _observacoes.text,
        );
      } else {
        await widget.controle.atualizarLamina(
          e,
          jpeg: _jpeg,
          patientId: _id.text,
          classificacao: classificacao,
          data: _data,
          descricao: _descricao.text,
          observacoes: _observacoes.text,
        );
      }
      if (!mounted) return;
      Navigator.of(context).pop();
      // Edição abre a partir do detalhe, que mostraria a versão antiga.
      if (e != null) Navigator.of(context).maybePop();
    } catch (e) {
      if (mounted) setState(() => _erro = 'Não foi possível salvar: $e');
    } finally {
      if (mounted) setState(() => _salvando = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final editando = _existente != null;
    return Scaffold(
      appBar: AppBar(title: Text(editando ? 'Editar lâmina' : 'Nova lâmina')),
      body: Form(
        key: _chave,
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Cores.alertaFundo,
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Text(
                'Privacidade: use apenas um código (L-0001). Não registre nome, '
                'prontuário, CPF ou qualquer dado que identifique a paciente, e '
                'só cadastre material que você tem autorização para usar.',
                style: TextStyle(color: Cores.alerta, height: 1.45),
              ),
            ),
            const SizedBox(height: 16),
            AspectRatio(
              aspectRatio: 4 / 3,
              child: ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: ColoredBox(
                  color: const Color(0xFF0D1A22),
                  child: _preparandoFoto
                      ? const Center(child: CircularProgressIndicator())
                      : _jpeg != null
                          ? Image.memory(_jpeg!, fit: BoxFit.contain)
                          : editando
                              ? Image.memory(
                                  _existente!.bytes ?? Uint8List(0),
                                  fit: BoxFit.contain,
                                  errorBuilder: (_, __, ___) => const Center(
                                    child: Icon(Icons.image_outlined,
                                        color: Colors.white54, size: 40),
                                  ),
                                )
                              : const Center(
                                  child: Icon(Icons.image_outlined,
                                      color: Colors.white54, size: 40),
                                ),
                ),
              ),
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    style: OutlinedButton.styleFrom(minimumSize: const Size(0, 48)),
                    onPressed: _preparandoFoto || _salvando
                        ? null
                        : () => _escolherFoto(ImageSource.camera),
                    icon: const Icon(Icons.photo_camera_outlined),
                    label: const Text('Câmera'),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: OutlinedButton.icon(
                    style: OutlinedButton.styleFrom(minimumSize: const Size(0, 48)),
                    onPressed: _preparandoFoto || _salvando
                        ? null
                        : () => _escolherFoto(ImageSource.gallery),
                    icon: const Icon(Icons.photo_library_outlined),
                    label: const Text('Galeria'),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            TextFormField(
              controller: _id,
              textCapitalization: TextCapitalization.characters,
              maxLength: 40,
              decoration: const InputDecoration(
                labelText: 'ID da lâmina',
                helperText: 'Padrão L-0001',
                border: OutlineInputBorder(),
              ),
              validator: (v) =>
                  (v == null || v.trim().isEmpty) ? 'Informe o ID.' : null,
            ),
            const SizedBox(height: 12),
            InkWell(
              onTap: _escolherData,
              child: InputDecorator(
                decoration: const InputDecoration(
                  labelText: 'Data',
                  border: OutlineInputBorder(),
                  suffixIcon: Icon(Icons.calendar_today_outlined),
                ),
                child: Text(formatarData(_data)),
              ),
            ),
            const SizedBox(height: 16),
            const Text(
              'Classificação (Bethesda)',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.w800),
            ),
            const SizedBox(height: 4),
            for (final c in Classificacao.values)
              Semantics(
                inMutuallyExclusiveGroup: true,
                selected: _classificacao == c,
                child: ListTile(
                  contentPadding: EdgeInsets.zero,
                  onTap: () => setState(() => _classificacao = c),
                  leading: Icon(
                    _classificacao == c
                        ? Icons.radio_button_checked
                        : Icons.radio_button_unchecked,
                    color: _classificacao == c ? c.cor : Cores.tintaFraca,
                  ),
                  title: Text(
                    c.sigla,
                    style: const TextStyle(fontWeight: FontWeight.w700),
                  ),
                  subtitle: Text(c.descricao),
                ),
              ),
            const SizedBox(height: 8),
            TextFormField(
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
            TextFormField(
              controller: _observacoes,
              minLines: 1,
              maxLines: 4,
              maxLength: 2000,
              decoration: const InputDecoration(
                labelText: 'Observações (opcional)',
                border: OutlineInputBorder(),
              ),
            ),
            if (_erro != null) ...[
              const SizedBox(height: 8),
              Text(
                _erro!,
                style: const TextStyle(color: Color(0xFF9B2C2C), height: 1.4),
              ),
            ],
            const SizedBox(height: 16),
            FilledButton.icon(
              style: FilledButton.styleFrom(minimumSize: const Size(0, 52)),
              onPressed: _salvando || _preparandoFoto ? null : _salvar,
              icon: _salvando
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2.5),
                    )
                  : const Icon(Icons.save_outlined),
              label: Text(editando ? 'Salvar alterações' : 'Cadastrar lâmina'),
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }
}
