import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image_picker/image_picker.dart';

import '../servicos/classificador.dart';
import '../tema.dart';
import '../widgets/cartao_modelo.dart';
import '../widgets/comuns.dart';
import 'tela_resultado.dart';

/// Campos do IARC Digital Atlas que acompanham o aplicativo.
/// A descrição é morfológica, não diagnóstica: o gabarito do atlas não foi
/// declarado aqui para não induzir a leitura.
const List<({String arquivo, String titulo, String descricao})> exemplos = [
  (
    arquivo: 'assets/exemplos/cyto5950.jpg',
    titulo: 'Campo 1',
    descricao: 'Escamosas maduras, núcleos pequenos, flora bacilar',
  ),
  (
    arquivo: 'assets/exemplos/cyto2870.jpg',
    titulo: 'Campo 2',
    descricao: 'Colunares endocervicais em paliçada, fundo inflamatório',
  ),
  (
    arquivo: 'assets/exemplos/cyt10131a.jpg',
    titulo: 'Campo 3',
    descricao: 'Halos perinucleares indicados por setas, exsudato abundante',
  ),
  (
    arquivo: 'assets/exemplos/cyt16243.jpg',
    titulo: 'Campo 4',
    descricao: 'Agrupamentos densos, alta relação núcleo/citoplasma',
  ),
];

/// Primeira aba do profissional: escolher um campo e analisá-lo com a IA.
/// O modelo só é carregado na primeira análise (ver [Classificador.garantirPronto]).
class AbaAnalisar extends StatefulWidget {
  const AbaAnalisar({super.key, required this.classificador});

  final Classificador classificador;

  @override
  State<AbaAnalisar> createState() => _AbaAnalisarState();
}

class _AbaAnalisarState extends State<AbaAnalisar> {
  final _seletor = ImagePicker();
  bool _verificarRobustez = true;
  bool _ocupado = false;

  Future<void> _abrir(Uint8List bytes, String origem) async {
    if (!mounted) return;
    await Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => TelaResultado(
          classificador: widget.classificador,
          bytes: bytes,
          origem: origem,
          verificarRobustez: _verificarRobustez,
        ),
      ),
    );
  }

  Future<void> _escolher(ImageSource fonte) async {
    if (_ocupado) return;
    setState(() => _ocupado = true);
    try {
      final arquivo = await _seletor.pickImage(
        source: fonte,
        imageQuality: 100,
      );
      if (arquivo == null) return;
      final bytes = await arquivo.readAsBytes();
      await _abrir(bytes, arquivo.name);
    } catch (e) {
      _avisar('Não foi possível abrir a imagem. $e');
    } finally {
      if (mounted) setState(() => _ocupado = false);
    }
  }

  Future<void> _abrirExemplo(String caminho, String titulo) async {
    if (_ocupado) return;
    setState(() => _ocupado = true);
    try {
      final dados = await rootBundle.load(caminho);
      await _abrir(dados.buffer.asUint8List(), titulo);
    } catch (e) {
      _avisar('Não foi possível abrir o exemplo. $e');
    } finally {
      if (mounted) setState(() => _ocupado = false);
    }
  }

  void _avisar(String mensagem) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(mensagem), behavior: SnackBarBehavior.floating),
    );
  }

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 32),
      children: [
        const AvisoEtico(),
        const SizedBox(height: 20),
        Text(
          'Analisar um campo',
          style: Theme.of(context).textTheme.headlineSmall,
        ),
        const SizedBox(height: 6),
        const Text(
          'Escolha uma imagem de citologia em meio líquido. A análise roda '
          'inteiramente no aparelho e a imagem não é enviada a lugar nenhum. '
          'A internet só é usada para baixar o modelo, uma única vez.',
          style: TextStyle(fontSize: 15, color: Cores.tintaFraca, height: 1.5),
        ),
        const SizedBox(height: 14),
        CartaoModelo(gerenciador: widget.classificador.gerenciador),
        const SizedBox(height: 18),
        ListenableBuilder(
          listenable: widget.classificador.gerenciador,
          builder: (context, _) {
            final liberado = !_ocupado &&
                widget.classificador.gerenciador.prontoParaAnalisar;
            return Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                FilledButton.icon(
                  onPressed:
                      liberado ? () => _escolher(ImageSource.gallery) : null,
                  icon: const Icon(Icons.photo_library_outlined),
                  label: const Text('Escolher da galeria'),
                ),
                const SizedBox(height: 12),
                OutlinedButton.icon(
                  onPressed:
                      liberado ? () => _escolher(ImageSource.camera) : null,
                  icon: const Icon(Icons.photo_camera_outlined),
                  label: const Text('Fotografar pela ocular'),
                ),
              ],
            );
          },
        ),
        const SizedBox(height: 8),
        const Padding(
          padding: EdgeInsets.symmetric(horizontal: 4),
          child: Text(
            'Para fotografar pela ocular do microscópio, encoste a câmera na '
            'lente e enquadre só o campo iluminado.',
            style: TextStyle(fontSize: 13, color: Cores.tintaFraca, height: 1.4),
          ),
        ),
        const SizedBox(height: 24),
        Card(
          child: SwitchListTile.adaptive(
            value: _verificarRobustez,
            onChanged: (v) => setState(() => _verificarRobustez = v),
            title: const Text(
              'Verificar robustez',
              style: TextStyle(fontWeight: FontWeight.w600),
            ),
            subtitle: const Text(
              'Analisa o campo em oito orientações e informa se a rede muda '
              'de opinião. Deixa a análise mais lenta.',
              style: TextStyle(fontSize: 13.5, height: 1.4),
            ),
            contentPadding: const EdgeInsets.symmetric(
              horizontal: 16,
              vertical: 6,
            ),
          ),
        ),
        const SizedBox(height: 28),
        Text(
          'Campos de exemplo',
          style: Theme.of(context).textTheme.titleMedium,
        ),
        const SizedBox(height: 4),
        const Text(
          'Imagens do IARC Digital Atlas, para conhecer a ferramenta sem usar '
          'material de paciente.',
          style: TextStyle(fontSize: 14, color: Cores.tintaFraca, height: 1.45),
        ),
        const SizedBox(height: 12),
        ListenableBuilder(
          listenable: widget.classificador.gerenciador,
          builder: (context, _) {
            final liberado = !_ocupado &&
                widget.classificador.gerenciador.prontoParaAnalisar;
            return Card(
          child: Column(
            children: [
              for (var i = 0; i < exemplos.length; i++) ...[
                ListTile(
                  contentPadding: const EdgeInsets.symmetric(
                    horizontal: 14,
                    vertical: 8,
                  ),
                  leading: ClipRRect(
                    borderRadius: BorderRadius.circular(8),
                    child: Image.asset(
                      exemplos[i].arquivo,
                      width: 56,
                      height: 56,
                      fit: BoxFit.cover,
                      // Decorativa: o texto ao lado já descreve o campo.
                      excludeFromSemantics: true,
                    ),
                  ),
                  title: Text(
                    exemplos[i].titulo,
                    style: const TextStyle(fontWeight: FontWeight.w600),
                  ),
                  subtitle: Text(
                    exemplos[i].descricao,
                    style: const TextStyle(fontSize: 13.5, height: 1.35),
                  ),
                  trailing: const Icon(Icons.chevron_right),
                  onTap: !liberado
                      ? null
                      : () => _abrirExemplo(
                            exemplos[i].arquivo,
                            exemplos[i].titulo,
                          ),
                ),
                if (i < exemplos.length - 1) const Divider(height: 1),
              ],
            ],
          ),
            );
          },
        ),
        const SizedBox(height: 24),
        const ReguaBethesda(),
      ],
    );
  }
}
