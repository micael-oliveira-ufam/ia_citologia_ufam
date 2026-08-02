import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image_picker/image_picker.dart';

import '../servicos/classificador.dart';
import '../tema.dart';
import '../widgets/comuns.dart';
import 'tela_resultado.dart';
import 'tela_sobre.dart';

/// Campos do IARC Digital Atlas que acompanham o aplicativo.
/// A descrição é morfológica, não diagnóstica: o gabarito do atlas não foi
/// declarado aqui para não induzir a leitura.
const List<({String arquivo, String titulo, String descricao, String? referencia})>
    exemplos = [
  (
    arquivo: 'assets/exemplos/cyto5940.jpg',
    titulo: 'Ectocérvice normal',
    referencia: 'NILM',
    descricao: 'Células escamosas intermediárias e superficiais, basofílicas ou '
        'eosinofílicas. Alguns polimorfonucleares. (obj. 10x)',
  ),
  (
    arquivo: 'assets/exemplos/cyt14686a.jpg',
    titulo: 'Carcinoma invasivo',
    referencia: 'SCC',
    descricao: 'Agrupamento de células malignas pleomórficas, pouco '
        'diferenciadas, e células isoladas queratinizadas com formas anômalas. '
        'Inflamação, sangue e necrose ao fundo. (obj. 20x)',
  ),
  (
    arquivo: 'assets/exemplos/cyto5950.jpg',
    titulo: 'Campo adicional 1',
    referencia: null,
    descricao: 'Escamosas maduras, núcleos pequenos, flora bacilar',
  ),
  (
    arquivo: 'assets/exemplos/cyto2870.jpg',
    titulo: 'Campo adicional 2',
    referencia: null,
    descricao: 'Colunares endocervicais em paliçada, fundo inflamatório',
  ),
  (
    arquivo: 'assets/exemplos/cyt10131a.jpg',
    titulo: 'Campo adicional 3',
    referencia: null,
    descricao: 'Halos perinucleares indicados por setas, exsudato abundante',
  ),
  (
    arquivo: 'assets/exemplos/cyt16243.jpg',
    titulo: 'Campo adicional 4',
    referencia: null,
    descricao: 'Agrupamentos densos, alta relação núcleo/citoplasma',
  ),
];

class TelaInicial extends StatefulWidget {
  const TelaInicial({super.key, required this.classificador});

  final Classificador classificador;

  @override
  State<TelaInicial> createState() => _TelaInicialState();
}

class _TelaInicialState extends State<TelaInicial> {
  final _seletor = ImagePicker();
  bool _verificarRobustez = true;
  bool _ocupado = false;

  Future<void> _abrir(Uint8List bytes, String origem, {String? referencia}) async {
    if (!mounted) return;
    await Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => TelaResultado(
          classificador: widget.classificador,
          bytes: bytes,
          origem: origem,
          verificarRobustez: _verificarRobustez,
          referencia: referencia,
        ),
      ),
    );
  }

  Future<void> _escolher(ImageSource fonte) async {
    if (_ocupado) return;
    setState(() => _ocupado = true);
    try {
      final arquivo = await _seletor.pickImage(source: fonte, imageQuality: 100);
      if (arquivo == null) return;
      final bytes = await arquivo.readAsBytes();
      await _abrir(bytes, arquivo.name);
    } catch (e) {
      _avisar('Não foi possível abrir a imagem. $e');
    } finally {
      if (mounted) setState(() => _ocupado = false);
    }
  }

  Future<void> _abrirExemplo(String caminho, String titulo, String? referencia) async {
    if (_ocupado) return;
    setState(() => _ocupado = true);
    try {
      final dados = await rootBundle.load(caminho);
      await _abrir(dados.buffer.asUint8List(), titulo, referencia: referencia);
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
    return Scaffold(
      appBar: AppBar(
        title: const Text('Citologia com IA'),
        actions: [
          IconButton(
            icon: const Icon(Icons.help_outline),
            tooltip: 'Sobre o projeto',
            onPressed: () => Navigator.of(context).push(
              MaterialPageRoute(builder: (_) => const TelaSobre()),
            ),
          ),
        ],
      ),
      body: SafeArea(
        child: ListView(
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
              'inteiramente no aparelho, sem internet, e a imagem não é enviada '
              'a lugar nenhum.',
              style: TextStyle(fontSize: 15, color: Cores.tintaFraca, height: 1.5),
            ),
            const SizedBox(height: 18),
            FilledButton.icon(
              onPressed: _ocupado ? null : () => _escolher(ImageSource.gallery),
              icon: const Icon(Icons.photo_library_outlined),
              label: const Text('Escolher da galeria'),
            ),
            const SizedBox(height: 12),
            OutlinedButton.icon(
              onPressed: _ocupado ? null : () => _escolher(ImageSource.camera),
              icon: const Icon(Icons.photo_camera_outlined),
              label: const Text('Fotografar pela ocular'),
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
            Card(
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
                      onTap: _ocupado
                          ? null
                          : () => _abrirExemplo(
                                exemplos[i].arquivo,
                                exemplos[i].titulo,
                                exemplos[i].referencia,
                              ),
                    ),
                    if (i < exemplos.length - 1) const Divider(height: 1),
                  ],
                ],
              ),
            ),
            const SizedBox(height: 24),
            const ReguaBethesda(),
          ],
        ),
      ),
    );
  }
}
