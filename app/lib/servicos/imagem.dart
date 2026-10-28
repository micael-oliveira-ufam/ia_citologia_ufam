import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img;

const int ladoEntrada = 224;
const int ladoCam = 7;

/// Imagem já convertida no formato que o modelo espera.
class EntradaPreparada {
  EntradaPreparada({
    required this.tensor,
    required this.rgb,
    required this.larguraOriginal,
    required this.alturaOriginal,
  });

  /// Float32 em [0, 1], layout NCHW: 1 x 3 x 224 x 224.
  /// A normalização ImageNet está embutida no grafo ONNX, não aqui.
  final Float32List tensor;

  /// A mesma imagem redimensionada, em RGB entrelaçado, para desenhar o overlay.
  final Uint8List rgb;

  final int larguraOriginal;
  final int alturaOriginal;
}

/// Decodifica, redimensiona e converte para tensor. Pesado: rode em isolate.
EntradaPreparada prepararImagem(Uint8List bytes) {
  final original = img.decodeImage(bytes);
  if (original == null) {
    throw const FormatException('Não foi possível ler esta imagem.');
  }

  final redimensionada = img.copyResize(
    original,
    width: ladoEntrada,
    height: ladoEntrada,
    interpolation: img.Interpolation.linear,
  );

  final total = ladoEntrada * ladoEntrada;
  final tensor = Float32List(3 * total);
  final rgb = Uint8List(3 * total);

  for (var y = 0; y < ladoEntrada; y++) {
    for (var x = 0; x < ladoEntrada; x++) {
      final pixel = redimensionada.getPixel(x, y);
      final i = y * ladoEntrada + x;
      final r = pixel.r.toInt();
      final g = pixel.g.toInt();
      final b = pixel.b.toInt();

      rgb[i * 3] = r;
      rgb[i * 3 + 1] = g;
      rgb[i * 3 + 2] = b;

      tensor[i] = r / 255.0;                 // canal R
      tensor[total + i] = g / 255.0;         // canal G
      tensor[2 * total + i] = b / 255.0;     // canal B
    }
  }

  return EntradaPreparada(
    tensor: tensor,
    rgb: rgb,
    larguraOriginal: original.width,
    alturaOriginal: original.height,
  );
}

/// Uma das oito simetrias do quadrado (giros e rotações de 90°), aplicada ao
/// tensor. São as mesmas transformações usadas no treinamento, então a predição
/// não deveria mudar entre elas.
Float32List aplicarSimetria(Float32List tensor, int indice) {
  if (indice == 0) return tensor;

  final rotacoes = indice ~/ 2;
  final espelhar = indice.isOdd;
  final lado = ladoEntrada;
  final area = lado * lado;
  final saida = Float32List(tensor.length);

  for (var c = 0; c < 3; c++) {
    final base = c * area;
    for (var y = 0; y < lado; y++) {
      for (var x = 0; x < lado; x++) {
        var sx = x, sy = y;
        for (var r = 0; r < rotacoes; r++) {
          final tx = sx;
          sx = sy;
          sy = lado - 1 - tx;
        }
        if (espelhar) sx = lado - 1 - sx;
        saida[base + y * lado + x] = tensor[base + sy * lado + sx];
      }
    }
  }
  return saida;
}

/// Rampa de cor quente (aproximação de "inferno"): escuro para pouca
/// influência, claro para muita. Escolhida por ser legível também em escala de
/// cinza, o que ajuda quem não distingue cores.
const List<List<int>> _rampa = [
  [0, 0, 4],
  [87, 16, 110],
  [188, 55, 84],
  [249, 142, 9],
  [252, 255, 164],
];

List<int> _corDoValor(double t) {
  final v = t.clamp(0.0, 1.0) * (_rampa.length - 1);
  final i = v.floor().clamp(0, _rampa.length - 2);
  final f = v - i;
  final a = _rampa[i], b = _rampa[i + 1];
  return [
    (a[0] + (b[0] - a[0]) * f).round(),
    (a[1] + (b[1] - a[1]) * f).round(),
    (a[2] + (b[2] - a[2]) * f).round(),
  ];
}

/// Interpolação bilinear do mapa 7x7 para a resolução de exibição.
double _amostrar(Float32List mapa, double x, double y) {
  final fx = (x * (ladoCam - 1)).clamp(0.0, ladoCam - 1.0);
  final fy = (y * (ladoCam - 1)).clamp(0.0, ladoCam - 1.0);
  final x0 = fx.floor(), y0 = fy.floor();
  final x1 = math.min(x0 + 1, ladoCam - 1), y1 = math.min(y0 + 1, ladoCam - 1);
  final dx = fx - x0, dy = fy - y0;

  final v00 = mapa[y0 * ladoCam + x0], v10 = mapa[y0 * ladoCam + x1];
  final v01 = mapa[y1 * ladoCam + x0], v11 = mapa[y1 * ladoCam + x1];
  return v00 * (1 - dx) * (1 - dy) +
      v10 * dx * (1 - dy) +
      v01 * (1 - dx) * dy +
      v11 * dx * dy;
}

class PedidoSobreposicao {
  PedidoSobreposicao({
    required this.rgb,
    required this.cam,
    this.opacidade = 0.45,
  });

  final Uint8List rgb;
  final Float32List cam;
  final double opacidade;
}

/// Compõe o campo original com o mapa de calor e devolve um PNG.
/// Pesado: rode em isolate.
Uint8List gerarSobreposicao(PedidoSobreposicao pedido) {
  var maximo = 0.0;
  for (final v in pedido.cam) {
    if (v > maximo) maximo = v;
  }
  if (maximo <= 0) maximo = 1.0;

  final normalizado = Float32List(pedido.cam.length);
  for (var i = 0; i < pedido.cam.length; i++) {
    normalizado[i] = math.max(0.0, pedido.cam[i]) / maximo;
  }

  final saida = img.Image(width: ladoEntrada, height: ladoEntrada, numChannels: 3);
  for (var y = 0; y < ladoEntrada; y++) {
    for (var x = 0; x < ladoEntrada; x++) {
      final i = (y * ladoEntrada + x) * 3;
      final intensidade = _amostrar(
        normalizado,
        x / (ladoEntrada - 1),
        y / (ladoEntrada - 1),
      );
      final cor = _corDoValor(intensidade);
      // Regiões frias quase não são pintadas: o citologista continua vendo a
      // morfologia por baixo.
      final alfa = pedido.opacidade * intensidade;
      saida.setPixelRgb(
        x,
        y,
        (pedido.rgb[i] * (1 - alfa) + cor[0] * alfa).round(),
        (pedido.rgb[i + 1] * (1 - alfa) + cor[1] * alfa).round(),
        (pedido.rgb[i + 2] * (1 - alfa) + cor[2] * alfa).round(),
      );
    }
  }
  return img.encodePng(saida);
}

/// Reconstrói um PNG do campo já redimensionado, para exibir lado a lado.
Uint8List pngDoCampo(Uint8List rgb) {
  final saida = img.Image(width: ladoEntrada, height: ladoEntrada, numChannels: 3);
  for (var y = 0; y < ladoEntrada; y++) {
    for (var x = 0; x < ladoEntrada; x++) {
      final i = (y * ladoEntrada + x) * 3;
      saida.setPixelRgb(x, y, rgb[i], rgb[i + 1], rgb[i + 2]);
    }
  }
  return img.encodePng(saida);
}
