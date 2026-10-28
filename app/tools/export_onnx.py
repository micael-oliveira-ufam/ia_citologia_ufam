#!/usr/bin/env python3
"""
Exporta o ConvNeXt treinado (.pt) para ONNX, pronto para rodar offline no Android.

O modelo exportado devolve DUAS saídas:
  logits  [1, 4]      — escores por classe
  cam     [1, 4, 7, 7] — mapa de ativação por classe (Class Activation Map)

O CAM é calculado dentro do grafo, sem autograd: a cabeça do ConvNeXt do
torchvision é AdaptiveAvgPool2d -> LayerNorm2d -> Linear, então projetar o mapa
de características normalizado pelos pesos da camada linear dá exatamente o CAM
da classe. Isso permite mostrar o mapa de calor no celular só com passagem
direta — nada de backpropagation em dispositivo.

A normalização ImageNet fica embutida no grafo. A entrada esperada é RGB em
[0, 1], layout NCHW, 1x3x224x224 — assim não há risco de o pré-processamento do
Dart divergir do treino.

Uso:
    python tools/export_onnx.py \
        --checkpoint convnext_liquid_based_citology_IARC_digital_atlas_06_05_26.pt \
        --saida assets/model/citologia_convnext.onnx \
        --exemplos assets/exemplos
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

CLASS_NAMES = ["SCC", "HSIL", "LSIL", "NILM"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Carregamento do checkpoint
# ---------------------------------------------------------------------------

def detectar_variante(sd):
    stem = sd.get("features.0.0.weight")
    if stem is None:
        return "convnext_tiny"
    largura = stem.shape[0]
    if largura == 128:
        return "convnext_base"
    if largura == 192:
        return "convnext_large"
    blocos = {int(k.split(".")[2]) for k in sd if k.startswith("features.5.")}
    return "convnext_small" if len(blocos) > 12 else "convnext_tiny"


def extrair_state_dict(objeto):
    sd = objeto
    if isinstance(objeto, dict) and not any(torch.is_tensor(v) for v in objeto.values()):
        for chave in ("state_dict", "model_state_dict", "model"):
            if chave in objeto and isinstance(objeto[chave], dict):
                sd = objeto[chave]
                break
    return {k.replace("module.", "", 1): v for k, v in sd.items()}


def carregar(caminho):
    try:
        bruto = torch.load(caminho, map_location="cpu", weights_only=True)
    except Exception:
        bruto = torch.load(caminho, map_location="cpu", weights_only=False)
    sd = extrair_state_dict(bruto)
    variante = detectar_variante(sd)
    modelo = getattr(models, variante)(weights=None)
    entradas = modelo.classifier[2].in_features
    if "classifier.2.1.weight" in sd:
        # Cabeça treinada como Sequential(Dropout, Linear): os pesos ficam em
        # classifier.2.1.*. Sem isto, a cabeça ficaria aleatória.
        modelo.classifier[2] = nn.Sequential(nn.Dropout(0.5), nn.Linear(entradas, len(CLASS_NAMES)))
    else:
        modelo.classifier[2] = nn.Linear(entradas, len(CLASS_NAMES))
    faltando, sobrando = modelo.load_state_dict(sd, strict=False)
    if faltando or sobrando:
        raise SystemExit(
            f"ERRO: o checkpoint não casa com a arquitetura ({len(faltando)} pesos ausentes: "
            f"{list(faltando)[:3]}; {len(sobrando)} inesperados: {list(sobrando)[:3]}). "
            "Exportar assim geraria um modelo com pesos aleatórios.")
    modelo.eval()
    return modelo, variante


# ---------------------------------------------------------------------------
# Módulo exportável
# ---------------------------------------------------------------------------

class ConvNextExportavel(nn.Module):
    """ConvNeXt com normalização embutida e CAM como segunda saída."""

    def __init__(self, base):
        super().__init__()
        self.features = base.features
        self.avgpool = base.avgpool
        self.norm = base.classifier[0]   # LayerNorm2d
        cabeca = base.classifier[2]
        self.fc = cabeca[-1] if isinstance(cabeca, nn.Sequential) else cabeca  # Linear
        self.register_buffer("media", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("desvio", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, imagem):
        x = (imagem - self.media) / self.desvio
        f = self.features(x)                              # [B, C, 7, 7]
        logits = self.fc(torch.flatten(self.norm(self.avgpool(f)), 1))
        # MatMul (e não einsum): a quantização int8 do onnxruntime trata MatMul
        # com peso constante, mas quebra o einsum.
        cam = torch.matmul(self.norm(f).permute(0, 2, 3, 1), self.fc.weight.t())
        cam = cam.permute(0, 3, 1, 2)
        return logits, cam


# ---------------------------------------------------------------------------
# Verificação
# ---------------------------------------------------------------------------

def preparar(caminho_img):
    img = Image.open(caminho_img).convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))[None]


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def conferir(caminho_onnx, modelo_torch, imagens, titulo):
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime não instalado — verificação numérica pulada")
        return True
    sessao = ort.InferenceSession(caminho_onnx, providers=["CPUExecutionProvider"])
    nome_entrada = sessao.get_inputs()[0].name
    maior_desvio, divergencias = 0.0, 0
    for caminho in imagens:
        entrada = preparar(caminho)
        with torch.no_grad():
            ref = softmax(modelo_torch(torch.from_numpy(entrada))[0].numpy())[0]
        obtido = softmax(sessao.run(None, {nome_entrada: entrada})[0])[0]
        maior_desvio = max(maior_desvio, float(np.abs(ref - obtido).max()))
        if int(ref.argmax()) != int(obtido.argmax()):
            divergencias += 1
            print(f"  DIVERGÊNCIA em {os.path.basename(caminho)}: "
                  f"PyTorch={CLASS_NAMES[int(ref.argmax())]} "
                  f"ONNX={CLASS_NAMES[int(obtido.argmax())]}")
    print(f"  {titulo}: desvio máximo de probabilidade {maior_desvio:.4f}, "
          f"{divergencias} de {len(imagens)} classificações divergentes")
    return divergencias == 0


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="arquivo .pt do treinamento")
    ap.add_argument("--saida", default="assets/model/citologia_convnext.onnx")
    ap.add_argument("--exemplos", default="assets/exemplos",
                    help="pasta com imagens para conferir o modelo exportado")
    ap.add_argument("--sem-quantizar", action="store_true",
                    help="mantém float32 (arquivo ~4x maior, sem perda numérica)")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        sys.exit(f"checkpoint não encontrado: {args.checkpoint}")
    os.makedirs(os.path.dirname(args.saida) or ".", exist_ok=True)

    print(f"[1/4] Carregando {args.checkpoint}")
    base, variante = carregar(args.checkpoint)
    print(f"  arquitetura: {variante}")

    modelo = ConvNextExportavel(base).eval()
    exemplo = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        logits, cam = modelo(exemplo)
    print(f"  saídas: logits {tuple(logits.shape)}, cam {tuple(cam.shape)}")

    print(f"[2/4] Exportando para ONNX (opset {args.opset})")
    caminho_fp32 = args.saida.replace(".onnx", "_fp32.onnx")
    torch.onnx.export(
        modelo, exemplo, caminho_fp32,
        input_names=["imagem"], output_names=["logits", "cam"],
        opset_version=args.opset, do_constant_folding=True, dynamo=False,
    )

    try:
        import onnx
        m = onnx.load(caminho_fp32)
        m.metadata_props.extend([
            onnx.StringStringEntryProto(key="classes", value=json.dumps(CLASS_NAMES)),
            onnx.StringStringEntryProto(key="entrada", value="RGB [0,1], NCHW, 1x3x224x224"),
            onnx.StringStringEntryProto(key="arquitetura", value=variante),
            onnx.StringStringEntryProto(key="dados", value="IARC Digital Atlas (público)"),
            onnx.StringStringEntryProto(key="uso", value="Academico. Sem aprovacao de CEP."),
        ])
        onnx.checker.check_model(m)
        onnx.save(m, caminho_fp32)
        print("  grafo validado e metadados gravados")
    except ImportError:
        print("  pacote onnx não instalado — validação do grafo pulada")

    imagens = []
    if os.path.isdir(args.exemplos):
        imagens = sorted(
            os.path.join(args.exemplos, f) for f in os.listdir(args.exemplos)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
        )

    print("[3/4] Conferindo o ONNX float32 contra o PyTorch")
    if imagens:
        conferir(caminho_fp32, modelo, imagens, "float32")
    else:
        print("  nenhuma imagem em --exemplos; conferência pulada")

    if args.sem_quantizar:
        os.replace(caminho_fp32, args.saida)
    else:
        print("[4/4] Quantizando os pesos para int8")
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
            quantize_dynamic(caminho_fp32, args.saida,
                             weight_type=QuantType.QUInt8, extra_options={"MatMulConstBOnly": True})
            if imagens:
                ok = conferir(args.saida, modelo, imagens, "int8")
                if not ok:
                    print("  ATENÇÃO: a quantização mudou pelo menos uma classificação.\n"
                          "  Rode com --sem-quantizar e valide o int8 no conjunto completo "
                          "antes de publicar o APK.")
            os.remove(caminho_fp32)
        except ImportError:
            print("  onnxruntime.quantization indisponível — mantendo float32")
            os.replace(caminho_fp32, args.saida)

    mb = os.path.getsize(args.saida) / 1_048_576
    print(f"\nPronto: {args.saida}  ({mb:.1f} MB)")
    if mb > 95:
        print("Aviso: acima de 95 MB o APK fica desconfortável para distribuição.\n"
              "Considere uma variante menor do ConvNeXt ou quantização estática.")


if __name__ == "__main__":
    main()
