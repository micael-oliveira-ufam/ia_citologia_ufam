#!/usr/bin/env python3
"""
Exporta o ConvNeXt treinado (.pt) para ONNX, pronto para rodar offline no Android.

O objetivo é que o aparelho e a plataforma web produzam EXATAMENTE a mesma
classificação para a mesma imagem. Por isso o grafo exportado replica o
pré-processamento da web (Resize 224 bilinear + Normalize ImageNet) e o script
confere numericamente o resultado contra o PyTorch antes de gravar o arquivo.

O modelo exportado devolve DUAS saídas:
  logits  [1, 4]: escores por classe, na ordem CLASS_NAMES
  cam     [1, 4, 7, 7]: mapa de ativação por classe

Sobre o mapa de calor: o grafo calcula o Grad-CAM em forma fechada, sem
autograd, propagando o gradiente analiticamente pela LayerNorm da cabeça. O
resultado é idêntico ao Grad-CAM que a plataforma web mostra, e o script mede
essa correlação a cada exportação. Ver a docstring de ConvNextExportavel.

Uso:
    python3 tools/export_onnx.py \
        --checkpoint convnext_liquid_based_citology_IARC_digital_atlas_01_08_26.pt \
        --saida assets/model/citologia_convnext.onnx \
        --exemplos assets/exemplos
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models

# ---------------------------------------------------------------------------
# PARÂMETROS COMPARTILHADOS COM A PLATAFORMA WEB
# Qualquer alteração aqui precisa ser espelhada em app.py (web) e em
# lib/data/bethesda.dart + lib/servicos/imagem.dart (Android).
# ---------------------------------------------------------------------------
CLASS_NAMES = ["SCC", "HSIL", "LSIL", "NILM"]
LADO = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Carregamento do checkpoint
# ---------------------------------------------------------------------------

def detectar_variante(sd):
    stem = sd.get("features.0.0.weight")
    if stem is None:
        return "convnext_large"
    largura = stem.shape[0]
    if largura == 192:
        return "convnext_large"
    if largura == 128:
        return "convnext_base"
    blocos = {int(k.split(".")[2]) for k in sd if k.startswith("features.5.")}
    return "convnext_small" if len(blocos) > 12 else "convnext_tiny"


def cabeca_com_dropout(sd):
    """O treinamento usou classifier[2] = Sequential(Dropout, Linear)."""
    return any(k.startswith("classifier.2.1.") for k in sd)


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
    com_dropout = cabeca_com_dropout(sd)

    modelo = getattr(models, variante)(weights=None)
    in_features = modelo.classifier[2].in_features
    if com_dropout:
        modelo.classifier[2] = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(in_features, len(CLASS_NAMES))
        )
    else:
        modelo.classifier[2] = nn.Linear(in_features, len(CLASS_NAMES))

    faltando, sobrando = modelo.load_state_dict(sd, strict=False)
    if faltando or sobrando:
        print(f"  aviso: {len(faltando)} pesos ausentes, {len(sobrando)} inesperados")
    modelo.eval()
    return modelo, variante, com_dropout


# ---------------------------------------------------------------------------
# Módulo exportável
# ---------------------------------------------------------------------------

class CabecaCAM(nn.Module):
    """Calcula o Grad-CAM em forma fechada, sem autograd.

    Fica em um submódulo próprio por um motivo prático: o exportador nomeia os
    nós ONNX com o caminho do módulo, então todos os nós daqui saem com o
    prefixo "/cabeca_cam/". Isso permite excluí-los da quantização pelo nome,
    sem depender de heurística sobre formas de tensores.

    Os pesos combinados u = gamma * W são constantes e ficam guardados como
    buffer. Assim o ramo do mapa de calor não compartilha o tensor de pesos com
    a camada linear da classificação: quando a quantização reescreve a camada
    linear, ela não arrasta junto o ramo do CAM.
    """

    def __init__(self, gamma, pesos_lineares, eps):
        super().__init__()
        u = (gamma.detach().unsqueeze(0) * pesos_lineares.detach()).contiguous()
        self.register_buffer("u", u)                            # [K, C]
        self.register_buffer("u_medio", u.mean(dim=1, keepdim=True))   # [K, 1]
        self.register_buffer("u_transposto", u.t().contiguous())       # [C, K]
        self.eps = float(eps)
        self.num_canais = int(u.shape[1])

    def forward(self, f, p):
        """f: mapa de características [B, C, H, W]. p: vetor agrupado [B, C]."""
        media = p.mean(dim=1, keepdim=True)
        var = p.var(dim=1, unbiased=False, keepdim=True)
        sigma = torch.sqrt(var + self.eps)                      # [B, 1]
        z = (p - media) / sigma                                 # [B, C]

        uz = torch.matmul(z, self.u_transposto) / self.num_canais   # [B, K]
        pesos = (
            self.u.unsqueeze(0) - self.u_medio.unsqueeze(0)
            - uz.unsqueeze(2) * z.unsqueeze(1)
        ) / sigma.unsqueeze(1)                                  # [B, K, C]

        b, c, h, w = f.shape
        # matmul em vez de einsum: o einsum com peso constante era reescrito
        # pela quantização dinâmica e quebrava o grafo int8.
        return torch.matmul(pesos, f.reshape(b, c, h * w)).reshape(b, -1, h, w)


class ConvNextExportavel(nn.Module):
    """ConvNeXt com normalização embutida e Grad-CAM como segunda saída.

    O mapa de calor é calculado em forma fechada, sem autograd, de modo a
    reproduzir exatamente o Grad-CAM que a plataforma web mostra.

    A cabeça do ConvNeXt é
        AdaptiveAvgPool2d -> LayerNorm2d -> Flatten -> (Dropout) -> Linear
    Se fosse só pooling seguido de camada linear, o gradiente do logit k em
    relação ao mapa de características seria constante no espaço e igual a
    w_k/(H·W), e bastaria projetar o mapa pelos pesos. A LayerNorm no meio
    quebra essa simplificação: ela é não linear, e seu jacobiano depende das
    estatísticas do vetor agrupado.

    Escrevendo p para o vetor agrupado, mu e sigma para média e desvio sobre os
    canais, z = (p - mu)/sigma e u_k = gamma * w_k, o gradiente vale

        d logit_k / dp = (1/sigma) * [ u_k - media(u_k) - media(u_k * z) * z ]

    que é aritmética simples e cabe no grafo. O fator 1/(H·W) some na
    normalização pelo máximo, então é omitido. O resultado bate com o Grad-CAM
    do PyTorch na precisão do float32; o próprio script mede isso.

    Importante: a projeção usa o mapa de características BRUTO, não o
    normalizado. Uma versão anterior deste exportador projetava o mapa
    normalizado pelos pesos lineares, o que só correlacionava cerca de 0,87 com
    o Grad-CAM real.
    """

    def __init__(self, base, com_dropout):
        super().__init__()
        self.features = base.features
        self.avgpool = base.avgpool
        self.norm = base.classifier[0]                      # LayerNorm2d
        self.achatar = base.classifier[1]                   # Flatten
        # A cabeça pode ser Linear puro ou Sequential(Dropout, Linear).
        # Dropout é identidade em eval, então basta pegar a camada linear.
        self.fc = base.classifier[2][1] if com_dropout else base.classifier[2]
        self.cabeca_cam = CabecaCAM(
            self.norm.weight, self.fc.weight, getattr(self.norm, "eps", 1e-6)
        )
        self.register_buffer("media_img", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("desvio_img", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, imagem):
        x = (imagem - self.media_img) / self.desvio_img
        f = self.features(x)                                # [B, C, H, W]
        agrupado = self.avgpool(f)                          # [B, C, 1, 1]
        logits = self.fc(self.achatar(self.norm(agrupado)))
        cam = self.cabeca_cam(f, self.achatar(agrupado))
        return logits, cam


# ---------------------------------------------------------------------------
# Referência: Grad-CAM como a plataforma web calcula
# ---------------------------------------------------------------------------

def grad_cam_referencia(modelo, entrada, indice):
    """Reproduz o Grad-CAM do app.py, para comparar com o CAM do grafo.

    O gradiente é capturado por um hook no TENSOR de ativação, não por
    register_full_backward_hook no módulo. A diferença importa: um hook de
    módulo que devolve algo substitui o grad_input, e no último bloco do
    ConvNeXt o grad_input tem forma diferente do grad_output, o que faz o
    autograd abortar com "hook has changed the size of value". Hook de tensor
    não tem esse contrato e devolve exatamente o gradiente daquela ativação.
    """
    guardado = {}
    alvo = modelo.features[-1][-1]

    def ao_passar(_modulo, _entrada, saida):
        guardado["ativacao"] = saida
        if saida.requires_grad:
            saida.register_hook(lambda g: guardado.__setitem__("gradiente", g))
        # Devolver None é obrigatório: qualquer retorno substitui a saída.
        return None

    alca = alvo.register_forward_hook(ao_passar)
    try:
        modelo.zero_grad(set_to_none=True)
        saida = modelo(entrada)
        saida[0, indice].backward()

        ativacao = guardado.get("ativacao")
        gradiente = guardado.get("gradiente")
        if ativacao is None or gradiente is None:
            raise RuntimeError(
                "não consegui capturar ativação e gradiente da camada alvo"
            )

        pesos = gradiente.mean(dim=(2, 3), keepdim=True)
        mapa = F.relu((pesos * ativacao).sum(dim=1)).squeeze(0)
    finally:
        alca.remove()

    maximo = mapa.max()
    if maximo > 0:
        mapa = mapa / maximo
    return mapa.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Verificação
# ---------------------------------------------------------------------------

def preparar(caminho_img):
    """Mesmo pré-processamento da web: redimensiona para 224 e escala a [0,1]."""
    img = Image.open(caminho_img).convert("RGB").resize((LADO, LADO), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))[None]


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def conferir(caminho_onnx, modelo_torch, imagens, titulo):
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime não instalado: verificação numérica pulada")
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

    print(f"  {titulo}: desvio máximo de probabilidade {maior_desvio:.5f}, "
          f"{divergencias} de {len(imagens)} classificações divergentes")
    return divergencias == 0


def conferir_mapa_de_calor(caminho_onnx, base, imagens):
    """Compara o CAM do grafo com o Grad-CAM que a web calcula."""
    try:
        import onnxruntime as ort
    except ImportError:
        return

    sessao = ort.InferenceSession(caminho_onnx, providers=["CPUExecutionProvider"])
    nome_entrada = sessao.get_inputs()[0].name
    correlacoes = []

    for caminho in imagens:
        entrada = preparar(caminho)
        # A web normaliza fora do grafo; aqui o grafo já normaliza, então o
        # Grad-CAM de referência recebe a entrada normalizada à mão.
        media = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
        desvio = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)
        tensor = torch.from_numpy((entrada - media) / desvio)

        logits, cam = sessao.run(None, {nome_entrada: entrada})
        idx = int(np.argmax(logits[0]))

        mapa_onnx = np.maximum(cam[0, idx], 0)
        if mapa_onnx.max() > 0:
            mapa_onnx = mapa_onnx / mapa_onnx.max()
        mapa_web = grad_cam_referencia(base, tensor, idx)

        a, b = mapa_onnx.ravel(), mapa_web.ravel()
        if a.std() > 1e-8 and b.std() > 1e-8:
            correlacoes.append(float(np.corrcoef(a, b)[0, 1]))

    if correlacoes:
        print(f"  mapa de calor: correlação com o Grad-CAM da web "
              f"{np.mean(correlacoes):.6f} (mínimo {np.min(correlacoes):.6f})")
        if np.min(correlacoes) < 0.99:
            print("  ATENÇÃO: a correlação deveria ser praticamente 1. Verifique se "
                  "a cabeça do modelo é a esperada (avgpool, LayerNorm2d, Linear).")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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

    print(f"[1/5] Carregando {args.checkpoint}")
    base, variante, com_dropout = carregar(args.checkpoint)
    print(f"  arquitetura: {variante}")
    print(f"  cabeça: {'Sequential(Dropout, Linear)' if com_dropout else 'Linear'}")

    modelo = ConvNextExportavel(base, com_dropout).eval()
    exemplo = torch.zeros(1, 3, LADO, LADO)
    with torch.no_grad():
        logits, cam = modelo(exemplo)
    print(f"  saídas: logits {tuple(logits.shape)}, cam {tuple(cam.shape)}")

    # Sanidade: o módulo exportável tem de reproduzir o modelo original.
    entrada_aleatoria = torch.rand(1, 3, LADO, LADO)
    media = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    desvio = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    with torch.no_grad():
        esperado = base((entrada_aleatoria - media) / desvio)
        obtido = modelo(entrada_aleatoria)[0]
    delta = float((esperado - obtido).abs().max())
    print(f"  diferença máxima em relação ao modelo original: {delta:.2e}")
    if delta > 1e-4:
        sys.exit("  ERRO: o módulo exportável não reproduz o modelo original.")

    print(f"[2/5] Exportando para ONNX (opset {args.opset})")
    caminho_fp32 = args.saida.replace(".onnx", "_fp32.onnx")
    # dynamo=False é deliberado: o exportador baseado em TorchScript produz um
    # grafo mais previsível para o einsum do CAM e é o que o ONNX Runtime móvel
    # consome sem sobressaltos. O aviso de depreciação é esperado.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
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
            onnx.StringStringEntryProto(key="entrada", value=f"RGB [0,1], NCHW, 1x3x{LADO}x{LADO}"),
            onnx.StringStringEntryProto(key="arquitetura", value=variante),
            onnx.StringStringEntryProto(key="checkpoint", value=os.path.basename(args.checkpoint)),
            onnx.StringStringEntryProto(key="dados", value="IARC Digital Atlas (público)"),
            onnx.StringStringEntryProto(key="uso", value="Academico. Sem aprovacao de CEP."),
        ])
        onnx.checker.check_model(m)
        onnx.save(m, caminho_fp32)
        print("  grafo validado e metadados gravados")
    except ImportError:
        print("  pacote onnx não instalado: validação do grafo pulada")

    imagens = []
    if os.path.isdir(args.exemplos):
        imagens = sorted(
            os.path.join(args.exemplos, f) for f in os.listdir(args.exemplos)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
        )

    print("[3/5] Conferindo o ONNX float32 contra o PyTorch")
    if imagens:
        conferir(caminho_fp32, modelo, imagens, "float32")
        # A comparação do mapa de calor é diagnóstico auxiliar. Se falhar, o
        # modelo exportado continua válido: a classificação já foi conferida
        # acima, e é ela que precisa bater com a plataforma web.
        try:
            conferir_mapa_de_calor(caminho_fp32, base, imagens)
        except Exception as erro:
            print(f"  não foi possível comparar o mapa de calor: {erro}")
            print("  a exportação segue; a classificação já foi validada.")
    else:
        print("  nenhuma imagem em --exemplos; conferência pulada")

    if args.sem_quantizar:
        print("[4/5] Quantização desativada por opção")
        os.replace(caminho_fp32, args.saida)
    else:
        print("[4/5] Quantizando os pesos para int8")
        quantizou = False
        caminho_int8 = args.saida.replace(".onnx", "_int8.onnx")
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            # O ramo do mapa de calor fica fora da quantização. Ele tem poucos
            # parâmetros, então não há ganho de tamanho, e seus matmuls com peso
            # constante são justamente os que o quantizador reescreve de forma
            # incompatível. Os nós são identificados pelo prefixo do submódulo.
            excluir = []
            try:
                import onnx
                grafo = onnx.load(caminho_fp32).graph
                excluir = [
                    no.name for no in grafo.node
                    if no.name and "cabeca_cam" in no.name
                ]
                print(f"  {len(excluir)} nós do ramo do mapa de calor preservados em float32")
            except Exception:
                print("  não consegui listar os nós do CAM; seguindo sem exclusões")

            quantize_dynamic(
                caminho_fp32, caminho_int8,
                weight_type=QuantType.QUInt8,
                nodes_to_exclude=excluir or None,
                extra_options={"MatMulConstBOnly": True},
            )

            # A quantização pode reescrever nós e quebrar o grafo. Só ficamos
            # com o int8 se ele rodar E concordar com o PyTorch: um modelo menor
            # que classifica diferente da plataforma web não serve de nada.
            if imagens:
                if conferir(caminho_int8, modelo, imagens, "int8"):
                    conferir_mapa_de_calor(caminho_int8, base, imagens)
                    quantizou = True
                else:
                    print("  a quantização mudou pelo menos uma classificação.")
            else:
                quantizou = True
                print("  sem imagens para conferir; aceitando o int8 sem validação")
        except ImportError:
            print("  onnxruntime.quantization indisponível")
        except Exception as erro:
            print(f"  a quantização falhou: {erro}")

        if quantizou:
            os.replace(caminho_int8, args.saida)
            os.remove(caminho_fp32)
            print("  int8 validado e adotado")
        else:
            print("  mantendo o float32, que já foi conferido contra o PyTorch.")
            print("  O arquivo fica maior, e a paridade com a web é o que importa.")
            if os.path.exists(caminho_int8):
                os.remove(caminho_int8)
            os.replace(caminho_fp32, args.saida)

    print("[5/5] Concluído")
    mb = os.path.getsize(args.saida) / 1_048_576
    print(f"\nPronto: {args.saida}  ({mb:.1f} MB)")
    if mb > 150:
        print(
            "\nAviso de tamanho. O modelo tem "
            f"{mb:.0f} MB e o APK ficará maior que isso.\n"
            "A Play Store limita o APK a 100 MB (até 200 MB com expansão), então um\n"
            "modelo deste porte serve para instalação manual, não para publicação\n"
            "direta na loja. Três caminhos:\n"
            "  1. distribuir o APK por download direto, que é o uso previsto agora;\n"
            "  2. baixar o modelo na primeira execução em vez de embutir, o que\n"
            "     preserva a paridade mas exige internet uma única vez;\n"
            "  3. treinar uma variante menor do ConvNeXt para o aparelho, ciente de\n"
            "     que isso quebra a equivalência com a plataforma web e obriga a\n"
            "     revalidar o modelo do aparelho por conta própria.\n"
            "Meça também o tempo de inferência num aparelho representativo: um\n"
            "ConvNeXt Large em CPU de celular pode passar de vários segundos por campo."
        )


if __name__ == "__main__":
    main()
