#!/usr/bin/env python3
"""
Gera o ícone do CitoPred: um microscópio desenhado vetorialmente e rasterizado
nas densidades que o Android pede.

Produz:
  assets/icone/icone_1024.png              mestre, para lojas e material gráfico
  assets/icone/icone_adaptativo_frente.png camada de frente do ícone adaptativo
  android/app/src/main/res/mipmap-*/ic_launcher.png

O desenho é feito com PIL, sem dependência de conversor de SVG, para que o
script rode em qualquer máquina que já tenha o ambiente do projeto.

Uso:
    python3 tools/gerar_icone.py
"""

import math
import os

from PIL import Image, ImageDraw

# Paleta compartilhada com a interface
TEAL = (14, 107, 123)
TEAL_ESCURO = (9, 74, 86)
TEAL_CLARO = (228, 241, 243)
PORCELANA = (244, 247, 249)
BRANCO = (255, 255, 255)
LENTE = (176, 219, 226)

LADO = 1024

# Densidades do Android e o tamanho do ícone em cada uma
MIPMAPS = {
    "mdpi": 48,
    "hdpi": 72,
    "xhdpi": 96,
    "xxhdpi": 144,
    "xxxhdpi": 192,
}


def _rotacionar(pontos, angulo_graus, centro):
    """Gira uma lista de pontos em torno de um centro."""
    rad = math.radians(angulo_graus)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    cx, cy = centro
    saida = []
    for x, y in pontos:
        dx, dy = x - cx, y - cy
        saida.append((cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a))
    return saida


def desenhar_microscopio(lado=LADO, com_fundo=True, escala_conteudo=1.0):
    """Desenha o microscópio. Devolve uma imagem RGBA quadrada."""
    # Trabalha em 4x e reduz no final: é o jeito mais simples de conseguir
    # bordas suaves sem depender de antialiasing do PIL.
    s = lado * 4
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    if com_fundo:
        raio = int(s * 0.22)
        d.rounded_rectangle([0, 0, s, s], radius=raio, fill=PORCELANA)

    # Sistema de coordenadas do desenho: 0..1 sobre o lado, centralizado
    cx, cy = s / 2, s / 2
    k = s * escala_conteudo

    def P(x, y):
        """Converte coordenadas relativas (centro na origem) em pixels."""
        return (cx + x * k, cy + y * k)

    def caixa(x0, y0, x1, y1):
        return [P(x0, y0), P(x1, y1)]

    # ---- Base do microscópio ------------------------------------------
    d.rounded_rectangle(caixa(-0.30, 0.28, 0.30, 0.37), radius=k * 0.045,
                        fill=TEAL_ESCURO)
    # pé inclinado, ligando a base à coluna
    d.polygon([P(-0.20, 0.29), P(0.18, 0.29), P(0.12, 0.17), P(-0.12, 0.17)],
              fill=TEAL_ESCURO)

    # ---- Platina (onde a lâmina fica) ---------------------------------
    d.rounded_rectangle(caixa(-0.30, 0.04, 0.22, 0.10), radius=k * 0.018,
                        fill=TEAL)
    # a lâmina em si, em destaque claro
    d.rounded_rectangle(caixa(-0.25, 0.055, -0.02, 0.085), radius=k * 0.008,
                        fill=TEAL_CLARO)

    # ---- Coluna / braço ------------------------------------------------
    d.rounded_rectangle(caixa(0.10, -0.30, 0.22, 0.10), radius=k * 0.055,
                        fill=TEAL)

    # ---- Botões de foco -------------------------------------------------
    d.ellipse([P(0.20, -0.06)[0] - k * 0.055, P(0.20, -0.06)[1] - k * 0.055,
               P(0.20, -0.06)[0] + k * 0.055, P(0.20, -0.06)[1] + k * 0.055],
              fill=TEAL_ESCURO)
    d.ellipse([P(0.20, -0.06)[0] - k * 0.022, P(0.20, -0.06)[1] - k * 0.022,
               P(0.20, -0.06)[0] + k * 0.022, P(0.20, -0.06)[1] + k * 0.022],
              fill=TEAL_CLARO)

    # ---- Tubo e ocular, inclinados -------------------------------------
    centro_tubo = P(0.02, -0.20)
    tubo = [P(-0.16, -0.30), P(0.16, -0.30), P(0.16, -0.11), P(-0.16, -0.11)]
    tubo = _rotacionar(tubo, -22, centro_tubo)
    d.polygon(tubo, fill=TEAL)

    ocular = [P(-0.30, -0.31), P(-0.15, -0.31), P(-0.15, -0.20), P(-0.30, -0.20)]
    ocular = _rotacionar(ocular, -22, centro_tubo)
    d.polygon(ocular, fill=TEAL_ESCURO)

    # ---- Objetiva, apontando para a lâmina -----------------------------
    objetiva = [P(-0.055, -0.10), P(0.055, -0.10), P(0.030, 0.02), P(-0.030, 0.02)]
    d.polygon(objetiva, fill=TEAL_ESCURO)

    # ---- Campo iluminado sob a objetiva --------------------------------
    r = k * 0.052
    centro_luz = P(0.0, 0.068)
    d.ellipse([centro_luz[0] - r, centro_luz[1] - r,
               centro_luz[0] + r, centro_luz[1] + r], fill=BRANCO)

    # três células dentro do campo: a leitura citológica em miniatura
    for dx, dy, rr in ((-0.016, -0.004, 0.016), (0.014, 0.010, 0.013),
                       (0.006, -0.018, 0.010)):
        c = P(dx, 0.068 + dy)
        rp = k * rr
        d.ellipse([c[0] - rp, c[1] - rp, c[0] + rp, c[1] + rp], fill=LENTE)
        rn = rp * 0.42
        d.ellipse([c[0] - rn, c[1] - rn, c[0] + rn, c[1] + rn], fill=TEAL)

    return img.resize((lado, lado), Image.LANCZOS)


def main():
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pasta_icone = os.path.join(raiz, "assets", "icone")
    os.makedirs(pasta_icone, exist_ok=True)

    print("[1/3] Desenhando o ícone mestre")
    mestre = desenhar_microscopio(LADO, com_fundo=True, escala_conteudo=1.0)
    caminho_mestre = os.path.join(pasta_icone, "icone_1024.png")
    mestre.save(caminho_mestre)
    print(f"  {caminho_mestre}")

    # Ícone adaptativo: o Android recorta a camada de frente num círculo ou
    # quadrado arredondado, então o conteúdo precisa caber na zona segura
    # central, que é cerca de 66% do lado.
    print("[2/3] Gerando a camada de frente do ícone adaptativo")
    frente = desenhar_microscopio(LADO, com_fundo=False, escala_conteudo=0.62)
    caminho_frente = os.path.join(pasta_icone, "icone_adaptativo_frente.png")
    frente.save(caminho_frente)
    print(f"  {caminho_frente}")

    print("[3/3] Exportando as densidades do Android")
    base_res = os.path.join(raiz, "android", "app", "src", "main", "res")
    if not os.path.isdir(base_res):
        print("  pasta android/ ainda não existe; rode o build_apk.sh primeiro.")
        print("  Os PNGs mestres já estão em assets/icone/.")
        return

    for densidade, tamanho in MIPMAPS.items():
        destino = os.path.join(base_res, f"mipmap-{densidade}")
        os.makedirs(destino, exist_ok=True)
        redimensionado = mestre.resize((tamanho, tamanho), Image.LANCZOS)
        redimensionado.save(os.path.join(destino, "ic_launcher.png"))
        frente.resize((tamanho, tamanho), Image.LANCZOS).save(
            os.path.join(destino, "ic_launcher_foreground.png")
        )
        print(f"  mipmap-{densidade}: {tamanho}x{tamanho}")

    # Cor de fundo do ícone adaptativo
    valores = os.path.join(base_res, "values")
    os.makedirs(valores, exist_ok=True)
    with open(os.path.join(valores, "ic_launcher_background.xml"), "w",
              encoding="utf-8") as arq:
        arq.write(
            '<?xml version="1.0" encoding="utf-8"?>\n'
            "<resources>\n"
            '    <color name="ic_launcher_background">#F4F7F9</color>\n'
            "</resources>\n"
        )

    anydpi = os.path.join(base_res, "mipmap-anydpi-v26")
    os.makedirs(anydpi, exist_ok=True)
    xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<adaptive-icon xmlns:android="http://schemas.android.com/apk/res/android">\n'
        '    <background android:drawable="@color/ic_launcher_background" />\n'
        '    <foreground android:drawable="@mipmap/ic_launcher_foreground" />\n'
        "</adaptive-icon>\n"
    )
    for nome in ("ic_launcher.xml", "ic_launcher_round.xml"):
        with open(os.path.join(anydpi, nome), "w", encoding="utf-8") as arq:
            arq.write(xml)
    print("  ícone adaptativo configurado")

    print("\nPronto.")


if __name__ == "__main__":
    main()
