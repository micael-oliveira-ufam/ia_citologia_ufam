#!/usr/bin/env python3
"""
Ajusta os esqueletos nativos gerados por `flutter create` (android/ e ios/):

  - Android: nome exibido e a permissão INTERNET (o `flutter create` só a põe
    nos manifestos de debug/profile; sem ela o release não baixa o modelo nem
    abre o catálogo).
  - iOS: nome exibido e os textos de permissão de câmera e galeria no
    Info.plist. Sem NSCameraUsageDescription / NSPhotoLibraryUsageDescription o
    iOS fecha o aplicativo ao abrir o seletor de imagem, e a App Store recusa o
    envio.

É idempotente: pode rodar quantas vezes quiser. Roda sem dependências além da
biblioteca padrão.

Uso:  python3 tools/ajustar_plataformas.py [--raiz CAMINHO_DO_PROJETO_FLUTTER]
"""

import argparse
import os
import plistlib
import re
import sys
from pathlib import Path

NOME_EXIBIDO = "Colo do Útero IA"

PERMISSOES_IOS = {
    "NSCameraUsageDescription": (
        "A câmera é usada para fotografar, pela ocular do microscópio, o campo "
        "de citologia a ser analisado. A análise roda no aparelho e a imagem "
        "não é enviada a lugar nenhum."
    ),
    "NSPhotoLibraryUsageDescription": (
        "A galeria é usada para escolher a imagem de um campo de citologia a "
        "ser analisado ou cadastrado no catálogo de lâminas."
    ),
}


def ajustar_android(raiz: Path) -> str:
    manifesto = raiz / "android" / "app" / "src" / "main" / "AndroidManifest.xml"
    if not manifesto.exists():
        return "android: AndroidManifest.xml não encontrado (pulado)"
    texto = manifesto.read_text(encoding="utf-8")
    novo, n = re.subn(r'android:label="[^"]*"', f'android:label="{NOME_EXIBIDO}"', texto, count=1)
    if n == 0:
        return "android: atributo android:label não encontrado (pulado)"
    if "android.permission.INTERNET" not in novo:
        novo = re.sub(
            r"(<manifest[^>]*>)",
            r'\1\n    <uses-permission android:name="android.permission.INTERNET"/>',
            novo,
            count=1,
        )
    if novo != texto:
        manifesto.write_text(novo, encoding="utf-8")
    return f'android: nome exibido = "{NOME_EXIBIDO}" + permissão INTERNET'


def ajustar_ios(raiz: Path) -> str:
    plist = raiz / "ios" / "Runner" / "Info.plist"
    if not plist.exists():
        return "ios: Info.plist não encontrado (pulado)"
    with plist.open("rb") as f:
        dados = plistlib.load(f)
    dados["CFBundleDisplayName"] = NOME_EXIBIDO
    dados["CFBundleName"] = NOME_EXIBIDO
    dados.update(PERMISSOES_IOS)
    # Login com Google no iOS (opcional): informe o ID do cliente iOS do
    # Google Cloud em GOOGLE_CLIENTE_IOS (formato NNN-xxx.apps.googleusercontent.com).
    cliente = os.environ.get("GOOGLE_CLIENTE_IOS", "").strip()
    if cliente.endswith(".apps.googleusercontent.com"):
        dados["GIDClientID"] = cliente
        esquema = "com.googleusercontent.apps." + cliente.split(".apps.")[0]
        tipos = dados.get("CFBundleURLTypes", [])
        if not any(esquema in t.get("CFBundleURLSchemes", []) for t in tipos):
            tipos.append({"CFBundleURLSchemes": [esquema]})
        dados["CFBundleURLTypes"] = tipos
    with plist.open("wb") as f:
        plistlib.dump(dados, f, fmt=plistlib.FMT_XML, sort_keys=False)
    return f'ios: nome exibido = "{NOME_EXIBIDO}" + textos de permissão de câmera e galeria'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raiz", default=str(Path(__file__).resolve().parent.parent))
    args = ap.parse_args()
    raiz = Path(args.raiz)
    for resultado in (ajustar_android(raiz), ajustar_ios(raiz)):
        print("  " + resultado)
    return 0


if __name__ == "__main__":
    sys.exit(main())
