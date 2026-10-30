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


MARCA = "colo_ia: assinatura"


def _gradle_app(raiz: Path):
    for nome in ("build.gradle.kts", "build.gradle"):
        c = raiz / "android" / "app" / nome
        if c.exists():
            return c
    return None


def _inserir_apos_plugins(texto: str, carga: str) -> str:
    """O Gradle exige o bloco plugins {} antes de qualquer outra instrução (só
    import/comentários podem vir antes): o import vai ao topo e o resto, logo
    depois do bloco plugins."""
    linhas = carga.splitlines(keepends=True)
    imports = [l for l in linhas if l.startswith("import ")]
    resto = "".join(l for l in linhas if not l.startswith("import "))
    topo = "".join(imports)
    m = re.search(r"^plugins\s*\{", texto, re.M)
    if not m:
        return topo + resto + texto
    i, prof = m.end(), 1
    while i < len(texto) and prof:
        prof += {"{": 1, "}": -1}.get(texto[i], 0)
        i += 1
    return topo + texto[:i] + "\n\n" + resto.rstrip("\n") + "\n" + texto[i:]


def ajustar_gradle(raiz: Path) -> list:
    """applicationId opcional (APPLICATION_ID) e assinatura de release por key.properties."""
    arq = _gradle_app(raiz)
    if arq is None:
        return ["gradle: android/app/build.gradle(.kts) não encontrado (pulado)"]
    kts = arq.suffix == ".kts"
    t = arq.read_text(encoding="utf-8")
    msgs = []

    app_id = os.environ.get("APPLICATION_ID", "").strip()
    if app_id:
        if not re.fullmatch(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+", app_id):
            msgs.append(f"gradle: APPLICATION_ID inválido ({app_id!r}); ignorado")
        else:
            t2, n = re.subn(r'applicationId(\s*=\s*|\s+)"[^"]*"', lambda m: f'applicationId{m.group(1)}"{app_id}"', t, count=1)
            t = t2
            msgs.append(f"gradle: applicationId = {app_id}" if n else "gradle: applicationId não encontrado (pulado)")

    if MARCA in t:
        msgs.append("gradle: assinatura de release já configurada")
    else:
        if kts:
            carga = (
                "// colo_ia: assinatura (lê android/key.properties, se existir)\n"
                "import java.util.Properties\n"
                "val keystoreProperties = Properties()\n"
                'val keystorePropertiesFile = rootProject.file("key.properties")\n'
                "if (keystorePropertiesFile.exists()) {\n"
                "    keystoreProperties.load(keystorePropertiesFile.inputStream())\n"
                "}\n\n"
            )
            bloco = (
                "    signingConfigs {\n"
                '        create("release") {\n'
                "            if (keystorePropertiesFile.exists()) {\n"
                '                storeFile = file(keystoreProperties["storeFile"] as String)\n'
                '                storePassword = keystoreProperties["storePassword"] as String\n'
                '                keyAlias = keystoreProperties["keyAlias"] as String\n'
                '                keyPassword = keystoreProperties["keyPassword"] as String\n'
                "            }\n"
                "        }\n"
                "    }\n\n"
            )
            troca = ('signingConfig = if (keystorePropertiesFile.exists()) signingConfigs.getByName("release") '
                     'else signingConfigs.getByName("debug")')
            padrao = r'signingConfig\s*=\s*signingConfigs\.getByName\("debug"\)'
        else:
            carga = (
                "// colo_ia: assinatura (lê android/key.properties, se existir)\n"
                "def keystoreProperties = new Properties()\n"
                'def keystorePropertiesFile = rootProject.file("key.properties")\n'
                "if (keystorePropertiesFile.exists()) {\n"
                "    keystoreProperties.load(new FileInputStream(keystorePropertiesFile))\n"
                "}\n\n"
            )
            bloco = (
                "    signingConfigs {\n"
                "        release {\n"
                "            if (keystorePropertiesFile.exists()) {\n"
                "                storeFile file(keystoreProperties['storeFile'])\n"
                "                storePassword keystoreProperties['storePassword']\n"
                "                keyAlias keystoreProperties['keyAlias']\n"
                "                keyPassword keystoreProperties['keyPassword']\n"
                "            }\n"
                "        }\n"
                "    }\n\n"
            )
            troca = "signingConfig keystorePropertiesFile.exists() ? signingConfigs.release : signingConfigs.debug"
            padrao = r"signingConfig\s+signingConfigs\.debug"
        t3, n1 = re.subn(r"(\n\s*)buildTypes\s*\{", lambda m: "\n" + bloco.rstrip("\n") + "\n" + m.group(0), t, count=1)
        t3, n2 = re.subn(padrao, troca, t3, count=1)
        if n1 and n2:
            t = _inserir_apos_plugins(t3, carga)
            msgs.append("gradle: assinatura de release por android/key.properties configurada")
        else:
            msgs.append("gradle: modelo do build.gradle diferente do esperado; assinatura NÃO configurada (o APK sairá com a chave de depuração)")
    if t != arq.read_text(encoding="utf-8"):
        arq.write_text(t, encoding="utf-8")
    return msgs


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
    resultados = [ajustar_android(raiz), *ajustar_gradle(raiz), ajustar_ios(raiz)]
    for resultado in resultados:
        print("  " + resultado)
    return 0


if __name__ == "__main__":
    sys.exit(main())
