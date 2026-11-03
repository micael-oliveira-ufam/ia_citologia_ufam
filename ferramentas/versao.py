"""Versão única do CitoIlumina.

Fonte da verdade: VERSAO (na raiz do repositório), no formato NOME+BUILD, ex.: 1.1.0+2.

  python3 ferramentas/versao.py              sincroniza app/pubspec.yaml, app/lib/versao.dart e docs/js/versao.js
  python3 ferramentas/versao.py --verificar  falha se algum arquivo estiver fora de sincronia
  python3 ferramentas/versao.py --slug       imprime o nome de pasta/arquivo: citoilumina-v1.1.0-b2
  python3 ferramentas/versao.py --nome       imprime 1.1.0
  python3 ferramentas/versao.py --build      imprime 2
"""
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ARQ_VERSAO = RAIZ / "VERSAO"
PUBSPEC = RAIZ / "app" / "pubspec.yaml"
VERSAO_DART = RAIZ / "app" / "lib" / "versao.dart"
VERSAO_JS = RAIZ / "docs" / "js" / "versao.js"

PADRAO = re.compile(r"^(\d+\.\d+\.\d+)\+(\d+)$")


def ler():
    texto = ARQ_VERSAO.read_text(encoding="utf-8").strip()
    m = PADRAO.match(texto)
    if not m:
        sys.exit(f"VERSAO inválida: {texto!r} (use o formato 1.2.3+4)")
    return m.group(1), int(m.group(2))


def esperado(nome, build):
    completa = f"{nome}+{build}"
    dart = (
        "// Gerado por ferramentas/versao.py a partir de VERSAO. Não edite à mão.\n"
        f"const String versaoNome = '{nome}';\n"
        f"const int versaoBuild = {build};\n"
        f"const String versaoCompleta = '{completa}';\n"
    )
    js = (
        "// Gerado por ferramentas/versao.py a partir de VERSAO. Não edite à mão.\n"
        f"export const VERSAO_NOME = '{nome}';\n"
        f"export const VERSAO_BUILD = {build};\n"
        f"export const VERSAO_COMPLETA = '{completa}';\n"
    )
    return dart, js


def main(args):
    nome, build = ler()
    completa = f"{nome}+{build}"
    slug = f"citoilumina-v{nome}-b{build}"
    if "--slug" in args:
        return print(slug)
    if "--nome" in args:
        return print(nome)
    if "--build" in args:
        return print(build)

    dart, js = esperado(nome, build)
    pubspec_atual = PUBSPEC.read_text(encoding="utf-8")
    novo_pubspec = re.sub(r"^version:\s*.*$", f"version: {completa}", pubspec_atual, count=1, flags=re.M)

    if "--verificar" in args:
        problemas = []
        if f"version: {completa}" not in pubspec_atual:
            problemas.append("app/pubspec.yaml")
        if not VERSAO_DART.exists() or VERSAO_DART.read_text(encoding="utf-8") != dart:
            problemas.append("app/lib/versao.dart")
        if not VERSAO_JS.exists() or VERSAO_JS.read_text(encoding="utf-8") != js:
            problemas.append("docs/js/versao.js")
        if problemas:
            sys.exit("Fora de sincronia com VERSAO (rode python3 ferramentas/versao.py): " + ", ".join(problemas))
        return print(f"Versão {completa} sincronizada em todos os arquivos.")

    PUBSPEC.write_text(novo_pubspec, encoding="utf-8")
    VERSAO_DART.write_text(dart, encoding="utf-8")
    VERSAO_JS.write_text(js, encoding="utf-8")
    print(f"Versão {completa} aplicada (pasta/arquivo: {slug}).")


if __name__ == "__main__":
    main(sys.argv[1:])
