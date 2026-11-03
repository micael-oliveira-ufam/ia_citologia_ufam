"""Gera o pacote de entrega com versão no nome.

  python3 ferramentas/empacotar.py                 -> dist/citoilumina-v1.1.0-b2.tgz (sem o modelo de IA)
  python3 ferramentas/empacotar.py --com-modelo    -> inclui docs/model (arquivo grande)

Dentro do .tgz existe UMA pasta de topo com o mesmo nome (citoilumina-v1.1.0-b2/),
então ao descompactar a pasta já sai com a versão no nome.
"""
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ITENS = [
    "README.md", "VERSAO", "docs", "ferramentas", "firebase", "ml", ".github",
    "app/README.md", "app/assets", "app/pubspec.yaml", "app/lib", "app/tools",
    "app/build_apk.sh", "app/test", "app/android", "app/ios", "app/web",
]
EXCLUIR_PADRAO = ["docs/model", "docs/vendor", "__pycache__"]


def slug():
    return subprocess.run(
        [sys.executable, str(RAIZ / "ferramentas" / "versao.py"), "--slug"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def main(args):
    subprocess.run([sys.executable, str(RAIZ / "ferramentas" / "versao.py"), "--verificar"], check=True)
    nome = slug()
    com_modelo = "--com-modelo" in args
    excluir = [] if com_modelo else EXCLUIR_PADRAO
    destino = RAIZ / "dist"
    destino.mkdir(exist_ok=True)
    pasta = destino / nome
    if pasta.exists():
        shutil.rmtree(pasta)
    pasta.mkdir()

    def ignorar(_dir, nomes):
        return {n for n in nomes if n == "__pycache__" or n.endswith(".pyc")}

    for item in ITENS:
        origem = RAIZ / item
        if not origem.exists():
            continue
        if any(item == e or item.startswith(e + "/") for e in excluir):
            continue
        alvo = pasta / item
        alvo.parent.mkdir(parents=True, exist_ok=True)
        if origem.is_dir():
            shutil.copytree(origem, alvo, ignore=ignorar, dirs_exist_ok=True)
            for e in excluir:
                caminho = pasta / e
                if caminho.exists() and caminho.is_dir() and e.startswith(item + "/"):
                    shutil.rmtree(caminho)
        else:
            shutil.copy2(origem, alvo)

    arquivo = destino / f"{nome}.tgz"
    with tarfile.open(arquivo, "w:gz") as tar:
        tar.add(pasta, arcname=nome)
    shutil.rmtree(pasta)
    print(f"Pacote: {arquivo}")
    print(f"Ao descompactar: {nome}/")


if __name__ == "__main__":
    main(sys.argv[1:])
