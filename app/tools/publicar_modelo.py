#!/usr/bin/env python3
"""
Publica o modelo de citologia (.onnx) para o site e para o aplicativo baixarem.

O aplicativo e o site NÃO embarcam o modelo: eles baixam, do GitHub Pages, o
arquivo e o manifesto que este script grava em docs/model/. O manifesto traz
versão, tamanho, SHA-256 e a ordem das classes; quem baixa confere tudo antes de
usar e recusa o modelo se algo não bater.

Fluxo:
    1. python3 tools/export_onnx.py --checkpoint ARQUIVO.pt --saida /tmp/citologia_convnext.onnx
    2. python3 tools/publicar_modelo.py --onnx /tmp/citologia_convnext.onnx
    3. git add docs/model && git commit && git push   (o Pages republica sozinho)

Limite do GitHub: arquivos de 100 MB ou mais são recusados num push comum. O
modelo int8 do ConvNeXt-Tiny/Small/Base fica abaixo disso; o float32 do Base não.
"""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

CLASSES = ["SCC", "HSIL", "LSIL", "NILM"]  # mesma ordem de lib/data/bethesda.dart
NOME_ARQUIVO = "citologia_convnext.onnx"
LIMITE_GITHUB = 100 * 1024 * 1024
TAMANHO_PARTE = 64 * 1024 * 1024


def sha256_de(caminho: Path) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def ler_metadados(caminho: Path) -> dict:
    """Metadados gravados pelo export_onnx.py; vazio se o pacote onnx não existir."""
    try:
        import onnx
    except ImportError:
        print("  aviso: pacote 'onnx' não instalado — não consigo conferir classes/entradas no arquivo")
        return {}
    modelo = onnx.load(str(caminho), load_external_data=False)
    meta = {p.key: p.value for p in modelo.metadata_props}
    entradas = [i.name for i in modelo.graph.input]
    saidas = [o.name for o in modelo.graph.output]
    meta["_entradas"] = entradas
    meta["_saidas"] = saidas
    return meta


def main() -> int:
    raiz_app = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, type=Path, help="modelo gerado por tools/export_onnx.py")
    ap.add_argument("--saida", type=Path, default=raiz_app.parent / "docs" / "model",
                    help="pasta publicada pelo GitHub Pages (padrão: docs/model)")
    ap.add_argument("--versao", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    help="rótulo da versão (padrão: data de hoje, UTC). Mude a cada novo treinamento.")
    ap.add_argument("--arquitetura", default=None, help="ex.: convnext_tiny (padrão: lida do .onnx)")
    ap.add_argument("--quantizacao", default=None, choices=["int8", "float32"],
                    help="como o arquivo foi gerado (informativo)")
    ap.add_argument("--parte-mib", type=int, default=64,
                    help="tamanho das partes, em MiB, se o modelo passar de 100 MB (padrão: 64)")
    args = ap.parse_args()
    tamanho_parte = args.parte_mib * 1024 * 1024

    if not args.onnx.is_file():
        sys.exit(f"arquivo não encontrado: {args.onnx}")

    tamanho = args.onnx.stat().st_size
    print(f"[1/3] Conferindo {args.onnx} ({tamanho / 1048576:.1f} MB)")
    dividir = tamanho >= LIMITE_GITHUB
    if dividir:
        print("  ≥ 100 MB: o GitHub recusaria o arquivo inteiro; ele será publicado em partes "
              "(o site e o app as juntam e conferem o SHA-256 do todo).")

    meta = ler_metadados(args.onnx)
    if meta:
        if meta["_entradas"][:1] != ["imagem"] or not {"logits", "cam"} <= set(meta["_saidas"]):
            sys.exit(f"entradas/saídas inesperadas: {meta['_entradas']} -> {meta['_saidas']} "
                     "(esperado: imagem -> logits, cam). Reexporte com tools/export_onnx.py.")
        if "classes" in meta and json.loads(meta["classes"]) != CLASSES:
            sys.exit(f"A ordem das classes no .onnx ({meta['classes']}) difere de {CLASSES}. "
                     "Publicar assim inverteria diagnósticos.")
        print("  entradas, saídas e ordem das classes conferem")

    print("[2/3] Calculando o SHA-256")
    destino = args.saida
    destino.mkdir(parents=True, exist_ok=True)
    sha = sha256_de(args.onnx)
    for velho in destino.glob(NOME_ARQUIVO + "*"):
        velho.unlink()  # versões antigas (inteiro ou partes)
    partes = []
    if dividir:
        with args.onnx.open("rb") as f:
            n = 0
            while True:
                bloco = f.read(tamanho_parte)
                if not bloco:
                    break
                n += 1
                nome = f"{NOME_ARQUIVO}.parte{n:02d}"
                (destino / nome).write_bytes(bloco)
                partes.append({"arquivo": nome, "bytes": len(bloco),
                               "sha256": hashlib.sha256(bloco).hexdigest()})
        print(f"  {len(partes)} partes de até {args.parte_mib} MiB")
    else:
        shutil.copyfile(args.onnx, destino / NOME_ARQUIVO)

    manifesto = {
        "esquema": 1,
        "versao": args.versao,
        "arquivo": NOME_ARQUIVO,
        "bytes": tamanho,
        "sha256": sha,
        "partes": partes or None,
        "classes": CLASSES,
        "entrada": {"nome": "imagem", "forma": [1, 3, 224, 224], "formato": "RGB [0,1], NCHW"},
        "saidas": ["logits", "cam"],
        "arquitetura": args.arquitetura or meta.get("arquitetura"),
        "quantizacao": args.quantizacao,
        "criado_em": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifesto = {k: v for k, v in manifesto.items() if v is not None}
    (destino / "manifesto.json").write_text(json.dumps(manifesto, indent=2, ensure_ascii=False) + "\n",
                                            encoding="utf-8")

    print(f"[3/3] Publicado em {destino}")
    print(f"  versão {manifesto['versao']}  ·  sha256 {sha[:16]}…  ·  {tamanho / 1048576:.1f} MB")
    print("\nPróximo passo: git add docs/model && git commit && git push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
