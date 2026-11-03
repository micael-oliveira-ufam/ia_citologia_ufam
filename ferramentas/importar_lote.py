"""Importa uma pasta de imagens de citologia para o catálogo (Firestore) em lote.

Uso (no seu computador, com acesso à internet):
  pip3 install pillow
  python3 ferramentas/importar_lote.py "/caminho/Citologia Líquida"            # pasta extraída
  python3 ferramentas/importar_lote.py citologia.zip                          # .zip já juntado
  python3 ferramentas/importar_lote.py "/caminho/..." --pasta "03 de agosto de 2026" --simular

Regras:
  • A classe vem do nome do arquivo (HSIL, LSIL, ASCH -> ASC-H). Arquivos sem classe reconhecível são pulados.
  • Cada lâmina recebe um ID do catálogo (L-0001, L-0002…), a partir do maior já existente.
    As iniciais das pastas NÃO são gravadas: são dados de identificação de pacientes.
  • Todas ficam no campo "pasta" (padrão: "03 de agosto de 2026"), que aparece no filtro "Pasta".
  • Interrompível: as já enviadas ficam em ferramentas/importacao_feitos.jsonl e são puladas numa nova rodada.
  • Entra como sessão anônima (igual ao site). Quem enviou fica como criador; as regras só deixam o criador
    editar/excluir essas lâminas, então elas aparecem como "Importação em lote" para os outros.
"""
import argparse
import base64
import io
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_FEITOS = os.path.join(RAIZ, "ferramentas", "importacao_feitos.jsonl")
API_KEY = "AIzaSyDIye72T8UWKS04vheKeFlQhZiofFoDY14"  # mesma chave pública do site (config.js)
PROJ = "citoatlas"
BASE = f"https://firestore.googleapis.com/v1/projects/{PROJ}/databases/(default)/documents"
TENTATIVAS = [(1024, 78), (1024, 65), (800, 65), (640, 60), (480, 55)]  # igual ao site (imagem.js)
LIMITE_BASE64 = 250000
CLASSE = {"HSIL": "HSIL", "LSIL": "LSIL", "ASCH": "ASC-H", "NILM": "NILM", "SCC": "SCC"}

_tok = {"id": None, "exp": 0, "refresh": None}
_lock = threading.Lock()


def _pedir(url, corpo=None, metodo=None, cabecalhos=None):
    dados = json.dumps(corpo).encode() if corpo is not None else None
    req = urllib.request.Request(url, data=dados, method=metodo or ("POST" if dados else "GET"),
                                 headers={"Content-Type": "application/json", **(cabecalhos or {})})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:300]}")


def entrar():
    j = _pedir(f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={API_KEY}", {"returnSecureToken": True})
    _tok.update(id=j["idToken"], refresh=j["refreshToken"], exp=time.time() + int(j.get("expiresIn", 3600)) - 120)
    payload = json.loads(base64.urlsafe_b64decode(j["idToken"].split(".")[1] + "=="))
    return payload["user_id"] if "user_id" in payload else payload["sub"]


def token():
    with _lock:
        if time.time() > _tok["exp"]:
            j = _pedir(f"https://securetoken.googleapis.com/v1/token?key={API_KEY}",
                       {"grant_type": "refresh_token", "refresh_token": _tok["refresh"]})
            _tok.update(id=j["id_token"], refresh=j["refresh_token"], exp=time.time() + int(j.get("expires_in", 3600)) - 120)
        return _tok["id"]


def chamar(metodo, url, corpo=None):
    return _pedir(url, corpo, metodo, {"Authorization": f"Bearer {token()}"})


def maior_id_existente():
    maior, total, pagina = 0, 0, None
    while True:
        q = f"{BASE}/slides?pageSize=300&mask.fieldPaths=patientId" + (f"&pageToken={pagina}" if pagina else "")
        d = chamar("GET", q)
        for doc in d.get("documents", []):
            total += 1
            m = re.match(r"^L-(\d{1,6})$", doc.get("fields", {}).get("patientId", {}).get("stringValue", ""), re.I)
            if m:
                maior = max(maior, int(m.group(1)))
        pagina = d.get("nextPageToken")
        if not pagina:
            return maior, total


def comprimir(bytes_img):
    from PIL import Image
    im0 = Image.open(io.BytesIO(bytes_img)).convert("RGB")
    for lado, qualidade in TENTATIVAS:
        im = im0.copy()
        im.thumbnail((lado, lado), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=qualidade, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        if len(b64) <= LIMITE_BASE64:
            return "data:image/jpeg;base64," + b64
    raise RuntimeError("não foi possível reduzir a imagem o bastante")


def classe_do_nome(nome):
    m = re.search(r"_(HSIL|LSIL|ASCH|NILM|SCC)\b", os.path.basename(nome), re.I)
    return CLASSE[m.group(1).upper()] if m else None


def listar_imagens(caminho):
    """Devolve [(nome, leitor)]; leitor() lê os bytes só quando a imagem for enviada (poupa memória)."""
    if caminho.lower().endswith(".zip"):
        z = zipfile.ZipFile(caminho)
        return [(n, (lambda n=n: z.open(n).read())) for n in sorted(z.namelist())
                if n.lower().endswith((".jpg", ".jpeg", ".png", ".webp")) and not n.endswith("/")]
    saida = []
    for dp, _, fs in os.walk(caminho):
        for f in sorted(fs):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                p = os.path.join(dp, f)
                saida.append((p, (lambda p=p: open(p, "rb").read())))
    return sorted(saida, key=lambda t: t[0])


def campos(imagem, classificacao, pid, pasta, data, uid, t_ms):
    s = lambda v: {"stringValue": v}
    return {
        "patientId": s(pid), "imageUrl": s(imagem), "classification": s(classificacao),
        "date": s(data), "morphologyDescription": s(""),
        "notes": s(f"Importada em lote: {pasta}."),
        "creatorId": s(uid), "creatorName": s("Importação em lote"),
        "createdAt": {"integerValue": str(t_ms)}, "updatedAt": {"integerValue": str(t_ms)},
        "pasta": s(pasta),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("origem", help="pasta com as imagens ou arquivo .zip")
    ap.add_argument("--pasta", default="03 de agosto de 2026", help="nome da pasta no catálogo")
    ap.add_argument("--data", default="2026-08-03", help="data da coleta (AAAA-MM-DD)")
    ap.add_argument("--simular", action="store_true", help="não envia nada; só conta e confere")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    imagens = listar_imagens(a.origem)
    pares = [(n, ler, classe_do_nome(n)) for n, ler in imagens]
    sem_classe = [n for n, _, c in pares if c is None]
    validas = [(n, ler, c) for n, ler, c in pares if c]
    feitos = {}
    if os.path.exists(ARQ_FEITOS):
        for l in open(ARQ_FEITOS, encoding="utf-8"):
            r = json.loads(l)
            feitos[r["arquivo"]] = r
    pendentes = [t for t in validas if t[0] not in feitos]
    print(f"Imagens: {len(imagens)} | com classe: {len(validas)} | sem classe (puladas): {len(sem_classe)} | "
          f"já enviadas: {len(feitos)} | a enviar: {len(pendentes)}")
    if sem_classe[:5]:
        print("Exemplos sem classe:", *sem_classe[:5], sep="\n  ")
    if a.simular:
        print("Simulação: nada foi enviado.")
        return

    uid = entrar()
    maior, total = maior_id_existente()
    print(f"Banco atual: {total} lâminas (maior L-{maior:04d}). Enviando…", flush=True)
    t0 = int(time.time() * 1000)
    arq = open(ARQ_FEITOS, "a", encoding="utf-8")
    erros = []
    contador = {"ok": 0}

    def preparar(i_t):
        i, (nome, ler, classe) = i_t
        pid = f"L-{maior + 1 + i:04d}"
        imagem = comprimir(ler())
        corpo = {"fields": campos(imagem, classe, pid, a.pasta, a.data, uid, t0 + i)}
        for tentativa in range(3):
            try:
                d = chamar("POST", f"{BASE}/slides", corpo)
                return {"arquivo": nome, "id": d["name"].split("/")[-1], "patientId": pid}, None
            except Exception as e:  # tenta de novo com pausa
                erro = str(e)
                time.sleep(2 + tentativa * 3)
        return None, f"{nome}: {erro}"

    tarefas = list(enumerate(pendentes))
    with ThreadPoolExecutor(max_workers=a.threads) as ex:
        for res, erro in ex.map(preparar, tarefas):
            if res:
                arq.write(json.dumps(res, ensure_ascii=False) + "\n")
                arq.flush()
                contador["ok"] += 1
                if contador["ok"] % 25 == 0:
                    print(f"  {contador['ok']}/{len(pendentes)}", flush=True)
            else:
                erros.append(erro)
    arq.close()
    print(f"Concluído: {contador['ok']} criadas, {len(erros)} falhas.")
    for e in erros[:10]:
        print("  FALHA", e)
    if erros:
        print("Rode de novo o mesmo comando para tentar só as que falharam.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrompido. Rode de novo para continuar de onde parou.")
