// Compressão da foto no navegador para caber num documento do Firestore (< 1 MB).
// Gera um data URL JPEG com até ~250.000 caracteres base64.

export const LIMITE_BASE64 = 250000;
const TENTATIVAS = [[1024, 0.78], [1024, 0.65], [800, 0.65], [640, 0.6], [480, 0.55]];

function carregar(arquivo) {
  return new Promise((ok, falha) => {
    const url = URL.createObjectURL(arquivo);
    const img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); ok(img); };
    img.onerror = () => { URL.revokeObjectURL(url); falha(new Error('Não foi possível ler esta imagem. Use JPG, PNG ou WebP.')); };
    img.src = url;
  });
}

export async function comprimirImagem(arquivo) {
  if (!arquivo || !/^image\/(jpeg|png|webp)$/.test(arquivo.type)) {
    throw new Error('Escolha uma imagem JPG, PNG ou WebP.');
  }
  const img = await carregar(arquivo);
  const w0 = img.naturalWidth;
  const h0 = img.naturalHeight;
  if (!w0 || !h0) throw new Error('Imagem inválida.');
  for (const [lado, qualidade] of TENTATIVAS) {
    const escala = Math.min(1, lado / Math.max(w0, h0));
    const c = document.createElement('canvas');
    c.width = Math.max(1, Math.round(w0 * escala));
    c.height = Math.max(1, Math.round(h0 * escala));
    const g = c.getContext('2d');
    g.fillStyle = '#fff'; // PNG/WebP transparentes
    g.fillRect(0, 0, c.width, c.height);
    g.drawImage(img, 0, 0, c.width, c.height);
    const dados = c.toDataURL('image/jpeg', qualidade);
    if (dados.length - dados.indexOf(',') - 1 <= LIMITE_BASE64) return dados;
  }
  throw new Error('Não foi possível reduzir esta imagem o bastante. Tente uma foto menor.');
}
