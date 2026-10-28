// Navegação por abas (equivalente aos seis menus do Streamlit). Cada seção é um <section
// class="secao" id="...">; sem JavaScript, todas aparecem empilhadas.

const SECOES = [...document.querySelectorAll('.secao')];
const IDS = SECOES.map((s) => s.id);
const ABAS = [...document.querySelectorAll('.abas a[data-secao]')];
const PADRAO = 'analise';

function mostrar(id, { foco = false } = {}) {
  if (!IDS.includes(id)) id = PADRAO;
  for (const s of SECOES) s.classList.toggle('ativa', s.id === id);
  for (const a of ABAS) {
    if (a.dataset.secao === id) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  }
  const abas = document.querySelector('.abas');
  if (abas && window.scrollY > abas.offsetTop) window.scrollTo({ top: Math.max(0, abas.offsetTop - 12) });
  if (foco) document.getElementById('conteudo')?.focus({ preventScroll: true });
}

export function irPara(id) {
  if (location.hash.slice(1) === id) mostrar(id, { foco: true });
  else location.hash = `#${id}`;
}

export function iniciarNavegacao() {
  document.documentElement.classList.add('js');
  const aoMudar = (foco) => mostrar(decodeURIComponent(location.hash.slice(1)) || PADRAO, { foco });
  window.addEventListener('hashchange', () => aoMudar(true));
  aoMudar(false);
}
