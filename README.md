# Sistema de Apoio ao Diagnóstico Citológico: UFAM & SEMSA Manaus

Classificação de citologia em meio líquido pelo Sistema Bethesda com ConvNeXt Large e Grad-CAM.

> **Aviso:** Uso acadêmico e demonstrativo. Modelo treinado apenas com o acervo público IARC Digital Atlas, sem aprovação de Comitê de Ética. Não emite laudo, não substitui o citopatologista e não deve receber imagens de pacientes.

---

## 🔗 Acesso às Plataformas

- **Plataforma de Diagnóstico Citológico:** [ia-citologia-ufam.streamlit.app](https://ia-citologia-ufam.streamlit.app/)
- **Site (GitHub Pages): catálogo de lâminas e predição de imagens, em módulos separados:** [micael-oliveira-ufam.github.io/ia_citologia_ufam](https://micael-oliveira-ufam.github.io/ia_citologia_ufam/) *(ativo após publicar o modelo e habilitar o Pages em `main` / `/docs`)*
- **Sistema Automatizado para Catalogação de Lâminas (Citoatlas Sebastião Marinho):** [citoatlas-semsa-sebastiao-marinho.ai.studio](https://citoatlas-semsa-sebastiao-marinho.ai.studio/)

---

## 👥 Autoria e Instituições

**Equipe de Pesquisa e Desenvolvimento:**
* **Micael Davi Lima de Oliveira** ¹
* **Fabíola Guerra Nakamura** ²
* **Felipe Gomes de Oliveira** ³
* **Carolina Marinho da Costa** ⁴
* **Ivanete de Lima Sampaio** ⁴
* **Toni Ricardo Martins** ⁵,⁶,⁷

**Vínculos Institucionais:**
1. **FCF / UFAM** – Faculdade de Ciências Farmacêuticas, Universidade Federal do Amazonas.
2. **ICOMP / UFAM** – Instituto de Computação, Universidade Federal do Amazonas.
3. **ICET / UFAM** – Instituto de Ciências Exatas e Tecnologia, Universidade Federal do Amazonas.
4. **SEMSA** – Laboratório de Citopatologia Professor Sebastião Ferreira Marinho, Secretaria Municipal de Saúde de Manaus.
5. **FCF / UFAM** – Docente da Faculdade de Ciências Farmacêuticas, Universidade Federal do Amazonas.
6. **PPGIBA / UFAM** – Programa de Pós-graduação em Imunologia Básica e Aplicada, Universidade Federal do Amazonas.
7. **LIM 52 / USP** – Instituto de Medicina Tropical, Laboratório de Virologia, Universidade de São Paulo, São Paulo - SP, Brasil.

---

## 📂 Estrutura do Repositório

```text
app.py
requirements.txt
.streamlit/config.toml            tema claro
logo_ufam.png  logo-icomp.png  semsa.png
assets/
  matriz_confusao.png             figura original do experimento
  historico_treinamento.png       (opcional) curvas de acurácia e perda
exemplos/
  cyto5940.jpg  cyt14686a.jpg     campos com gabarito do atlas
  cyto5950.jpg  cyto2870.jpg  cyt10131a.jpg  cyt16243.jpg
app/                              aplicativo Flutter (Android/iOS): paciente ou profissional,
                                  análise de citologia (modelo baixado do GitHub Pages) e catálogo de lâminas
docs/                             site no GitHub Pages (index, catalogo.html, predicao.html, model/)
ml/                               treino do modelo de risco (Python)
firebase/firestore.rules.sugerido regras sugeridas do Firestore (revisar antes de publicar)
```

> O aplicativo, o site e o passo a passo de compilação, publicação do modelo e login Google estão em
> [`app/README.md`](app/README.md), incluindo o que foi e o que **não** foi verificado.
> A plataforma Streamlit (`app.py`) segue como está.
