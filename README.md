# Sistema de Apoio ao Diagnóstico Citológico: UFAM & SEMSA Manaus

Classificação de citologia em meio líquido pelo Sistema Bethesda com ConvNeXt Large e Grad-CAM.

> **Aviso:** Uso acadêmico e demonstrativo. Modelo treinado apenas com o acervo público IARC Digital Atlas, sem aprovação de Comitê de Ética. Não emite laudo, não substitui o citopatologista e não deve receber imagens de pacientes.

---

## 🔗 Acesso às Plataformas

- **Plataforma de Diagnóstico Citológico:** [ia-citologia-ufam.streamlit.app](https://ia-citologia-ufam.streamlit.app/)
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
