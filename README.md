# Sistema de Apoio ao Diagnóstico Citológico: UFAM & SEMSA Manaus

Classificação de citologia em meio líquido pelo Sistema Bethesda com ConvNeXt
Large e Grad-CAM.

> Uso acadêmico e demonstrativo. Modelo treinado apenas com o acervo público
> IARC Digital Atlas, sem aprovação de Comitê de Ética. Não emite laudo, não
> substitui o citopatologista e não deve receber imagens de pacientes.

## Estrutura

```
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
```

O checkpoint `convnext_liquid_based_citology_IARC_digital_atlas_01_08_26.pt` é
baixado do GitHub Releases na primeira execução e fica em cache na raiz.

## Executar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Onde editar

| O quê | Onde |
| --- | --- |
| Equipe e vínculos | lista `EQUIPE` |
| Instituições e o papel de cada uma | lista `INSTITUICOES` |
| Métricas exibidas | dicionário `METRICAS` |
| Matriz de confusão | `MATRIZ_CONFUSAO` (ordem NILM, LSIL, HSIL, SCC) |
| Imagens de teste e descrições | lista `EXEMPLOS` |
| Cores e paleta | bloco CSS em `injetar_css()` |
| Textos didáticos do Bethesda | `BETHESDA_DETALHE` e `TEXTO_BETHESDA_*` |
| Itens do menu | lista `MENUS` |
| Ano no rodapé | `ANO_DESENVOLVIMENTO` |

## Ordem das classes

`CLASSES = ["SCC", "HSIL", "LSIL", "NILM"]` corresponde às pastas
`Carcinoma / HSIL / LSIL / Normal` ordenadas alfabeticamente pelo `ImageFolder`: a mesma ordem dos eixos da matriz de confusão do experimento. Se o dataset
for reorganizado, esta lista precisa ser revista: uma troca aqui inverte
diagnósticos sem gerar erro visível.

## Detecção automática da arquitetura

O carregador inspeciona o `state_dict` e monta sozinho a variante do ConvNeXt
(`tiny`/`small`/`base`/`large`) e a cabeça de classificação, com ou sem o
`Dropout` antes da camada linear. Isso evita erro silencioso ao trocar de
checkpoint.


## Navegação

O menu principal é um `st.radio` estilizado como abas, dentro de um
`st.container(key="navegacao")`, a classe `st-key-navegacao` é o gancho do CSS.
Foi preciso abandonar `st.tabs` porque não há como trocar a aba ativa por
código: o botão *Analisar* das imagens de teste precisa levar o usuário até a
seção Análise, e isso é feito no `on_click`, que roda antes da renderização do
próximo ciclo.

Não há barra lateral: o envio de imagem fica na própria seção Análise e os
créditos, no cabeçalho.


## Publicação na internet

`publicar.sh` sobe o Streamlit localmente, abre um túnel da Cloudflare e mostra
o QR code de acesso.

```bash
./publicar.sh                      # endereço trycloudflare.com, sem conta
./publicar.sh --porta 8502         # outra porta local
./publicar.sh --nomeado meu-tunel  # túnel nomeado, exige conta Cloudflare
./publicar.sh --so-qrcode https://... # QR code de um endereço já existente
```

O script instala o `cloudflared` na pasta do projeto se não o encontrar, espera
o servidor responder ao teste de saúde, lê o endereço público do log do túnel e
grava `dist/qrcode_plataforma.png` e `dist/endereco_publico.txt`. O QR code
também é impresso em ASCII, o que ajuda quando o script roda por SSH.

O túnel rápido não exige conta nem domínio próprio, mas o endereço é sorteado a
cada execução e o túnel cai quando o script encerra. Para um endereço fixo,
crie um túnel nomeado no painel da Cloudflare e use `--nomeado`.

Qualquer pessoa com o link acessa a plataforma. Como o projeto não tem aprovação
de CEP, ela não deve ser usada para processar imagens de pacientes.

## Descoberta em buscadores

`injetar_metadados()` grava título, descrição, palavras-chave, Open Graph e
dados estruturados JSON-LD no documento. As palavras-chave estão em
`SEO_PALAVRAS_CHAVE`.

O Streamlit monta o `<head>` por conta própria e não expõe meta tags, então elas
são inseridas por script a partir do corpo da página. Buscadores que executam
JavaScript indexam normalmente; para indexação garantida, o caminho é servir a
plataforma atrás de um proxy que devolva as tags no HTML inicial.

## Ferramentas de apoio

O código foi desenvolvido com apoio do modelo de IA generativa Claude Opus 5
(Anthropic). A concepção científica, o treinamento, a validação e a revisão do
código são de responsabilidade da equipe do projeto. Essa informação aparece na
seção Dados e créditos e no rodapé da plataforma.
