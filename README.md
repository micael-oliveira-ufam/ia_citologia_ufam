# Colo do Útero com IA

Aplicativo Flutter (Android e iOS) que reúne, em um só lugar:

- **conscientização e prevenção** do câncer do colo do útero, com estatísticas e fontes;
- **estimativa educativa de perfil de risco** por um modelo de IA treinado em dados públicos;
- **análise de campos de citologia em meio líquido** por um ConvNeXt embarcado (Sistema Bethesda,
  com mapa de calor e verificação de robustez).

- **catálogo de lâminas** (CitoAtlas, LabSeba): cadastro e **avaliação cega** por citopatologistas;
- um **site no GitHub Pages** (`docs/`) com o catálogo (somente leitura) e a predição de imagens,
  em módulos separados.

A análise de citologia roda **dentro do aparelho/navegador**: a imagem não é enviada a servidor. A
internet é usada para (1) baixar **uma vez** o arquivo do modelo, publicado no GitHub Pages, e
(2) acessar o catálogo (Firestore) e o login Google.

> Uso acadêmico e demonstrativo. O modelo de citologia foi treinado só com o IARC Digital Atlas,
> sem aprovação de Comitê de Ética, e o de risco tem poder discriminativo fraco (veja
> *Limitações*). Nenhum dos dois emite laudo nem substitui exame ou profissional.

## Como o perfil muda o aplicativo

Ao abrir, o usuário escolhe **paciente** ou **profissional de saúde** (a tela aparece a cada
abertura e há um botão "Trocar perfil" no topo). A escolha define as abas e o nível de detalhe:

| | Paciente | Profissional de saúde |
|---|---|---|
| Abas | Início · Prevenção · Meu risco · Seu exame · Números | **Analisar** · Bethesda · **Catálogo** · Orientar · Risco |
| Citologia com IA | Não tem. "Seu exame" explica NILM/LSIL/HSIL/SCC em linguagem simples, com "o que significa" e "o que fazer" | Análise de campo (galeria, câmera na ocular ou exemplos), mapa de calor, robustez em 8 orientações |
| Prevenção | Texto simples | Mesmo conteúdo com detalhe técnico, pontos para a conversa com a paciente e números com fonte |
| Risco | Faixa (menor/intermediária/maior) e orientação | Além da faixa: escore, cortes, taxa observada por faixa, AUC e teste de permutação |
| Sobre o projeto | O aplicativo · Equipe · Limites | O projeto · Equipe · Desempenho · Modelo de risco (gráficos) · Limitações |

A escolha é **de conteúdo, não de acesso**: qualquer pessoa pode escolher qualquer perfil (a
aba Catálogo, porém, exige login Google para avaliar ou cadastrar). A ferramenta de citologia traz o aviso ético em todas as telas.

## Estrutura

```
app/                      projeto Flutter
  lib/
    main.dart, casca.dart, perfil.dart      raiz, navegação por perfil, enum Perfil
    telas/                                   abas e telas (escolha_perfil, aba_*, tela_resultado, tela_sobre)
    servicos/                                classificador.dart (ONNX), gerenciador_modelo.dart (download), imagem.dart, modelo_risco.dart
    catalogo/                                modelos, firestore_rest, sessao_google, controle_catalogo
    data/                                    bethesda, projeto, estatisticas, conteudo_prevencao, bethesda_paciente
    widgets/                                 comuns.dart (citologia), conteudo.dart (conscientização)
  assets/
    exemplos/                                4 campos do IARC Digital Atlas
    risco/modelo_risco.json                  modelo de risco, gerado por ml/train.py
    dados/estatisticas.json                  números e fontes (editável, com selo "a confirmar")
    graficos/                                gráficos de desempenho, gerados por ml/train.py
  tools/export_onnx.py                       checkpoint .pt -> ONNX (com CAM e normalização embutidos)
  tools/ajustar_plataformas.py               nome do app e permissões do iOS
  build_apk.sh                               gera os APKs
  test/                                      testes (modelo de risco contra scikit-learn; abertura do app)
docs/                     SITE (GitHub Pages), no estilo da versão Streamlit: index.html (6 abas), catalogo.html, js/, css/, fontes/, img/, vendor/onnxruntime-web
ferramentas/              gerar_site.py + conteudo_site.py: geram docs/index.html, catalogo.html e js/dados.js (fonte única dos textos e números)
  model/                                     citologia_convnext.onnx + manifesto.json  <- GERADO por tools/publicar_modelo.py
firebase/firestore.rules.sugerido            regras sugeridas (revisar!)
ml/                       treino do modelo de risco (Python) e dados
  train.py, requirements.txt, data_raw/risk_factors_cervical_cancer.csv (UCI), outputs/
```

## Compilar

Pré-requisitos: Flutter estável (Dart ≥ 3.6), JDK 17 e Android SDK para Android; macOS + Xcode para iOS.

### 1. Publicar o modelo de citologia (uma vez, e a cada novo treino)

O `.onnx` não vai no APK nem no código: fica em `docs/model/` e é servido pelo GitHub Pages. O
app e o site baixam de lá, conferem tamanho e **SHA-256** do `manifesto.json` e recusam uma ordem
de classes diferente de `SCC, HSIL, LSIL, NILM`.

```bash
cd app
pip install torch torchvision onnx onnxruntime pillow numpy
python3 tools/export_onnx.py \
  --checkpoint convnext_liquid_based_citology_IARC_digital_atlas_06_05_26.pt \
  --saida /tmp/citologia_convnext.onnx
python3 tools/publicar_modelo.py --onnx /tmp/citologia_convnext.onnx   # grava docs/model/ + manifesto.json
cd .. && git add docs/model && git commit -m "Publica modelo" && git push
```

Arquivos acima de 100 MB não entram no GitHub (o script recusa). Ative o Pages em
*Settings → Pages → Deploy from a branch → `main` / `/docs`*. A URL do modelo no app está em
`app/lib/config.dart` (`urlBaseModelo`, ou `--dart-define=URL_BASE_MODELO=...`); a do site em
`docs/js/config.js`. O repositório é `micael-oliveira-ufam/ia_citologia_ufam` (Pages em `https://micael-oliveira-ufam.github.io/ia_citologia_ufam/`).

> A ordem das classes em `lib/data/bethesda.dart` (`ordemDoModelo`: SCC, HSIL, LSIL, NILM) precisa
> bater com a do treinamento. Trocá-la inverte diagnósticos sem erro visível.

### 1b. Catálogo e login Google

O catálogo lê o Firestore do projeto CitoAtlas (`app/lib/config.dart` e `docs/js/config.js`).
Para o login funcionar no app: no Firebase Authentication, ative o provedor Google; no Google
Cloud, crie um cliente OAuth **Android** (pacote do app + SHA-1 da chave de assinatura) e, para
iOS, um cliente **iOS** (`GOOGLE_CLIENTE_IOS=... python3 tools/ajustar_plataformas.py`). Se a chave
de API tiver restrição de aplicativo/referenciador, inclua o app e o domínio do Pages. Use
`firebase/firestore.rules.sugerido` como ponto de partida e **leia o aviso nele** sobre sigilo
das avaliações. O app grava com o token do Firebase; o site também pede login Google para abrir o catálogo e é só leitura. Para o login do site, inclua `https://micael-oliveira-ufam.github.io` em *Origens JavaScript autorizadas* do cliente OAuth web (Google Cloud → Credenciais) e confirme o domínio em *Firebase Authentication → Domínios autorizados*. O banco real exige login para ler e não aceita `orderBy` na listagem (a ordenação é feita por página, no cliente).

### 2. Android

```bash
cd app
./build_apk.sh                 # um APK por arquitetura, em dist/, com SHA-256
./build_apk.sh --universal     # APK único
./build_apk.sh --debug         # depuração
```

O script gera `android/` e `ios/` na primeira execução, fixa `minSdk` em 24 (exigido pelo ONNX
Runtime), aplica a regra de ProGuard, ajusta o nome do app, garante a permissão `INTERNET`
(necessária ao download do modelo e ao catálogo) e a confere no APK. Sem keystore,
o APK sai assinado com a chave de depuração: serve para instalação manual, não para a Play Store.

### 3. iOS

```bash
cd app
flutter create --platforms=ios --project-name colo_ia --org br.ufam.fcf .   # se ios/ ainda não existir
python3 tools/ajustar_plataformas.py    # nome e textos de permissão de câmera/galeria no Info.plist
flutter pub get
cd ios && pod install && cd ..
flutter build ios --release             # assinatura pelo Xcode
```

Sem os textos de permissão (`NSCameraUsageDescription`, `NSPhotoLibraryUsageDescription`) o iOS
fecha o app ao abrir o seletor de imagens; o script os adiciona. O `flutter_onnxruntime`
exige **iOS 16+** e linkagem estática (`use_frameworks! :linkage => :static` no `ios/Podfile`);
ajuste `platform :ios, '16.0'`.

### 4. Modelo de risco (opcional: já vem treinado)

```bash
pip install -r ml/requirements.txt
python3 ml/train.py     # ~3 min; regrava ml/outputs/*, app/assets/risco/modelo_risco.json,
                        # app/assets/graficos/*.png e app/test/casos_referencia.csv
```

Dados: UCI, *Cervical Cancer (Risk Factors)* — 858 pacientes do Hospital Universitário de Caracas,
55 com biópsia positiva (6,4%). Alvo: `Biopsy`. Só variáveis de questionário entram (exames e
`Dx:*` ficam de fora, para não vazar o alvo). Nove algoritmos, validação cruzada 5×5 estratificada.

## O que foi verificado e o que não foi

**Verificado**

- Modelo de risco: o Python reproduz o scikit-learn (diferença de 3e-16 nos 30 casos de
  referência); o `flutter test` repete essa checagem sobre o código Dart.
  O treino é reprodutível: rodado duas vezes, as 200 árvores e as métricas saem idênticas.
- Sintaxe de todos os arquivos Dart (parser tree-sitter), imports entre arquivos e assets
  referenciados no código.
- `ajustar_plataformas.py` (Info.plist e AndroidManifest), em arquivos de exemplo.

**Não verificado: faça antes de distribuir**

- **O app não foi compilado nem executado**: o SDK do Flutter não estava disponível onde o código
  foi escrito. Rode `flutter analyze` e `flutter test` primeiro; podem aparecer erros de
  compilação ou avisos de lint que a revisão estática não pega. Os dois testes de abertura
  (`test/widget_test.dart`) também nunca foram executados.
- O `.onnx` foi gerado a partir do checkpoint v.1.1 (ConvNeXt Large, int8, 188 MB, em 7 partes) e
  o **site** foi testado em Chromium real, inclusive com o modelo real e com o Firestore simulado;
  com o banco real só o fluxo de login foi verificado até onde o Google Cloud permite (veja abaixo).
  O **download do modelo, o catálogo e o login Google do app** são código Dart nunca compilado.
- Redimensionamento: site, treino e Streamlit usam `cv2.INTER_LINEAR` (o site reproduz esse filtro
  e foi conferido contra o OpenCV); o app usa o do pacote `image` (bilinear sem antialiasing), que
  pode divergir um pouco nas probabilidades.
- iOS: nada foi construído. Confirme no pub.dev que a versão usada de `flutter_onnxruntime`
  suporta iOS e qual a versão mínima exigida.
- Estatísticas (`app/assets/dados/estatisticas.json`): 3 de 6 foram conferidas em matérias
  jornalísticas (INCA via InfoMoney; vacina HPV em dose única, Canaltech, abr/2024; transição
  para DNA-HPV, Agência Brasil, mar/2025). As outras 3 (OMS: casos e mortes no mundo, HPV como
  causa, meta 90-70-90) estão com o selo "a confirmar". Os textos de rastreamento refletem o
  que a matéria de março de 2025 dizia, quando ainda faltava a avaliação final do Ministério da
  Saúde: **confirme o que está em vigor** antes de publicar. Revisão por profissional de saúde e
  política de privacidade para as lojas também são recomendadas.

## Limitações (também exibidas no app)

- **Citologia**: treinada só com o IARC Digital Atlas; validação interna (220 imagens, 20
  normais); prevalência do atlas é invertida em relação ao rastreio real; sem aprovação de CEP nem
  registro na ANVISA. Veja a aba *Limitações* (profissional).
- **Risco**: AUC fora da amostra ≈ 0,61 (acaso = 0,50) e p ≈ 0,06 no teste de permutação: sinal
  fraco. Só 55 positivos, um único hospital, sem renda ou escolaridade no conjunto. O modelo não é
  monotônico nas respostas, por isso o app mostra **risco relativo**, nunca diagnóstico.

## Catálogo: ética e segurança

- **Avaliação cega**: site e app nunca mostram a classificação, descrição ou observação de
  outras pessoas; só a contagem de votos. IDs fora do padrão `L-0001` aparecem como código
  derivado ("Lâmina 3FA2"), pois o campo original é texto livre e pode conter nome/prontuário.
- A IA de predição **não** é aplicada às imagens do catálogo, de propósito.
- O catálogo é restrito ao perfil profissional **apenas por escolha na interface** (autodeclarada).
- A senha do painel de desenvolvedor do CitoAtlas original é verificada no navegador e esse
  painel **não foi portado**.
- Material de pacientes numa página pública exige autorização/aprovação ética: confirme antes de
  publicar o site.
