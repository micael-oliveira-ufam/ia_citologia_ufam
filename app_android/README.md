# CitoPred, aplicativo Android

Classificação de campos de citologia em meio líquido pelo Sistema Bethesda,
com o modelo rodando **inteiramente dentro do aparelho**. Sem servidor, sem
envio de imagem, funciona em modo avião.

> Uso acadêmico e demonstrativo. Modelo treinado apenas com o IARC Digital
> Atlas, sem aprovação de Comitê de Ética. Não emite laudo, não substitui o
> citopatologista e não deve receber imagens de pacientes.

> **Paridade com a plataforma web:** os dois usam o mesmo checkpoint e os mesmos
> parâmetros de inferência. O contrato está em `PARAMETROS_COMPARTILHADOS.md` e
> é verificado numericamente pelo exportador.

## Como a inferência local funciona

O checkpoint PyTorch é convertido para ONNX por `tools/export_onnx.py`, que faz
três coisas além da conversão:

1. **Embute a normalização ImageNet no grafo.** A entrada do modelo passa a ser
   RGB em `[0,1]`, `NCHW`, `1x3x224x224`. Assim o pré-processamento em Dart não
   tem como divergir do treino.
2. **Adiciona o mapa de calor como segunda saída.** A cabeça do ConvNeXt é
   `AdaptiveAvgPool2d → LayerNorm2d → Linear`, então projetar o mapa de
   características normalizado pelos pesos da camada linear dá o Class
   Activation Map exato, sem backpropagation, o que seria inviável no celular.
   O modelo devolve `logits [1,4]` e `cam [1,4,7,7]`.
3. **Quantiza os pesos para int8** e confere o resultado contra o PyTorch nas
   imagens de exemplo, avisando se alguma classificação mudou. Nesse caso, o
   aparelho deixaria de concordar com a web.
4. **Mede a correlação** entre o CAM do grafo e o Grad-CAM que a web calcula,
   avisando se cair abaixo de 0,9.

No aparelho, o `flutter_onnxruntime` executa o grafo pela CPU. A decodificação
da imagem e a composição do mapa de calor rodam em isolate, para a interface
não travar.

## Preparar e compilar

```bash
pip install torch torchvision onnx onnxruntime pillow numpy
./build_apk.sh
```

O script baixa o checkpoint do mesmo endereço usado pela plataforma web se ele
não estiver na raiz, exporta para ONNX e compila. Para exportar à mão:

```bash
python3 tools/export_onnx.py \
  --checkpoint convnext_liquid_based_citology_IARC_digital_atlas_01_08_26.pt \
  --saida assets/model/citologia_convnext.onnx
```

**ConvNeXt Large é grande.** Mesmo quantizado, o modelo passa dos 190 MB, o que
torna o APK pesado. Meça o tempo de inferência num aparelho representativo antes
de distribuir: se ficar inviável, a alternativa é treinar uma variante menor: ciente de que isso quebra a equivalência com a plataforma web e exige revalidar
o modelo do aparelho por conta própria.

O `build_apk.sh` confere o ambiente, exporta o modelo se estiver faltando e o
checkpoint estiver na raiz, gera o `android/` na primeira execução, fixa o
`minSdk` em 24 (exigido pelo ONNX Runtime), aplica a regra de ProGuard do
runtime, compila e deposita os APKs em `dist/` com o SHA-256 de cada um.

| Comando | Resultado |
| --- | --- |
| `./build_apk.sh` | um APK por arquitetura, menores |
| `./build_apk.sh --universal` | APK único, roda em qualquer aparelho |
| `./build_apk.sh --debug` | build de depuração |
| `./build_apk.sh --limpar` | limpa antes de compilar |

Sem keystore configurado o APK sai assinado com a chave de depuração: serve
para instalação manual, não para a Play Store. Para assinar de verdade:

```bash
export KEYSTORE_ARQUIVO=/caminho/chave.jks KEYSTORE_SENHA=… KEY_ALIAS=… KEY_SENHA=…
./build_apk.sh
```

Instalação: `adb install -r dist/citologia-ia-*.apk`

## Conferência de que é offline

O build de release do Flutter **não declara a permissão INTERNET**. Ela só
existe nos builds de debug, para o hot reload. O script verifica isso com o
`aapt2` ao final e avisa se algo tiver mudado. É a garantia mais forte de que a
imagem não sai do aparelho: sem essa permissão, o sistema operacional impede
qualquer conexão.

## Estrutura

```
build_apk.sh                gerador do APK
tools/export_onnx.py        conversão PyTorch → ONNX (+ CAM, + quantização)
assets/model/               modelo embarcado (gerado)
assets/exemplos/            campos do IARC Digital Atlas
lib/
  main.dart                 carregamento do modelo e abertura
  tema.dart                 paleta e tipografia
  data/bethesda.dart        as quatro categorias, cores e ícones
  data/projeto.dart         equipe, instituições, métricas, textos
  servicos/imagem.dart      pré-processamento, simetrias, mapa de calor
  servicos/classificador.dart  sessão ONNX e inferência
  telas/                    inicial, resultado, sobre
  widgets/comuns.dart       aviso ético, régua Bethesda, barras
```

## Identidade do aplicativo

O nome é **CitoPred**. O ícone é um microscópio desenhado vetorialmente por
`tools/gerar_icone.py`, que gera o mestre de 1024 px, a camada de frente do
ícone adaptativo e todas as densidades de `mipmap-*`. O `build_apk.sh` chama
esse script e ajusta o `android:label` do manifesto.

Para redesenhar o ícone, edite as coordenadas em `desenhar_microscopio()` e rode:

```bash
python3 tools/gerar_icone.py
```

A tela de abertura (`lib/telas/tela_abertura.dart`) mostra a marca, todos os
participantes agrupados por instituição, os três logos e o aviso de uso
acadêmico. Ela fica visível por no mínimo 2,6 segundos ou até o modelo terminar
de carregar, o que demorar mais.

## Acessibilidade

- Alvos de toque de 56 dp, acima do mínimo de 48 dp.
- Nenhuma informação depende só de cor: cada categoria tem ícone e sigla, e o
  degrau ativo da régua ganha borda grossa e negrito.
- Rótulos semânticos em todos os elementos gráficos; o resultado é anunciado
  por `SemanticsService.announce` para leitores de tela.
- Contraste conferido para WCAG AA.
- Nenhuma altura fixa em contêiner de texto, para o aumento de fonte do sistema
  não cortar conteúdo.

## Ajustes frequentes

- **Equipe e instituições**: `lib/data/projeto.dart`.
- **Métricas exibidas**: classe `Metricas`, no mesmo arquivo.
- **Textos explicativos**: constantes ao final de `projeto.dart`.
- **Ordem das classes**: `ordemDoModelo` em `lib/data/bethesda.dart`: precisa
  bater com a ordem do treinamento (`Carcinoma, HSIL, LSIL, Normal`,
  alfabética). Trocar isso inverte diagnósticos sem erro visível.


## Ferramentas de apoio

O código foi desenvolvido com apoio do modelo de IA generativa Claude Opus 5
(Anthropic). A concepção científica, o treinamento, a validação e a revisão do
código são de responsabilidade da equipe. A informação aparece na tela de
abertura e na aba Sobre.
