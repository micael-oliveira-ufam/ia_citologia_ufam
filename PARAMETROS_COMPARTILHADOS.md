# Parâmetros compartilhados: plataforma web e aplicativo Android

A plataforma web (`app.py`) e o aplicativo Android usam o **mesmo checkpoint** e
precisam produzir a **mesma classificação** para a mesma imagem. Este documento
é o contrato entre os dois. Alterar qualquer linha aqui exige alterar os dois
lados na mesma mudança.

## Modelo

| Item | Valor |
| --- | --- |
| Checkpoint | `convnext_liquid_based_citology_IARC_digital_atlas_01_08_26.pt` |
| Conjuntos de treino | IARC Digital Atlas e Liquid Based Cytology Pap Smear (Mendeley Data) |
| Origem | GitHub Releases `micael-oliveira-ufam/ia_citologia_ufam`, tag `v.1.1` |
| Arquitetura | ConvNeXt Large (torchvision), pesos iniciais do ImageNet |
| Cabeça | `classifier[2] = Sequential(Dropout(0.5), Linear(in_features, 4))` |
| Classes | `["SCC", "HSIL", "LSIL", "NILM"]` |

A ordem das classes vem das pastas do `ImageFolder` no treino
(`Carcinoma / HSIL / LSIL / Normal`, alfabética) e é a mesma dos eixos da matriz
de confusão. **Trocar essa ordem inverte diagnósticos sem gerar erro visível.**

- Web: `CLASSES` em `app.py`
- Android: `ordemDoModelo` em `lib/data/bethesda.dart`
- Exportador: `CLASS_NAMES` em `tools/export_onnx.py`

## Pré-processamento de inferência

Idêntico nos dois. Os aumentos de treino (giros, rotações, jitter de cor) **não**
entram na inferência.

| Etapa | Valor |
| --- | --- |
| Redimensionamento | 224 × 224, interpolação bilinear |
| Escala | `[0, 1]` (divisão por 255) |
| Normalização | média `(0.485, 0.456, 0.406)`, desvio `(0.229, 0.224, 0.225)` |
| Layout | NCHW, `1 × 3 × 224 × 224`, float32 |

Na web, `A.Resize` usa `cv2.INTER_LINEAR`; no Android, `img.copyResize` com
`Interpolation.linear`. São o mesmo filtro.

No Android a normalização está **embutida no grafo ONNX**, o Dart entrega RGB
em `[0,1]` e o grafo aplica média e desvio. Isso elimina a chance de o
pré-processamento do aplicativo divergir do treino por um erro de digitação.

## Verificação de robustez

Mesma definição nos dois: as oito simetrias do quadrado (grupo diedral D4), que
são exatamente as transformações vistas no treinamento.

| Item | Valor |
| --- | --- |
| Número de vistas | 8 (4 rotações × 2 espelhamentos) |
| Probabilidade final relatada | leitura direta (a média das 8 é informativa) |
| Concordância | fração das 8 vistas cuja classe coincide com a leitura direta |

O sentido da rotação não precisa coincidir entre as implementações: o conjunto
das oito vistas é o mesmo grupo, então a média e a fração de concordância
independem da convenção.

## Limiares de alerta

| Condição | Valor | Mensagem |
| --- | --- | --- |
| Confiança baixa | `< 0.60` | Predição instável, tratar como indefinido |
| Concordância baixa | `< 0.75` | Idem |
| Classe da média ≠ leitura direta |: | Sugere repetir com outra imagem |

- Web: literais em `mostrar_resultado()`
- Android: `limiarConfianca` e `limiarConcordancia` em
  `lib/servicos/classificador.dart`

## Mapa de calor

Os dois mostram o **mesmo Grad-CAM**, calculado sobre o último bloco
convolucional.

A web usa autograd. O Android não pode: o ONNX Runtime móvel não faz
backpropagation. Em vez de aproximar, o grafo exportado calcula o gradiente em
forma fechada.

A cabeça do ConvNeXt é `AdaptiveAvgPool2d -> LayerNorm2d -> Flatten -> Dropout
-> Linear`. Fosse apenas pooling seguido de camada linear, o gradiente do logit
seria constante no espaço e igual a `w_k/(H·W)`, e bastaria projetar o mapa
pelos pesos. A LayerNorm no meio impede essa simplificação, porque é não linear
e seu jacobiano depende das estatísticas do vetor agrupado. Com `p` o vetor
agrupado, `mu` e `sigma` a média e o desvio sobre os canais, `z = (p-mu)/sigma`
e `u_k = gamma * w_k`:

```
d logit_k / dp = (1/sigma) * [ u_k - media(u_k) - media(u_k * z) * z ]
```

O grafo implementa exatamente isso e projeta o mapa **bruto** de
características. O fator `1/(H·W)` desaparece na normalização pelo máximo.

Uma versão anterior deste exportador projetava o mapa **normalizado** pelos
pesos lineares, ignorando o jacobiano da LayerNorm. A correlação com o Grad-CAM
real ficava em torno de 0,87, e não em 1. O erro foi encontrado justamente pela
verificação descrita abaixo.

O ramo do mapa de calor vive no submódulo `cabeca_cam`, o que faz seus nós ONNX
saírem com o prefixo `/cabeca_cam/`. Eles são excluídos da quantização por esse
prefixo: têm poucos parâmetros, então não há ganho de tamanho, e seus matmuls
com peso constante eram os que o quantizador reescrevia de forma incompatível.

### Captura do gradiente na web

O Grad-CAM da plataforma captura o gradiente com um hook no **tensor** de
ativação, não com `register_full_backward_hook` no módulo. Os dois dão o mesmo
número, mas o hook de módulo tem um contrato traiçoeiro: qualquer valor
devolvido substitui o `grad_input`, cuja forma difere do `grad_output` neste
bloco, e o autograd aborta com `hook has changed the size of value`.

## Verificação da equivalência

O exportador confere, antes de gravar o arquivo:

1. o módulo exportável reproduz o modelo original (diferença máxima < 1e-4);
2. o ONNX float32 reproduz o PyTorch nas imagens de exemplo (desvio máximo de
   probabilidade e número de classificações divergentes);
3. o ONNX int8 reproduz o float32: se a quantização mudar alguma
   classificação, o script avisa, porque aí o aparelho deixaria de concordar
   com a web;
4. o mapa de calor do grafo correlaciona com o Grad-CAM do PyTorch. Em
   float32 a correlação é 1,000000; em int8 fica acima de 0,999. Abaixo de
   0,99 o script avisa, porque aí a cabeça do modelo não é a esperada.

Se o int8 não passar em qualquer uma dessas conferências, o script descarta o
arquivo quantizado e mantém o float32. Um modelo menor que classifica diferente
da plataforma web não serve para nada.

Para uma checagem manual, analise `cyto5940.jpg` e `cyt14686a.jpg` nos dois: as
classes e as probabilidades devem bater até a primeira casa decimal.

## Métricas de validação exibidas

Mesmos números nos dois, derivados da matriz de confusão global da validação
cruzada 5-folds sobre 1096 imagens.

- Web: `METRICAS` e `MATRIZ_CONFUSAO` em `app.py`
- Android: `Metricas` em `lib/data/projeto.dart`
