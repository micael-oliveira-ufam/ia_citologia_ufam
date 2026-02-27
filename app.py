import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import urllib.request # <--- Biblioteca necessária para baixar o modelo pesado da nuvem

# =============================================================================
# SEÇÃO 1: CONFIGURAÇÕES DE INTERFACE E CAMINHOS
# =============================================================================
st.set_page_config(page_title="IA Citologia - UFAM & SEMSA", page_icon="🔬", layout="wide")

# Caminhos de arquivos (Usando caminho relativo para funcionar no servidor do Streamlit)
CURRENT_DIRECTORY = os.getcwd()
FILE_NAME = "melhor_modelo_ResNet50_citologia_meio_liquido.pt"
MODEL_PATH = os.path.join(CURRENT_DIRECTORY, FILE_NAME)

# !!! ATENÇÃO: COLE O LINK DO SEU GITHUB RELEASES AQUI DENTRO DAS ASPAS !!!
MODEL_URL = "https://github.com/micael-oliveira-ufam/ia_citologia_ufam/releases/download/v.1.0/melhor_modelo_ResNet50_citologia_meio_liquido_2026-02-26.pt"

# Logos Institucionais
LOGO_UFAM_PATH = os.path.join(CURRENT_DIRECTORY, "logo_ufam.png")
LOGO_SEMSA_PATH = os.path.join(CURRENT_DIRECTORY, "semsa_logo.jpg")

CLASS_NAMES = [
    "Squamous cell carcinoma (SCC)",
    "High squamous intra-epithelial lesion (HSIL)",
    "Low squamous intra-epithelial lesion (LSIL)",
    "Negative for intra-epithelial malignancy (NILM)"
]
NUM_CLASSES = len(CLASS_NAMES)

# =============================================================================
# SEÇÃO 2: ARQUITETURA, DOWNLOAD E LÓGICA DO MODELO
# =============================================================================

@st.cache_resource
def download_model():
    """Baixa o modelo do GitHub Releases se ele não existir na pasta local."""
    if not os.path.exists(MODEL_PATH):
        st.info("⬇️ Baixando o modelo de IA do servidor (Isso ocorre apenas na primeira execução. Aguarde...)")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            st.success("✅ Download do modelo concluído!")
        except Exception as e:
            st.error(f"Erro ao baixar o modelo. Verifique se o link em MODEL_URL está correto. Erro: {e}")
            st.stop()

@st.cache_resource 
def load_trained_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=False)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, NUM_CLASSES)
    )
    
    # Executa a função de download antes de tentar carregar o arquivo
    download_model()
    
    if not os.path.exists(model_path):
        st.error(f"⚠️ Arquivo do modelo não encontrado:\n{model_path}")
        st.stop()
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def get_infer_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_visualization_transforms():
    return {
        "Giro Horizontal": A.HorizontalFlip(p=1.0),
        "Rotação 90°": A.RandomRotate90(p=1.0),
        "Variação de Cor/Brilho": A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, p=1.0),
    }

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _hook_forward(self, module, input, output):
        self.activations = output

    def _hook_backward(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def _register_hooks(self):
        self.target_layer.register_forward_hook(self._hook_forward)
        self.target_layer.register_backward_hook(self._hook_backward)

    def __call__(self, input_tensor, target_class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if target_class_idx is None:
            target_class_idx = torch.argmax(output)
        output[0][target_class_idx].backward(retain_graph=True)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        max_val = torch.max(heatmap)
        if max_val > 0:
            heatmap /= max_val
        return heatmap.detach().cpu().numpy()

def predict_and_visualize_all(model, device, original_img, class_names):
    original_img_np = np.array(original_img)
    infer_transform = get_infer_transform()
    input_tensor = infer_transform(image=original_img_np)['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(probabilities).item()
        pred_prob = probabilities[pred_idx].item()
        pred_class_name = class_names[pred_idx]

    grad_cam = GradCAM(model, model.layer4[-1])
    heatmap = grad_cam(input_tensor, target_class_idx=pred_idx)
    heatmap_resized = cv2.resize(heatmap, (original_img_np.shape[1], original_img_np.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlaid_img = cv2.addWeighted(original_img_np, 0.6, heatmap_colored, 0.4, 0)

    augmented_images = {name: transform(image=original_img_np)['image'] for name, transform in get_visualization_transforms().items()}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    axes[0].imshow(original_img_np)
    axes[0].set_title('Imagem Original', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(overlaid_img)
    axes[1].set_title('Mapa de Calor (Grad-CAM)', fontsize=14)
    axes[1].axis('off')

    vis_items = list(augmented_images.items())
    for i in range(3):
        if i < len(vis_items):
            name, img = vis_items[i]
            axes[i+2].imshow(img)
            axes[i+2].set_title(f'{name}', fontsize=14)
        axes[i+2].axis('off')

    ax_bar = axes[5]
    ax_bar.axis('on')
    y_pos = np.arange(len(class_names))
    bars = ax_bar.barh(y_pos, probabilities.cpu().numpy(), align='center', color='#87CEFA', height=0.6)
    bars[pred_idx].set_color('#4169E1')
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(class_names, fontsize=10)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel('Probabilidade', fontsize=12)
    ax_bar.set_title('Confiança por Classe', fontsize=14)
    ax_bar.set_xlim(0, 1.1)
    
    for i, prob in enumerate(probabilities.cpu().numpy()):
        ax_bar.text(prob + 0.02, i, f'{prob:.1%}', va='center', fontsize=11, weight='bold' if i == pred_idx else 'normal')

    plt.tight_layout()
    return fig, pred_class_name, pred_prob

# =============================================================================
# SEÇÃO 3: CONSTRUÇÃO DA INTERFACE (UI/UX)
# =============================================================================
def build_ui():
    # --- CABEÇALHO ---
    col1, col2, col3 = st.columns([1, 6, 1])
    
    with col1:
        if os.path.exists(LOGO_UFAM_PATH):
            st.image(LOGO_UFAM_PATH, width=120)
        else:
            st.markdown("*(Logo UFAM)*")

    with col2:
        st.title("Sistema de Apoio ao Diagnóstico Citológico")
        st.markdown("Classificação automatizada de citologia em meio líquido auxiliada por Inteligência Artificial.")

    with col3:
        if os.path.exists(LOGO_SEMSA_PATH):
            st.image(LOGO_SEMSA_PATH, width=120)
        else:
            st.markdown("*(Logo SEMSA)*")

    st.markdown("---")

    # --- BARRA LATERAL (SIDEBAR) ---
    st.sidebar.header("📁 Entrada de Dados")
    uploaded_file = st.sidebar.file_uploader("Selecione a lâmina digitalizada", type=["jpg", "jpeg", "png", "tif", "tiff"])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎓 Sobre o Projeto")
    st.sidebar.markdown("""
    Pesquisa acadêmica voltada ao avanço tecnológico na saúde pública.
    
    **Desenvolvedor:** Micael Davi Lima de Oliveira *(Iniciação Científica)* **Coordenação:** Profº Dr. Toni Ricardo Martins  
    
    **Instituição:** Faculdade de Ciências Farmacêuticas - UFAM  
    **Parceria:** Laboratório Sebastião Marinho (SEMSA)
    """)
    st.sidebar.caption("Modelo: ResNet50 (Digital Atlas IARC)")

    # --- ÁREA PRINCIPAL ---
    # Carrega o modelo (e faz o download caso necessário)
    with st.spinner("Inicializando motor de Inteligência Artificial..."):
        model, device = load_trained_model(MODEL_PATH)

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        col_img, col_action = st.columns([1, 2])
        
        with col_img:
            st.image(image, caption="Lâmina Submetida", use_container_width=True)
            
        with col_action:
            st.markdown("### Processamento Pronto")
            st.info("A imagem foi devidamente carregada no sistema. Clique abaixo para acionar a rede neural.")
            analyze_button = st.button("🔍 Executar Análise Citológica", type="primary", use_container_width=True)

        if analyze_button:
            st.markdown("---")
            st.subheader("📊 Resultados da Inferência e Interpretação")
            
            with st.spinner('Extraindo características celulares e calculando probabilidades...'):
                try:
                    fig, pred_class, prob = predict_and_visualize_all(model, device, image, CLASS_NAMES)
                    
                    if prob > 0.85:
                        st.success(f"**Diagnóstico Sugerido pela IA:** {pred_class}  \n**Grau de Confiança do Algoritmo:** {prob:.2%} (Alta confiabilidade para suporte à decisão)")
                    elif prob > 0.60:
                        st.warning(f"**Diagnóstico Sugerido pela IA:** {pred_class}  \n**Grau de Confiança do Algoritmo:** {prob:.2%} (Confiabilidade moderada. Recomenda-se atenção aos diferenciais)")
                    else:
                        st.error(f"**Diagnóstico Sugerido pela IA:** {pred_class}  \n**Grau de Confiança do Algoritmo:** {prob:.2%} (Baixa confiabilidade. Necessária revisão minuciosa pelo citopatologista)")
                    
                    st.pyplot(fig)
                    
                    with st.expander("🔬 Como a IA chegou a esse resultado? (Interpretabilidade)"):
                        st.markdown("""
                        Este painel fornece transparência ao processo de decisão do modelo:
                        * **Mapa de Calor (Grad-CAM):** As regiões destacadas em cores quentes (vermelho/laranja) mostram exatamente quais agrupamentos celulares ou alterações nucleares a IA considerou mais suspeitas para definir a classe.
                        * **Aumentos (Data Augmentation):** O modelo é testado contra rotações e variações de iluminação para garantir que a predição não seja um falso positivo causado pela qualidade da foto.
                        * **Gráfico de Probabilidades:** Detalha o "nível de dúvida" do modelo entre as classes do Sistema Bethesda.
                        """)
                        
                except Exception as e:
                    st.error(f"Ocorreu um erro durante a inferência: {e}")
    else:
        # TELA INICIAL (Quando não há upload)
        st.markdown("### Bem-vindo à Plataforma de Rastreio Inteligente")
        st.write("Esta ferramenta atua como um sistema de suporte à decisão clínica (CDSS), projetada para auxiliar citopatologistas na triagem e análise de exames preventivos do câncer do colo do útero.")
        
        st.markdown("---")
        
        col_info1, col_info2, col_info3 = st.columns(3)
        
        with col_info1:
            st.markdown("#### 🩸 Importância do Rastreio")
            st.write("""
            O câncer do colo do útero é altamente evitável, mas continua sendo uma questão crítica de saúde pública. 
            O rastreamento eficiente por meio do exame Papanicolaou (e sua evolução para a Citologia em Meio Líquido) 
            permite detectar lesões precursoras anos antes de se tornarem malignas. Esta IA visa agilizar essa detecção, 
            potencializando diagnósticos precoces em grandes volumes de amostras.
            """)
            
        with col_info2:
            st.markdown("#### 🎯 Objetivo da Ferramenta")
            st.write("""
            Omitir o cansaço visual humano e fornecer uma "segunda opinião" automatizada. 
            O modelo lê a lâmina digitalizada e a categoriza rapidamente segundo as diretrizes do **Sistema Bethesda**, 
            apontando desde amostras normais (NILM) até lesões de alto grau (HSIL) e carcinomas (SCC), priorizando casos urgentes.
            """)
            
        with col_info3:
            st.markdown("#### ⚙️ Qualidade e Acurácia")
            st.write("""
            Construído sobre a robusta arquitetura de redes neurais **ResNet50**, o algoritmo foi submetido a técnicas 
            de *Transfer Learning* e balanceamento de dados rigoroso. Validado com métricas acadêmicas (Acurácia, Precisão, Recall e F1-Score), 
            o modelo também conta com o sistema **Grad-CAM**, que elimina a "caixa preta" da IA, mostrando visualmente onde a rede neural encontrou a lesão.
            """)
            st.caption("*Nota: O modelo apresenta alta sensibilidade para lesões pré-malignas no conjunto de validação IARC.*")

if __name__ == '__main__':
    build_ui()