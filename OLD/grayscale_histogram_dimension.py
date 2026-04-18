'''
Código gerado/revisado utilizando a LLM Gemini 3.1 Pro;
Todo o código e saídas foram revisados pelo autor.
'''

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def process_and_extract_histogram(base_folder, size=(224, 224)):
    # Cria a pasta de destino
    output_folder = f"{base_folder}_Processed_224"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    valid_extensions = ('.jpg', '.jpeg', '.png')
    files = [f for f in os.listdir(base_folder) if f.lower().endswith(valid_extensions)]

    print(f"{len(files)} imagens em {base_folder}")
    histograms = []

    for file in files:
        input_path = os.path.join(base_folder, file)
        output_path = os.path.join(output_folder, file)

        # 1. Carregar imagem BGR
        img_bgr = cv2.imread(input_path)
        if img_bgr is None:
            continue

        # 2. Transformar em Grayscale
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 3. Redimensionar para 224x224 (Padrão CNN)
        img_resized = cv2.resize(img_gray, size, interpolation=cv2.INTER_AREA)

        # 4. Salvar a matriz pronta na nova pasta
        cv2.imwrite(output_path, img_resized)

        # 5. Gerar o histograma da imagem processada
        hist = cv2.calcHist([img_resized], [0], None, [256], [0, 256])

        # 6. Normalizar (transformar contagem em probabilidade)
        hist /= hist.sum()
        histograms.append(hist)

    # Retorna o array médio de todos os histogramas calculados
    if histograms:
        return np.mean(histograms, axis=0)
    else:
        return None


# EXECUÇÃO DO PIPELINE
# Processa as duas classes e extrai as médias
mean_hist_good = process_and_extract_histogram('Good')
mean_hist_bad = process_and_extract_histogram('Bad')

# GERAÇÃO DO GRÁFICO (AED)

if mean_hist_good is not None and mean_hist_bad is not None:
    plt.figure(figsize=(12, 6))

    # Plotagem das duas curvas
    plt.plot(mean_hist_good, label='Tomates Bons (Saudáveis)', color='#2ecc71', linewidth=2.5)
    plt.plot(mean_hist_bad, label='Tomates Ruins (Com Defeito)', color='#e74c3c', linestyle='--', linewidth=2.5)

    plt.title('AED: Comparação de Histogramas Médios (Dataset 224x224 em Escala de Cinza)', fontsize=14, fontweight='bold')
    plt.xlabel('Intensidade de Cinza (0=Preto, 255=Branco)', fontsize=12)
    plt.ylabel('Densidade de Pixels (Frequência Relativa)', fontsize=12)
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlim([0, 255])

    plt.tight_layout()

    # Salva o gráfico
    nome_grafico = 'histograma_comparativo_aed_dataset01.png'
    plt.savefig(nome_grafico, dpi=300)
    print(f"\nSucesso!")

    # Exibe o gráfico na tela
    plt.show()
else:
    print("\nErro.")
