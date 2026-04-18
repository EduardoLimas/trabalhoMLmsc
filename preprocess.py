'''
Código gerado/revisado utilizando a LLM Gemini 3.1 Pro;
Todo o código e saídas foram revisados pelo autor.
'''

import cv2 #version 4.13
import os


def process_datasets_in_root(target_size=(224, 224)):
    """
    Identifica pastas de datasets na raiz, acessa as subpastas Good e Bad,
    converte as imagens para escala de cinza, redimensiona e salva em uma
    nova estrutura de diretórios isolada.
    """
    # 1. Identificar todas as pastas no diretório atual que começam com 'dataset'
    base_path = '.'
    dataset_folders = [d for d in os.listdir(base_path)
                       if os.path.isdir(os.path.join(base_path, d)) and d.startswith('dataset') and not d.endswith(
            '_processed')]

    if not dataset_folders:
        print("Nenhuma pasta com o prefixo 'dataset' foi encontrada.")
        return

    print(f"Datasets encontrados: {dataset_folders}\n")
    valid_extensions = ('.jpg', '.jpeg', '.png')

    # 2. Iterar sobre cada dataset encontrado
    for dataset in dataset_folders:
        print(f"Processando: {dataset}...")

        # 3. Iterar sobre as duas classes esperadas
        for category in ['Good', 'Bad']:
            input_dir = os.path.join(base_path, dataset, category)

            # Pula a categoria se a pasta não existir neste dataset específico
            if not os.path.exists(input_dir):
                continue

            # Define e cria o diretório de saída (ex: dataset01_mendeley_processed/Good)
            output_dataset_dir = f"{dataset}_processed"
            output_dir = os.path.join(base_path, output_dataset_dir, category)
            os.makedirs(output_dir, exist_ok=True)

            files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
            if not files:
                continue

            processed_count = 0
            # 4. Processamento de Imagem
            for file in files:
                input_path = os.path.join(input_dir, file)
                output_path = os.path.join(output_dir, file)

                # Carregamento em BGR
                img_bgr = cv2.imread(input_path)
                if img_bgr is None:
                    continue

                # Transformação para escala de cinza
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

                # Redimensionamento via interpolação por área
                img_resized = cv2.resize(img_gray, target_size, interpolation=cv2.INTER_AREA)

                # Persistência do dado
                cv2.imwrite(output_path, img_resized)
                processed_count += 1

            print(f"  -> [{category}] {processed_count} imagens redimensionadas e convertidas.")

    print("\nExecução finalizada. Dados processados salvos em diretórios isolados.")


# ==========================================
# EXECUÇÃO DO SCRIPT
# ==========================================
if __name__ == "__main__":
    # O tamanho alvo é configurável na chamada da função
    process_datasets_in_root(target_size=(224, 224))