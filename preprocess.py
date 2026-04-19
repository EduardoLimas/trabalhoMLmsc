'''
Código gerado utilizando a LLM Gemini 3.1 Pro;
Todo o código e saídas foram revisados pelo autor.
'''

import os
import cv2
import shutil

def generate_processed_datasets(base_path='.', target_size=(224, 224)):
    """
    Percorre os diretórios '_split', aplica conversão Grayscale e Resize (224x224)
    nas instâncias de Treino e Teste, e espelha a estrutura em pastas '_processed',
    preservando os arquivos de anotação (Gabaritos).
    """
    # 1. Identificação estrita dos diretórios de entrada
    split_folders = [d for d in os.listdir(base_path)
                     if os.path.isdir(os.path.join(base_path, d)) and d.endswith('_split')]

    if not split_folders:
        print("Erro Crítico: Nenhum diretório base com sufixo '_split' localizado.")
        return

    valid_extensions = ('.jpg', '.jpeg', '.png')

    for split_folder in split_folders:
        # Substituição de namespace para o diretório de saída
        processed_folder = split_folder.replace('_split', '_processed')
        processed_path = os.path.join(base_path, processed_folder)
        os.makedirs(processed_path, exist_ok=True)

        print(f"\n[ Início ] Processamento: {split_folder} -> {processed_folder}")

        # 2. Processamento Espacial e Espectral (Train / Test)
        for subset in ['train', 'test']:
            input_dir = os.path.join(base_path, split_folder, subset)
            output_dir = os.path.join(processed_path, subset)

            if not os.path.exists(input_dir):
                print(f"  -> Aviso: Partição '{subset}' não encontrada. Ignorando.")
                continue

            os.makedirs(output_dir, exist_ok=True)
            files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
            processed_count = 0

            for file in files:
                input_file_path = os.path.join(input_dir, file)
                output_file_path = os.path.join(output_dir, file)

                # Leitura em 3 canais
                img_bgr = cv2.imread(input_file_path)
                if img_bgr is None:
                    continue

                # Redução de dimensionalidade espectral (1 canal)
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

                # Otimização de downsampling para evitar perda de textura crítica (aliasing)
                img_resized = cv2.resize(img_gray, target_size, interpolation=cv2.INTER_AREA)

                # Persistência da matriz
                cv2.imwrite(output_file_path, img_resized)
                processed_count += 1

            print(f"  -> [Partição: {subset.upper()}] {processed_count} matrizes convertidas ({target_size[0]}x{target_size[1]}).")

        # 3. Migração de Metadados (Gabaritos CSV)
        for csv_file in ['train_gabarito.csv', 'test_gabarito.csv']:
            src_csv = os.path.join(base_path, split_folder, csv_file)
            dst_csv = os.path.join(processed_path, csv_file)

            if os.path.exists(src_csv):
                # shutil.copy2 preserva os timestamps do SO, garantindo auditabilidade
                shutil.copy2(src_csv, dst_csv)
                print(f"  -> Gabarito replicado: {csv_file}")
            else:
                print(f"  -> Falha de Metadados: {csv_file} ausente no diretório de origem.")

    print("\n[ Fim ] Transformações matriciais e espelhamento estrutural concluídos.")

if __name__ == "__main__":
    generate_processed_datasets()