'''
Código gerado utilizando a LLM Gemini 3.1 Pro;
Todo o código e saídas foram revisados pelo autor.
'''


import os
import shutil
import csv
from sklearn.model_selection import train_test_split


def split_datasets_and_generate_csv(base_path='.', test_ratio=0.2):
    """
    Identifica diretórios de datasets, segrega as imagens em treino e teste
    de forma estratificada e gera os CSVs de gabarito para cada subconjunto.
    """
    # Identifica pastas de dataset originais (ignorando as já processadas ou divididas)
    dataset_folders = [d for d in os.listdir(base_path)
                       if os.path.isdir(os.path.join(base_path, d))
                       and d.startswith('dataset')
                       and not d.endswith('_split')
                       and not d.endswith('_processed')]

    if not dataset_folders:
        print("Erro: Nenhum diretório de dataset bruto encontrado.")
        return

    # Dicionário formal de classes
    class_mapping = {'Bad': 0, 'Good': 1}

    for dataset in dataset_folders:
        print(f"Processando segregação para: {dataset}")

        all_files = []
        all_labels = []

        # 1. Coleta de caminhos e rótulos
        for category, label in class_mapping.items():
            category_path = os.path.join(base_path, dataset, category)
            if not os.path.exists(category_path):
                continue

            files = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for file in files:
                # Armazena o caminho original e o rótulo
                all_files.append((os.path.join(category_path, file), file, category))
                all_labels.append(label)

        if not all_files:
            print(f"  -> Aviso: Nenhuma imagem encontrada em {dataset}. Ignorando.")
            continue

        # 2. Divisão Estratificada
        # Garante que a proporção 50/50 de bons e ruins seja mantida no treino e no teste
        X_train, X_test, y_train, y_test = train_test_split(
            all_files, all_labels,
            test_size=test_ratio,
            random_state=42,  # Semente fixa para reprodutibilidade do experimento
            stratify=all_labels
        )

        # 3. Criação da estrutura de saída (ex: dataset01_mendeley_split)
        output_base = os.path.join(base_path, f"{dataset}_split")
        train_dir = os.path.join(output_base, 'train')
        test_dir = os.path.join(output_base, 'test')

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # 4. Função auxiliar para copiar arquivos e registrar no CSV
        def process_split(data_split, labels_split, target_dir, csv_filename):
            csv_path = os.path.join(output_base, csv_filename)
            rows = []

            for (original_path, filename, category), label in zip(data_split, labels_split):
                # Prefixo adicionado para evitar colisão de nomes se houver imagens com nomes idênticos nas pastas Good e Bad
                safe_filename = f"{category}_{filename}"
                target_path = os.path.join(target_dir, safe_filename)

                shutil.copy2(original_path, target_path)
                rows.append([safe_filename, label])

            # Escrita do Gabarito
            with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Nome', 'Label'])
                writer.writerows(rows)

            return len(rows)

        # 5. Execução da cópia e geração dos CSVs
        train_count = process_split(X_train, y_train, train_dir, 'train_gabarito.csv')
        test_count = process_split(X_test, y_test, test_dir, 'test_gabarito.csv')

        print(f"  -> Concluído: {train_count} amostras de treino, {test_count} amostras de teste.")

    print("\nSegregação metodológica finalizada.")


if __name__ == "__main__":
    split_datasets_and_generate_csv(test_ratio=0.2)