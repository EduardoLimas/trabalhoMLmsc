# trabalhoMLmsc

Dataset 01 Source:
https://data.mendeley.com/datasets/hfyth5t3gg/1

We did some data cleaning and modified the dataset. To check the original dataset, please visit the link above.
To cut the images we did some in manual mode (native editor in Windows) and some using the batch feature from https://pt.imgtools.co/crop-image


Dataset 02 Source: authors
Images were captured using a Samsung S24+ mobile camera.
We used daylight conditions.

Dataset 03 Source:
Dataset 01 + Dataset 02

[preprocess.py](preprocess.py) --> process the images (RGB to grayscale, resize)

[labels.py](labels.py) --> create the labels for the images

As amostras de treino e teste foram separadas em 80% (treino) e 20% (teste), sendo separadas de forma aleatória para cada um dos datasets.

Gabaritos --> 1 = Good, 0 = Bad

Obter datos > Montar dataset original > split train test and labels > preprocess images > EDA por pastas > treinar e testar

Para minimizar erros, a EDA será realizada na pasta de cada um dos datasets.