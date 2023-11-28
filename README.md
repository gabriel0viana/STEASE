# STEASE
Sistemas de detecção de anomalias em vídeo desempenham um papel fundamental em câmeras de vigilância, contribuindo para a prevenção de crimes e oferecendo informações cruciais para antecipar a abordagem. Um dos desafios enfrentados por esses sistemas é a perda de contexto no processo de classificação do modelo. Uma hipótese a ser considerada para enfrentar esse desafio é a aplicação de um mecanismo de atenção, com o objetivo de reequilibrar a importância das características no processo de classificação da entrada, podendo melhorar o desempenho do modelo. Foi executado o treinamento do modelo com a base de dados Ped2, com o Squeeze-and-Excitation focando nos canais (cSE), e com o Squeeze-and-Excitation focando nas características espaciais (eSE), em seguida a execução de testes com anomalias nos frames. O nosso modelo com as variantes cSE-encoder-decoder e eSE-encoder-decoder superou a AUC do modelo baseline-laplace executado em nosso servidor em pelo menos 0,12\%. É importante destacar a natureza não determinística da aplicação, o que poderia resultar em variações nos valores finais em testes adicionais.

## Deppendências
* Python 3.6
* PyTorch = 1.7.0 
* Numpy
* Sklearn

* ## Conjunto de dados
* USCD Ped2 [[dataset](https://drive.google.com/file/d/1w1yNBVonKDAp8uxw3idQkUr-a9Gj8yu1/view?usp=sharing)]

## Training
```bash
git clone https://github.com/gabriel0viana/STEASE
```

* Treinamento
```bash
python train.py --dataset_type ped2 --pseudo_anomaly_jump 0.01 --jump 2 3 4 5
```

* ## Resultados
| Modelos                  | Ped2     |
|--------------------------|----------|
| cSE-encoder              | 95.79%   |
| cSE-decoder              | 97.11%   |
| cSE-encoder-decoder      | 98.14%   |
| eSE-encoder-decoder      | 97.99%   |
| cSE-eSE-encoder-decoder  | 96.68%   |
| baseline-laplace         | 97.87%   |
| baseline                 | 98.4%    |

## Avaliação
* Testar o modelo
```bash
python evaluate.py --dataset_type ped2 --model_dir path_to_weight_file.pth
```
* Testar o modelo e salvar a imagem de resultado
```bash
python evaluate.py --dataset_type ped2 --model_dir path_to_weight_file.pth --img_dir folder_path_to_save_image_results
```
* Testar o modelo e gerar quadros de vídeo de demonstração
```bash
python evaluate.py --dataset_type ped2 --model_dir path_to_weight_file.pth --vid_dir folder_path_to_save_video_results
```
Em seguida, compile os quadros em um vídeo
```bash
ffmpeg -framerate 10 -i frame_00_%04d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p video_00.mp4
```
## Base
Este código usa como base o projeto de Astrid, M., Zaheer, M. Z., & Lee, S. I. [[github](https://github.com/aseuteurideu/STEAL)]
