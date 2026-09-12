# Atividade 2 – YOLO Ultralytics: Detecção de Objetos

Notebook de visão computacional usando o _framework_ **Ultralytics YOLO26**, que treina um modelo de **detecção de objetos** focado em identificar o uso de **celulares** em vídeo de vigilância de rua em ambientes urbanos.

## O que o notebook cobre

- **Configuração do ambiente**: instalação da biblioteca `ultralytics` e montagem do Google Drive no Colab.
- **Dataset**: download via API do `objects-phone-v3.ndjson` da Ultralytics Platform (já em NDJSON) e conversão para o formato YOLO (`data.yaml` com splits de treino/validação/teste).
- **Treinamento**: transfer learning a partir do `yolo26n.pt` pré-treinado, podendo usar `best.pt` já existente ou re-treinar (configurado para 20 épocas, `imgsz=640`).
- **Validação e teste**: métricas (P, R, mAP50, mAP50-95) e matriz de confusão sobre splits nunca vistos no treino.
- **Inferência em vídeo**: conversão webm → mp4, predição quadro a quadro e geração do vídeo anotado com as detecções de celular.
- **Persistência**: cópia automática de pesos, resultados e vídeos para o Google Drive.

## Dependências

`ultralytics`, `gdown`, `ffmpeg` (executável) e execução no **Google Colab** (o notebook exige montagem do Drive e API key do Ultralytics Platform).

---

## Contexto dos demais arquivos

- **`objects-phone-v3.ndjson`** – Dataset anotado (formato NDJSON) com frames de vídeo de vigilância urbana marcando pessoas usando celulares, baixado da Ultralytics Platform e convertido para o formato YOLO pelo notebook.
- **`yolo_video_input.mp4`** – Vídeo de entrada usado na etapa final de inferência, convertido e processado para gerar o vídeo anotado com as detecções.

Este notebook compõe a **Atividade 2** do projeto da disciplina, complementando a atividade anterior de visualização de dados com aprendizado de máquina aplicado.