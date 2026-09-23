# 3 - Inferência ONNX no Navegador com Next.js

Aplicação web **YONO (You Only Need ONNX)** para executar modelos de visão computacional diretamente no navegador: modelo → pré-processamento → inferência → pós-processamento → visualização dos resultados.

## Aplicação web

Clique [neste link](https://yolo-web-liart.vercel.app/) para acessar a aplicação publicada na Vercel:
`https://yolo-web-liart.vercel.app/`.

Toda a inferência acontece localmente no navegador com WebAssembly. As imagens, os vídeos e os modelos enviados pelo usuário não precisam ser processados por um servidor da aplicação.

## Visão geral

- **Modelo padrão**: detector YOLO exportado para ONNX, carregado automaticamente ao abrir a página.
- **Modelos personalizados**: envio de arquivos `.onnx` com entrada RGB em formato NCHW ou NHWC.
- **Tarefas suportadas**: detecção de objetos e classificação de imagens.
- **Mídia**: duas imagens e dois vídeos de exemplo, além do envio de arquivos próprios.
- **Inferência**: execução com `onnxruntime-web`, WebAssembly e uma thread.
- **Resultados**: caixas de detecção ou as cinco classes mais prováveis, com ajuste do limite de confiança.
- **Idiomas**: interface e classes em inglês ou português.
- **Classes**: leitura dos nomes armazenados nos metadados do modelo ONNX e tradução dinâmica de classes desconhecidas.
- **Publicação**: exportação estática do Next.js hospedada na Vercel.

## Como usar

1. Abra a [aplicação YONO](https://yolo-web-liart.vercel.app/).
2. Aguarde o indicador **Model Ready**.
3. Escolha uma mídia de exemplo ou envie uma imagem ou vídeo.
4. Ajuste o limite de confiança conforme necessário.
5. Alterne entre **EN** e **PT** para traduzir a interface e os nomes das classes.
6. Para testar outro modelo, clique em **Custom ONNX model** e selecione um arquivo `.onnx`.

> Modelos sem nomes de classes nos metadados continuam funcionando, mas as saídas são exibidas como `class 0`, `class 1`, etc. A ordem e o significado dessas classes devem ser fornecidos junto ao modelo.

## Dependências

`next`, `react`, `react-dom`, `onnxruntime-web`, `tailwindcss`, `typescript`
