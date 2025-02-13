# Background Subtraction - Processamento de Imagens

Este projeto implementa técnicas de **subtração de fundo** para segmentação de objetos em vídeos. Ele permite a geração automática de **máscaras Ground Truth**, avalia os resultados usando métricas quantitativas e oferece um pipeline completo para processamento de imagens.

## Estrutura do Projeto
```
background_subtraction/
│── data/
│   ├── ground_truth/         # Máscaras binárias reais (Ground Truth)
│   ├── processed/            # Resultados processados (máscaras segmentadas)
│   ├── raw/                  # Vídeos e frames originais
│   ├── results/              # Relatórios e métricas calculadas
│
│── src/                      # Código-fonte do projeto
│   ├── background_subtraction.py   # Algoritmo de Background Subtraction
│   ├── create_video.py        # Extração de frames do vídeo
│   ├── evaluate.py            # Avaliação das segmentações (Accuracy, IoU, etc.)
│   ├── generate_ground_truth.py  # Geração das máscaras Ground Truth
│   ├── main.py                # Pipeline completo de execução
│
│── venv/                      # Ambiente virtual (criado automaticamente)
│── README.md                  # ESTE ARQUIVO!
│── requirements.txt            # Dependências do projeto
│── setup_environment.bat       # Script para Windows (configuração do ambiente)
│── setup_environment.sh        # Script para Linux/macOS (configuração do ambiente)
```

---

## **Como Configurar o Ambiente**

Antes de executar o projeto, é necessário configurar o ambiente virtual.

### **Windows**
Abra o **PowerShell** ou **CMD** e execute:
```powershell
start setup_environment.bat
```

### **Linux/macOS**
```bash
chmod +x setup_environment.sh
./setup_environment.sh
```

---

## **Como Executar**

### **Rodar o Pipeline Completo**
Após configurar o ambiente, execute:
```bash
python src/main.py
```
Isso **automatiza todo o processo**:
- Extrai frames do vídeo
- Aplica o **Background Subtraction**
- Gera as máscaras **Ground Truth**
- Avalia os resultados e gera métricas

### **Executar Cada Etapa Manualmente**
Se quiser rodar os scripts **individualmente**, siga esta ordem:

```bash
python src/create_video.py               # Extrai frames do vídeo
python src/background_subtraction.py     # Realiza a segmentação
python src/generate_ground_truth.py      # Gera Ground Truth automaticamente
python src/evaluate.py                   # Avalia a segmentação
```

---

## **Métricas de Avaliação**
O projeto utiliza **4 métricas principais** para medir a qualidade da segmentação:

✔ **Accuracy** → Porcentagem de pixels corretamente classificados  
✔ **Precision** → Quantos pixels positivos foram classificados corretamente  
✔ **Recall** → Quantidade de objetos detectados corretamente  
✔ **IoU (Intersection over Union)** → Medida de sobreposição entre máscara segmentada e Ground Truth  

Os resultados são salvos automaticamente em:
```
data/results/evaluation_results.txt
```

---

### **Tecnologias Utilizadas**
✅ **Python 3.9+**  
✅ **OpenCV** (`cv2`)  
✅ **NumPy**  
✅ **Scikit-Learn**  
✅ **Matplotlib** *(para futuras visualizações de dados)*


