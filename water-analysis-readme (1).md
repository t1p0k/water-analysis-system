# 🌊 Water Analysis System

<div align="center">

![Docker](https://img.shields.io/badge/Docker-20.10+-2496ED.svg)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg)
![Spark](https://img.shields.io/badge/Apache%20Spark-3.0+-E25A1C.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-FF6F00.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

<h3>🛰️ GPU-ускоренная система прогнозирования изменений водных поверхностей</h3>

[Демо](#-демонстрация) • [Установка](#-установка) • [Структура](#-структура-проекта) • [API](#-api-документация)

</div>

---

## 📸 Демонстрация

### 🌍 Анализ водных поверхностей - Salton Sea
<div align="center">
  <img src="gif/water_analysis_demo1.gif" alt="Анализ водных поверхностей" width="100%">
  <p><em>Визуализация изменений водной поверхности с использованием спутниковых данных</em></p>
</div>

### 📊 Прогнозирование изменений на 2025 год
<div align="center">
  <img src="gif/water_analysis_demo2.gif" alt="Прогнозирование изменений" width="100%">
  <p><em>Сегментация воды (UNet) и прогноз изменений (ConvLSTM)</em></p>
</div>

## 💻 Установка

### 📋 Требования

- **Docker** 20.10+
- **Docker Compose** v3.8+
- **NVIDIA GPU** с поддержкой CUDA
- **nvidia-docker** runtime
- **RAM** 16GB+ рекомендуется
- **Disk** 20GB свободного места

### 🚀 Быстрый старт

```bash
# 1. Клонируйте репозиторий
git clone https://github.com/t1p0k/water-analysis-system.git
cd water-analysis-system

# 2. Проверьте структуру проекта
ls -la

# 3. Запустите систему
docker-compose up -d

# 4. Проверьте статус
docker-compose ps
```

### 📦 Доступные сервисы после запуска

| Сервис | URL | Описание |
|--------|-----|----------|
| 🎨 **Streamlit UI** | http://localhost:8501 | Веб-интерфейс |
| 🌐 **API** | http://localhost:8000 | REST API |
| ⚡ **Spark Master** | http://localhost:8080 | Spark кластер |
| 📊 **Grafana** | http://localhost:3000 | Мониторинг (admin/admin) |
| 📈 **Prometheus** | http://localhost:9090 | Метрики |

## 📁 Структура проекта

```
📂 water-analysis-system/
├── 📂 config/                      # Конфигурационные файлы
│   └── 🔧 spark-defaults.conf     # Настройки Spark
├── 📂 data/                        # Данные для анализа
├── 📂 docker/                      # Docker конфигурации
│   ├── 📂 grafana/                # Дашборды Grafana
│   └── 📂 prometheus/             # Конфигурация Prometheus
├── 📂 logs/                        # Логи приложения
├── 📂 models/                      # Обученные модели
│   ├── 🧠 convlstm_water_prediction.h5
│   ├── 🧠 final_unet_model.h5
│   └── 📄 *.identifier            # Метаданные моделей
├── 📂 scripts/                     # Вспомогательные скрипты
│   └── 🐍 simple-water-analysis   # Примеры анализа
├── 📂 src/                         # Исходный код
│   ├── 📂 __pycache__/            # Кэш Python
│   ├── 📂 api/                    # FastAPI приложение
│   ├── 📂 gee/                    # Google Earth Engine
│   ├── 📂 ml/                     # Модели машинного обучения
│   ├── 📂 spark/                  # Spark задачи
│   ├── 📂 ui/                     # Streamlit интерфейс
│   ├── 📂 utils/                  # Утилиты
│   ├── 📂 visualization/          # Визуализация
│   └── 🐍 __init__.py
├── 🐍 run_analysis_simple.py      # Простой запуск анализа
├── 🐍 check_model.py              # Проверка моделей
├── 🐳 docker-compose.yml          # Docker Compose конфигурация
├── 🐳 Dockerfile.spark-gpu        # Dockerfile для Spark с GPU
├── 📄 prometheus.yml              # Конфигурация Prometheus
└── 📄 README.md                   # Этот файл
```

## 🔧 Конфигурация

### Переменные окружения

Создайте файл `.env` для настройки:

```bash
# Spark настройки
SPARK_WORKER_CORES=6
SPARK_WORKER_MEMORY=10G
SPARK_EXECUTOR_MEMORY=4G

# GPU
CUDA_VISIBLE_DEVICES=0

# API
API_HOST=0.0.0.0
API_PORT=8000

# Мониторинг
GRAFANA_ADMIN_PASSWORD=admin
```

### Настройка моделей

Модели находятся в папке `models/`:
- `final_unet_model.h5` - UNet для сегментации воды
- `convlstm_water_prediction.h5` - ConvLSTM для прогнозирования

## 🌐 API Документация

### Основные эндпоинты

```http
POST /api/analyze
Content-Type: application/json

{
  "coordinates": [lat, lon],
  "date_range": ["2021-01-01", "2024-12-31"],
  "model": "unet"
}
```

### Примеры использования

```python
import requests

# Анализ водной поверхности
response = requests.post(
    "http://localhost:8000/api/analyze",
    json={
        "coordinates": [33.3, -115.8],  # Salton Sea
        "date_range": ["2021-01-01", "2024-12-31"],
        "model": "unet"
    }
)

result = response.json()
```

## 📊 Мониторинг

### Grafana дашборды

1. **Spark Cluster** - мониторинг кластера
2. **GPU Utilization** - использование GPU
3. **API Performance** - производительность API
4. **Model Inference** - метрики моделей

### Prometheus метрики

- `spark_executor_count` - количество executor'ов
- `gpu_utilization_percent` - загрузка GPU
- `api_request_duration_seconds` - время ответа API
- `model_inference_time_seconds` - время инференса

## 🛠️ Разработка

### Локальный запуск без Docker

```bash
# Установите зависимости
pip install -r requirements.txt

# Запустите API
uvicorn src.api.main:app --reload

# Запустите UI
streamlit run src/ui/app.py
```

### Добавление новых моделей

1. Поместите модель в `models/`
2. Обновите конфигурацию в `src/ml/config.py`
3. Перезапустите сервисы