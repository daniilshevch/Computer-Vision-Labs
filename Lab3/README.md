# Лабораторна робота №3

## Тема
Побудова та дослідження моделей на базі **ResNeXt-50** для датасету **Fashion-MNIST**:
1) пряма багатокласова класифікація (10 класів),
2) сіамська мережа для оцінки схожості пар зображень,
3) аналіз простору ознак за допомогою **t-SNE**.

## Мета роботи
- реалізувати (та порівняти) два варіанти ResNeXt-50: власний і на базі `torchvision`;
- адаптувати архітектуру до одноканальних зображень Fashion-MNIST;
- навчити прямий класифікатор на 10 класів одягу;
- реалізувати сіамський підхід для задачі binary similarity (один клас / різні класи);
- оцінити якість моделей за стандартними метриками;
- візуалізувати результати навчання, матриці помилок, ROC-криву і t-SNE проекцію ознак.

## Структура папки `Lab3`
- `CV_Lab3.ipynb` — основний ноутбук з повною реалізацією:
  - архітектура ResNeXt-50;
  - навчання класифікатора;
  - сіамська мережа;
  - оцінка та візуалізації.
- `lab3.ipynb` — альтернативний/додатковий ноутбук.
- `data/FashionMNIST/raw/` — локальні файли датасету Fashion-MNIST.

---

## Що реалізовано

### 1) Власна архітектура ResNeXt-50
У ноутбуці описано:
- `ResNeXtBottleneck` (group convolution, residual shortcut);
- `ResNeXt50FromScratch` (stack bottleneck-блоків 3-4-6-3);
- FC-head з `BatchNorm`, `ReLU`, `Dropout`.

Ключові параметри:
- `cardinality = 32`;
- `first_step_group_width = 4`;
- вхід: `1` канал (grayscale Fashion-MNIST).

### 2) Режими моделі: `custom` і `torchvision`
Через фабричні функції:
- `get_feature_extractor(...)`
- `get_classifier(...)`

Підтримуються два режими:
- **custom** — повністю власна реалізація;
- **torchvision** — `resnext50_32x4d` з модифікацією першого шару під 1 канал і новими head-шарами.

### 3) Пряма класифікація (10 класів)
Реалізовано навчання класифікатора:
- функція втрат: `CrossEntropyLoss`;
- оптимізатор: `Adam(lr=1e-3)`;
- епохи: `10`;
- батчі: `64` (train), `256` (test).

Збираються метрики:
- `Loss`,
- `Accuracy`,
- `Macro F1`,
- `Precision/Recall`.

### 4) Сіамська мережа
Реалізовано:
- `SiameseFashionMNIST` — генерація позитивних/негативних пар у runtime;
- `SiameseNet` — дві спільні гілки feature extractor + similarity head;
- вихідна ймовірність схожості (`Sigmoid`).

Навчання:
- функція втрат: `BCELoss`;
- оптимізатор: `Adam(lr=1e-3)`;
- епохи: `10`;
- метрики: `Loss`, `MAE`, `Accuracy`, `F1`.

### 5) Оцінка та візуалізація
Для прямої класифікації:
- `classification_report`;
- `confusion_matrix`;
- графіки динаміки навчання.

Для сіамської моделі:
- `ROC-AUC`;
- `classification_report`;
- `confusion_matrix`;
- `ROC curve`;
- приклади пар з передбаченою схожістю.

Додатково:
- t-SNE проекція ознак тестового набору Fashion-MNIST у 2D.

---

## Вхідні дані
Використовується `torchvision.datasets.FashionMNIST`:
- train: `train=True`
- test: `train=False`
- автоматичне завантаження: `download=True`
- базова трансформація: `transforms.ToTensor()`

Класи:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

---

## Вихідні результати (файли)
Після виконання ноутбука зберігаються:
- `direct_classification_learning_curves.png`
- `direct_classification_confusion_matrix.png`
- `siamese_learning_curves.png`
- `siamese_confusion_matrix_and_roc.png`
- `siamese_predictions_result.png`
- `clothes_tsne_resnext50.png`

---

## Ключові класи та функції
- `ResNeXtBottleneck`
- `ResNeXt50FromScratch`
- `ResNeXt50Classifier`
- `get_feature_extractor(...)`
- `get_classifier(...)`
- `SiameseNet`
- `SiameseFashionMNIST`
- `evaluate_siamese_metrics(...)`
- `show_model_predictions(...)`
- `visualize_tsne(...)`

---

## Залежності
Необхідні пакети:
- `torch`
- `torchvision`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `seaborn`

Приклад встановлення:
```powershell
pip install torch torchvision numpy matplotlib scikit-learn seaborn
```

---

## Запуск
Рекомендовано запускати у Jupyter / VS Code Notebook:

```powershell
cd "D:\University\4Course\ComputerVisionNew"
jupyter notebook .\Lab3\CV_Lab3.ipynb
```

або відкрити `CV_Lab3.ipynb` у VS Code і виконати клітинки послідовно.

---

## Короткі висновки
- Реалізовано повний deep-learning pipeline для Fashion-MNIST на базі ResNeXt-50.
- Показано два сценарії: пряма класифікація та сіамська верифікація схожості пар.
- Якість оцінюється комплексно: табличні метрики + графічні візуалізації + t-SNE аналіз ознак.
- Порівняння custom/torchvision підходів дає основу для подальших експериментів з архітектурою та гіперпараметрами.

