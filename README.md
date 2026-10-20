# ComputerVisionNew

Репозиторій з лабораторними роботами з комп'ютерного зору (4 курс).

## Структура проєкту

- `Lab1/` - класична обробка зображень: гістограми, зміна яскравості, RGB/HSV, морфологія.
- `Lab2/` - локальні ознаки та оцінка афінного перетворення (FAST + BRIEF, custom matcher, custom RANSAC).
- `Lab3/` - класифікація Fashion-MNIST у `Jupyter Notebook` (архітектури ResNeXt/Siamese).
- `Lab4/` - детекція об'єктів у `Jupyter Notebook` (SSD-пайплайн).
- `ШІ-31_Шевченко_Данііл_lab*.docx` - звітні документи по лабораторних.

## Навігація по лабораторних

### Lab 1
- Код: `Lab1/lab1.py`, `Lab1/lab1_colours.py`, `Lab1/extra.py`
- Дані: `Lab1/Images/`
- Опис: `Lab1/README.md`

### Lab 2
- Код: `Lab2/lab2.py`
- Дані: `Lab2/Images/`
- Опис: `Lab2/README.md`

### Lab 3
- Ноутбук: `Lab3/lab3.ipynb`
- Дані: `Lab3/data/FashionMNIST/raw/`

### Lab 4
- Ноутбук: `Lab4/lab4.ipynb`

## Базові залежності
Точний список залежить від конкретної лабораторної, але загалом використовується:
- Python 3.x
- `numpy`
- `matplotlib`
- `opencv-python` / `opencv-contrib-python`
- `torch`, `torchvision` (для Lab3/Lab4)
- `jupyter` (для Lab3/Lab4)

> Примітка: для BRIEF у Lab2 потрібен модуль `cv2.xfeatures2d`, який зазвичай доступний у `opencv-contrib-python`.

## Як запускати

### Lab 1 (скрипти)
```powershell
cd "D:\University\4Course\ComputerVisionNew\Lab1"
python .\lab1.py
python .\lab1_colours.py
python .\extra.py
```

### Lab 2 (скрипт)
```powershell
cd "D:\University\4Course\ComputerVisionNew\Lab2"
python .\lab2.py
```

### Lab 3-4 (ноутбуки)
```powershell
cd "D:\University\4Course\ComputerVisionNew"
jupyter notebook
```
Після запуску відкрий `Lab3/lab3.ipynb` або `Lab4/lab4.ipynb` у браузері.

## Призначення репозиторію
- Зібрати всі лабораторні в одному місці.
- Зафіксувати практичні реалізації з обробки зображень, локальних ознак, класифікації та детекції.
- Спростити перевірку коду і підготовку звітів.

