# Textfinder

Выделение текста на изобажении

## Requirements

* pytesseract
* opencv-python
* numpy

Установите библиотеки: pip install requirements.txt

Установите tesseract

## Работа с программой

* Введите в командной строке: python textfinder.py
* Введите путь до tesseract
* Введите путь до изображения
* Введите язык текста на изображении (rus, eng)

Изображение с выделенным текстом, а также документ с распознанным текстом появятся в папке result.

Путь к tesseract и изображению не должен содержать русских символов и пробелов.

