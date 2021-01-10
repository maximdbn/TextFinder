#!/usr/bin/env python
# coding: utf-8

# In[98]:


import csv
import pytesseract
import numpy as np
import cv2


# In[99]:


class textFinding:
    
    def __init__(self, tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    def getBestBinary(self, image):
        
        # Функция для улучшения изображения с помощью последовательности
        # морфологических и пороговый преобразований
        
        filtered = cv2.adaptiveThreshold(image.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        ret1, th1 = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
        ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blur = cv2.GaussianBlur(th2, (1, 1), 0)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        res = cv2.bitwise_or(th3, closing)
        cv2.imwrite('result\\bestBinary.jpg', res)
        return res
        
    
    def drawRects(self, image, details, confidence):
        
        # Функция выделяет текст, найденный на изображении, прямоугольниками
        print(details)
        total_boxes = len(details['text'])
        for sequence_number in range(total_boxes):
            if (int(details['conf'][sequence_number]) > confidence):
                (x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number],
                                details['width'][sequence_number], details['height'][sequence_number])
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite('result\\boundaries.jpg', image)
        cv2.imshow(image)
    
    def parseText(self, binary, lang):
        
        # Функция ищет текст на бинарном изображении
        
        tesseract_config = r'--oem 3 --psm 6'
        details = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT, config=tesseract_config, lang=lang)
        return details
    
    def formatText(self, details):
        
        # Функция, примерно форматирующая текст в тот вид, который есть на изображении
        
        parse_text = []
        word_list = []
        last_word = ''
        for word in details['text']:
            if word != '':
                word_list.append(word)
                last_word = word
            if (last_word != '' and word == '') or (word == details['text'][-1]):
                parse_text.append(word_list)
                word_list = []

        return parse_text
    
    def writeText(self, formatted_text):
        
        # Функция для найденного записи текста в файл
        
        with open('result\\text.txt', 'w', newline="") as file:
            csv.writer(file, delimiter=" ").writerows(formatted_text)
            
    def process(self, image_path, lang, confidence):
        
        # Функция для полной обработки изображения (поиска, выделения и записи текста)
        
        image = cv2.imread(image_path, 0)
        binary = self.getBestBinary(image)
        parsed_data = self.parseText(binary, lang)
        self.drawRects(binary, parsed_data, confidence)
        arranged_text = self.formatText(parsed_data)
        self.writeText(arranged_text)


# In[ ]:


tesseract_path = str(input('Type your path to tesseract: '))
image_path = str(input('Type your path to image: '))
lang = str(input('Type language: '))


# In[ ]:




