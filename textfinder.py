#!/usr/bin/env python
# coding: utf-8

# In[98]:


import csv
import pytesseract
import numpy as np
import cv2


# In[1]:


class textFinding:
    
    def __init__(self, tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    def getBestBinary(self, image):
        
        # Функция для улучшения изображения с помощью последовательности
        # морфологических и пороговый преобразований
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered = cv2.adaptiveThreshold(image.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 3)
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        ret, thresh1 = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
        ret, thresh2 = cv2.threshold(thresh1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blur = cv2.GaussianBlur(thresh2, (1, 1), 0)
        ret, thresh3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        res = cv2.bitwise_or(thresh3, closing)
        kernel1 = np.ones((3, 3), np.uint8)
        res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel1)
        kernel2 = np.ones((1, 1), np.uint8)
        res = cv2.erode(res, kernel2)
        return res
    
    def getBinary(self, image):
        
        # Функция для перевода изображения в черно-белое
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered = cv2.adaptiveThreshold(image.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 45, 3)
        return filtered
        
    
    def drawRects(self, image, details, confidence):
        
        # Функция выделяет текст, найденный на изображении, прямоугольниками
        total_boxes = len(details['text'])
        for sequence_number in range(total_boxes):
            if (int(details['conf'][sequence_number]) > confidence):
                (x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number],
                                details['width'][sequence_number], details['height'][sequence_number])
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite('result\\boundaries.jpg', image)
    
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
        
        with open('result\\text.txt', 'w', newline="", encoding='utf-8') as file:
            csv.writer(file, delimiter=" ").writerows(formatted_text)
            
    def lowBrightness(self, image):
        
        # Функция изменяет яркость и насыщенность изображения для более четкого распознавания
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if (hsv[i][j][2] < 40):
                    hsv[i][j][2] += 20
                if (hsv[i][j][2] > 110):
                    hsv[i][j][2] -= 20
                if (hsv[i][j][1] > 110):
                    hsv[i][j][1] /= 1.5
                
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return image
    
    def getBestRes(self, text1, text2):
        
        # Функция, которая выбирает лучший из двух вариантов текста
        
        cnt1 = 0
        cnt2 = 0
        for i in range(len(text1)):
            for j in range(len(text1[i])):
                if (len(text1[i][j]) > 1):
                    cnt1 += 1
                else:
                    cnt1 -= 1
        for i in range(len(text2)):
            for j in range(len(text2[i])):
                if (len(text2[i][j]) > 1):
                    cnt2 += 1
                else:
                    cnt2 -= 1
        return (cnt1 >= cnt2)
            
    def process(self, image_path, lang, confidence):
        
        # Функция для полной обработки изображения (поиска, выделения и записи текста)
        
        image = cv2.imread(image_path, 1)
        lowImg = self.lowBrightness(image)
        binary1 = self.getBestBinary(lowImg)
        parsed_data1 = self.parseText(binary1, lang)
        arranged_text1 = self.formatText(parsed_data1)
        
        binary2 = self.getBinary(lowImg)
        parsed_data2 = self.parseText(binary2, lang)
        arranged_text2 = self.formatText(parsed_data2)
        
        if (self.getBestRes(arranged_text1, arranged_text2)):
            self.drawRects(image, parsed_data1, confidence)
            self.writeText(arranged_text1)
        else:
            self.drawRects(image, parsed_data2, confidence)
            self.writeText(arranged_text2)


# In[ ]:


tesseract_path = str(input('Type your path to tesseract: '))
image_path = str(input('Type your path to image: '))
lang = str(input('Type language: '))


# In[ ]:


model = textFinding(tesseract_path)
model.process(image_path, lang, 30)

