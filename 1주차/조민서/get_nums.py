import cv2
import numpy as np
from inference import read_num
def get_nums(img):
    #256,256
    kernel = np.ones((7,7),np.uint8)
    img_copy = cv2.GaussianBlur(img,(5,5),2)
    img_copy = cv2.morphologyEx(img_copy,cv2.MORPH_OPEN,kernel)    
    _,img_th = cv2.threshold(img_copy,127,255,cv2.THRESH_BINARY_INV)
    countours,_ = cv2.findContours(img_th, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    rects = [cv2.boundingRect(each) for each in countours]
    rects = sorted(rects)
    num_imgs = []
    for each in rects:
        cur_num_img = preprocessing(img_copy[each[1]:each[1]+each[3],each[0]:each[0]+each[2]])
        num_imgs.append(cur_num_img)
        cv2.rectangle(img, (each[0],each[1]), (each[0]+each[2],each[1]+each[3]),(0,255,0), 3)
    num = 0
    try:
        num = int(''.join(map(str,read_num(num_imgs))))
    except:
        print('error')
    print(num)
    return img, num
def preprocessing(img):
    width,height = list(img.shape)
    border = max(width,height)
    border = int(border*1.3)
    paper = np.zeros((border,border),np.uint8)
    paper.fill(255)
    center = border//2
    x0 = center - width//2
    y0 = center - height//2
    paper[x0:x0+width,y0:y0+height] = img
    paper = cv2.resize(paper, (28,28),interpolation=cv2.INTER_AREA)
    return paper
