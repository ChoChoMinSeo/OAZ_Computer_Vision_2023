import cv2
import numpy as np
import get_nums
import random

paper = np.zeros((256,256),np.uint8)
paper.fill(255)
cv2.imshow('paper',paper)
clk = False
write = True
def onMouse(event, x, y, flags, param):
    global clk, write
    if event == cv2.EVENT_LBUTTONDOWN:
        clk = True
        write = True
        cv2.circle(paper,(x,y),5,(0,0,0),-1)
        cv2.imshow('paper',paper)
    elif event == cv2.EVENT_MOUSEMOVE:
        if clk and write:
            cv2.circle(paper,(x,y),5,(0,0,0),-1)
            cv2.imshow('paper',paper)
        elif clk and not write:
            cv2.circle(paper,(x,y),10,(255,255,255),-1)
            cv2.imshow('paper',paper)
    elif event == cv2.EVENT_LBUTTONUP:
        clk = False
    elif event == cv2.EVENT_RBUTTONDOWN:
        clk = True
        write = False
        cv2.circle(paper,(x,y),10,(255,255,255),-1)
        cv2.imshow('paper',paper)
    elif event == cv2.EVENT_RBUTTONUP:
        clk = False
cv2.setMouseCallback('paper',onMouse)

button = np.zeros((128,256))
cv2.putText(button, 'Submit',org= (20,75),fontFace=0, fontScale=2, color=(255,255,255),thickness=3)
cv2.imshow('submit',button)
def submit_button(event, x, y, flags, param):
    global paper, cur_ans
    if event == cv2.EVENT_LBUTTONDOWN:
        #submit
        paper,answer = get_nums.get_nums(paper)
        if answer == cur_ans:
            text = 'correct!'
        else:
            text = 'wrong!'
        cv2.putText(paper, f'{str(answer)} {text}',(0,240),0,1.2,(0,0,0),3)
        cv2.imshow('paper',paper)
cv2.setMouseCallback('submit',submit_button)
        
button2 = np.zeros((128,256))
cv2.putText(button2, 'Reset',org= (20,75),fontFace=0, fontScale=2, color=(255,255,255),thickness=3)
cv2.imshow('reset',button2)
def reset_button(event, x, y, flags, param):
    global paper, cur_ans
    if event == cv2.EVENT_LBUTTONDOWN:
        #submit
        paper = np.zeros((256,256),np.uint8)
        paper.fill(255)
        cv2.imshow('paper',paper)
        cur_ans = new_q()
cv2.setMouseCallback('reset',reset_button)

def new_q():
    question_p = np.zeros((120,300),np.uint8)
    question_p.fill(255)
    a= random.randrange(1,100)
    b = random.randrange(1,100)
    cv2.putText(question_p,f'{a} + {b} = ?', (20,55),fontFace=0, fontScale=1.2, color=(0,0,0),thickness=3)
    cv2.imshow('question',question_p)
    return a+b
cur_ans = new_q()


cv2.waitKey()
cv2.destroyAllWindows()
