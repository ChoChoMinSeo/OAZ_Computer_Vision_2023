from pathlib import Path
import torch
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_imshow, cv2,non_max_suppression, scale_boxes, xyxy2xywh)
from color.detect_color import get_color
import numpy as np

blank = np.zeros((300,300,3))
# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = './weights/best.pt'

model = DetectMultiBackend(weights, device=device)
stride, names, pt = model.stride, model.names, model.pt
# Dataloader
bs = 1  # batch_size

# enable webcam
webcam = True

source='../datas/test/4.jpg'  # file/dir/URL/glob/screen/0(webcam)
imgsz = (224,224)
if webcam:
    source = '0'
    view_img = check_imshow(warn=True)
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
    bs = len(dataset)
else:
    source='../datas/test/4.jpg' 
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)

model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

save_img = False
for path, im, im0s, vid_cap, s in dataset:
    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
    with dt[1]:
        pred = model(im)
    with dt[2]:
        pred = non_max_suppression(pred, 0.15, 0.45, None, True, max_det=2)
    rects=[]
    for i, det in enumerate(pred):  # per image
        if webcam:  # batch_size >= 1
            p, im0, frame = path[i], im0s[i].copy(), dataset.count
            s += f'{i}: '
        else:
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
        h,w,_ = im0.shape
        clean_im0 = im0.copy()
        p = Path(p)
        save_path = f'./res/{p.name}' # im.jpg
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = names[c]
                x1,y1,x2,y2 = torch.tensor(xyxy)
                confidence = float(conf)
                
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                x1 = int(xywh[0]*w)
                y1 = int(xywh[1]*h)
                x2 = int((xywh[0]+xywh[2])*w)
                y2 = int((xywh[1]+xywh[3])*h)
                rects.append([x1,y1,x2,y2,c])
                c = int(cls)  # integer class
                label = f'{names[c]} {conf:.2f}'
                cv2.rectangle(im0,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.putText(im0,label,fontFace=cv2.FONT_HERSHEY_SIMPLEX,org = (x1,y1),fontScale = 0.5,color = (255,0,0),thickness=2)
                    
    # Stream results
    if view_img:
        for idx,rect in enumerate(rects):
            try:
                color_w = get_color(clean_im0,rect[:4])
                cv2.imshow('color'+str(idx), color_w)
            except:
                cv2.imshow('color'+str(idx),blank)
        im0 = cv2.resize(im0,(0,0),fx=0.7,fy=0.7,interpolation = cv2.INTER_NEAREST)
        cv2.imshow(str(p), im0)
        if cv2.waitKey(1)==27: # 1 millisecond
            cv2.destroyAllWindows()
            break
    if save_img:
        if dataset.mode == 'image':
            cv2.imwrite(save_path, im0)
