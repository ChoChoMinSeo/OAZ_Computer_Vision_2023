import torch
import numpy as np
from model import half_efficientNetb0

model = half_efficientNetb0().to('cpu')
model.load_state_dict(torch.load('C:/cv_task/model.pt',map_location='cpu'))
model.eval()
def read_num(imgs_list):
    imgs_list = torch.from_numpy((np.array(imgs_list)/255-1)*-1).unsqueeze(1).float()
    res = model(imgs_list)
    res = np.array(torch.argmax(res,axis=1))
    return list(res)