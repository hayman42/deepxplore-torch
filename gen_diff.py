import torchvision.models as models
from deepxplore import deepXplore

import numpy as np
from utils import *
from PIL import Image


seeds = 20
img_dir = "./seeds/"
resnet50 = models.resnet50(pretrained=True)
resnet34 = models.resnet34(pretrained=True)
dxp = deepXplore([resnet50, resnet34], lambda_2=1, s=0.5, itr_num=1000)
rect_shape = (50, 50)
start_point = (
    random.randint(0, 224-50), random.randint(0, 224-50))
coverage_history = []


def constraint(x): return constraint_black(x)
def occl(x): return constraint_occl(x, start_point, rect_shape)


for i in range(30):
    try:
        x, orig_img = get_img(img_dir)
    except:
        continue
    gen_x = dxp.generate(x, occl)

    gen_img = Image.fromarray(to_image(gen_x[0]))
    gen_img.save(f"./gen_input/{orig_img}")
    # d = dict(dxp.output_tables[0])
    # for key in d.keys():
    #     d[key]=scale(d[key][0]).mean(dim=list(range(1,len(d[key][0].shape))))
    # for key in d.keys():
    #     print(len(d[key]))
    coverage_history.append(dxp.get_coverage())
    if all(p > 0.5 for p in dxp.get_coverage()):
        print("good")
        break

print(coverage_history)
