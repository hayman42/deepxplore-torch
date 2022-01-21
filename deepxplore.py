import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm

from utils import *


class deepXplore:
    def __init__(self, dnns, itr_num=500, lambda_1=2, lambda_2=1, threshold=0.75, s=0.1):
        self.dnns = dnns
        self.itr_num = itr_num
        self.output_tables = list(output_table(dnn) for dnn in dnns)
        self.coverage_tables = list({} for dnn in dnns)
        self.t = threshold
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.s = s

    def generate(self, x, constraint):
        gen_x = x.detach().clone()
        gen_x.requires_grad = True
        out = [dnn(gen_x).squeeze() for dnn in self.dnns]
        labels = [o.argmax() for o in out]
        for ct, ot in zip(self.coverage_tables, self.output_tables):
            if not ct:
                init_coverage(ct, ot)
        for itr in tqdm(range(self.itr_num)):
            if all(label == labels[0] for label in labels):
                obj1 = compute_obj1(labels[0], out, self.lambda_1)
                obj2 = compute_obj2(self.coverage_tables, self.output_tables)
                loss = obj1 + self.lambda_2 * obj2
                loss.backward()
                grads = constraint(gen_x.grad)
                gen_x.detach_()
                gen_x += self.s * grads
                gen_x.requires_grad = True
                out = [dnn(gen_x).squeeze() for dnn in self.dnns]
                labels = [o.argmax() for o in out]
            else:
                break
        for ct, ot in zip(self.coverage_tables, self.output_tables):
            update_coverage(ct, ot, self.t)
        return gen_x

    def get_coverage(self):
        return [neuron_coverage(ct) for ct in self.coverage_tables]
