# coding: utf-8
# Author：WangTianRui
# Date ：2020/11/4 18:24
from BaseModel import *
import model as model
from asteroid.engine import System


class MySystem(System):
    def common_step(self, batch, batch_nb, train=True):
        noisy, clean = batch
        est = self(noisy)
        lo = self.loss_func(est, clean)
        return lo


if __name__ == '__main__':
    model_ = model.DCCRN()
    my_system = MyBaseSystem("./conf.yml")
    my_system.init_optims(model=model_, criterion=model.loss)
    my_system.init_system_and_trainer(SystemClass=MySystem, model=model_, gpus=[0])
    my_system.fit()
