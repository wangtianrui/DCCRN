# coding: utf-8
# Author：WangTianRui
# Date ：2020/10/17 21:10
import warnings

warnings.filterwarnings("ignore")

from asteroid import torch_utils
import json, yaml
from dns_loader import DNSDataset, WavHopDataset
from torch.utils.data import DataLoader
from asteroid.engine.optimizers import make_optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
import torch, time, os


class MyBaseSystem():
    def __init__(self, conf_path):
        if not os.path.exists(conf_path):
            print("conf path error!")
        with open(conf_path, "r") as f:
            conf = yaml.safe_load(f)
        self.conf = conf
        self.save_file = conf["preset"]["save_file"]
        self.checkpoint_dir = os.path.join(conf["preset"]["save_file"], "checkpoints/")
        if self.conf["preset"]["dataset"] == 'D':
            if self.conf["preset"]["enframe"]:
                self.train_loader, self.val_loader = get_dns_data_frame_loader(
                    conf["train"]["batch_size"], conf["train"]["num_workers"], json_home=conf["preset"]["json_home"],
                    frame_dur=conf["preset"]["frame_dur"], hop_dur=conf["preset"]["hop_dur"],
                    data_home=conf["preset"]["data_home"]
                )
            else:
                self.train_loader, self.val_loader = get_dns_data_loader(
                    conf["train"]["batch_size"], conf["train"]["num_workers"], json_home=conf["preset"]["json_home"],
                    data_home=conf["preset"]["data_home"]
                )
        else:
            print("conf dataset is not D or V")
            return
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.checkpoint = None
        self.early_stop = None
        self.init_checkpoints()
        self.system = None
        self.trainer = None

    def init_checkpoints(self):
        os.makedirs(self.save_file, exist_ok=True)
        self.checkpoint = ModelCheckpoint(
            self.checkpoint_dir, monitor="val_loss", mode="min", save_top_k=5, verbose=True
        )

    def init_optims(self, model, criterion, optimizer=None):
        if optimizer is None:
            self.optimizer = make_optimizer(model.parameters(), **self.conf["optim"])
        else:
            self.optimizer = optimizer
        if self.conf["train"]["half_lr"]:
            self.scheduler = ReduceLROnPlateau(
                optimizer=self.optimizer, factor=self.conf["scheduler"]["factor"],
                patience=self.conf["scheduler"]["patience"], verbose=self.conf["scheduler"]["verbose"]
            )
        if self.conf["train"]["early_stop"]:
            self.early_stop = EarlyStopping(monitor="val_loss", patience=20, verbose=True)
        self.criterion = criterion

    def init_system_and_trainer(self, SystemClass, model, gpus=None):
        if gpus is None:
            gpus = [0, 1]
        self.system = SystemClass(
            model=model,
            loss_func=self.criterion,
            optimizer=self.optimizer,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            scheduler=self.scheduler,
            config=self.conf
        )
        self.trainer = pl.Trainer(
            max_epochs=self.conf["train"]["epochs"],
            checkpoint_callback=self.checkpoint,
            early_stop_callback=self.early_stop,
            default_root_dir=self.save_file,
            gpus=gpus,
            distributed_backend="dp",
            train_percent_check=1.0,
            gradient_clip_val=5.0
        )

    def fit(self):
        self.trainer.fit(self.system)
        best_k = {k: v.item() for k, v in self.checkpoint.best_k_models.items()}
        with open(os.path.join(self.save_file, "best_k_models.json"), "w") as f:
            json.dump(best_k, f, indent=0)


def get_dns_data_loader(batch_size, num_workers, json_home, data_home):
    train_json_file = os.path.join(json_home, "train_file_info.json")
    val_json_file = os.path.join(json_home, "test_file_info.json")
    train_set = DNSDataset(train_json_file, data_home=data_home)
    val_set = DNSDataset(val_json_file, data_home=data_home)
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True
    )
    return train_loader, val_loader


def get_dns_data_frame_loader(batch_size, num_workers, json_home, data_home, frame_dur, hop_dur):
    """
    :param batch_size:
    :param num_workers:
    :param json_home:
    :param frame_dur: 单位s
    :param hop_dur: 单位s
    :return:
    """
    train_json_file = os.path.join(json_home, "train_file_info.json")
    val_json_file = os.path.join(json_home, "test_file_info.json")
    train_set = WavHopDataset(train_json_file, frame_dur=frame_dur, hop_dur=hop_dur, data_home=data_home)
    val_set = WavHopDataset(val_json_file, frame_dur=frame_dur, hop_dur=hop_dur, data_home=data_home)
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True
    )
    return train_loader, val_loader


def load_best_param(log_file, model, gpu=False, test=True):
    """
    :param log_file: 保存参数的file
    :param model: 模型
    :param gpu: 是否使用GPU
    :return: 读取了最好参数的模型
    """
    if not os.path.exists(log_file):
        print("log path error")
        return
    with open(os.path.join(log_file, "best_k_models.json"), "r") as f:
        best_k = json.load(f)
    best_model_path = min(best_k, key=best_k.get)
    if not os.path.exists(best_model_path):
        ckpt_name = str(best_model_path).split('/')[-1]
        best_model_path = os.path.join(log_file, ckpt_name)
        if not os.path.exists(best_model_path):
            print("model param path error")
            return
    if gpu:
        ckpt = torch.load(best_model_path, map_location=torch.device("cuda:0"))
    else:
        ckpt = torch.load(best_model_path, map_location="cpu")
    model = torch_utils.load_state_dict_in(ckpt["state_dict"], model)
    if test:
        model.eval()
    print("load param ok")
    return model


def model_test_timer(model, input_size=(8, 16000 * 3), gpu=False):
    test_inp = torch.randn(input_size)
    if gpu:
        model = model.cuda()
        test_inp = test_inp.cuda()
    start = time.time()
    print("运行结果:", model(test_inp).size())
    print("消耗时间:%f" % (time.time() - start))


estimate_home = r"F:\Traindata\DNS-Challenge\make_data\test/"


def load_conf(path):
    with open(path, "r") as f:
        conf = yaml.safe_load(f)
        return conf
