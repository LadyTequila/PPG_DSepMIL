# nohup python -W ignore main.py > logs/train_delay10_win60_soft0.log 2>&1 &
import hydra
import logging
import os

from runner import load_dataset, run
from models import MODELS
from utils import config_gpu

log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="config/", config_name="train")
def main(config):
    # 根据 exp_tag 创建隔离的子目录
    exp_tag = config.exp_tag
    log_dir    = os.path.join(config.log_dir,    exp_tag)
    weight_dir = os.path.join(config.weight_dir, exp_tag)
    os.makedirs(log_dir,    exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)

    # 将训练日志写入文件（若使用命令行重定向保存日志则注释掉此段，避免内容重叠）
    # log_file = os.path.join(config.log_dir, f"train_{exp_tag}.log")
    # fh = logging.FileHandler(log_file, encoding="utf-8")
    # fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    # logging.getLogger().addHandler(fh)

    log.info(f"exp_tag={exp_tag}  log_dir={log_dir}  weight_dir={weight_dir}")
    log.info("Initializing...")

    ### GPU Configuration ###
    device = config_gpu(config)

    # TODO: Import by files
    lr = 1e-3
    batch_size = 1024

    # Start Running
    log.info("Starting the training...")
    for FOLD in range(config.starting_fold, config.starting_fold + config.num_folds):
        all_x, all_y, test_x, test_y = load_dataset(
            config.dataset, fold=FOLD, data_dir=config.dataset_dir, features=config.features)
        wavenet_ch = all_x[0].shape[1]

        run(
            train_set=(all_x, all_y),
            test_set=(test_x, test_y),
            # refer to "from models import *"
            model_class=MODELS[config.model],
            model_name=config.model,                 # for naming purpose
            dataset_name=config.dataset,             # for naming purpose
            outer_fold=FOLD,
            log_dir=log_dir,                         # 已含 exp_tag 子目录
            weight_dir=weight_dir,                   # 已含 exp_tag 子目录
            device=device,
            subsampling=config.subsampling,

            lr=lr,
            batch_size=batch_size,
            wavenet_ch=wavenet_ch,
        )


if __name__ == "__main__":
    main()
