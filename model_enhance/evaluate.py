# python evaluate.py 2>&1 | tee logs/evaluate_delay0_win60_soft0_result.log
import hydra
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr

from runner import load_dataset, evaluate_onset, evaluate_severity, forward, bland_altman_plot, get_ahis
from models import MODELS
from utils import config_gpu

log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="config/", config_name="evaluate")
def main(config):
    # 根据 exp_tag 定位对应的子目录
    exp_tag = config.exp_tag
    log_dir    = os.path.join(config.log_dir,    exp_tag)
    weight_dir = os.path.join(config.weight_dir, exp_tag)

    # 将评估日志写入文件（若使用命令行重定向保存日志则注释掉此段，避免内容重叠）
    # log_file = os.path.join(config.log_dir, f"evaluate_result_{exp_tag}.log")
    # fh = logging.FileHandler(log_file, encoding="utf-8")
    # fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    # logging.getLogger().addHandler(fh)

    log.info(f"exp_tag={exp_tag}  log_dir={log_dir}  weight_dir={weight_dir}")
    log.info("Initializing...")

    ### GPU Configuration ###
    device = config_gpu(config)

    # Start Running
    log.info("Starting the evaluation...")
    results = []

    all_gt_ahis = []
    all_pred_ahis = []

    for FOLD in range(int(config.num_folds)):
        train_x, train_y, test_x, test_y = load_dataset(
            config.dataset, fold=FOLD, data_dir=config.dataset_dir, features=config.features)

        if config.save_pred:
            result = np.load(
                f"{config.save_pred}/{FOLD}", allow_pickle=True)
            infer_time = result["infer_time"]
            test_pred = result["pred"]
        else:
            test_pred, infer_time, test_pred_prob = forward(
                train_set=(train_x, train_y),
                test_set=(test_x, test_y),
                model_class=MODELS[config.model],
                model_name=config.model,
                dataset_name=config.dataset,
                outer_fold=FOLD,
                log_dir=log_dir,
                weight_dir=weight_dir,
                device=device,
                wavenet_ch=train_x[0].shape[1]
            )

        if config.mode == "onset":
            result = evaluate_onset(
                test_y, test_pred, test_pred_prob, config, FOLD)

        elif config.mode == "severity":
            result = evaluate_severity(
                test_y, test_pred, config, FOLD)

        result["infer_time"] = infer_time / sum([len(a) for a in test_y])
        results.append(result)

        log.info(f"Fold {FOLD} | Acc: {result['acc']*100:.2f}% | F1: {result['f1']*100:.2f}% | "
                 f"Sensitivity: {result['sens']*100:.2f}% | Specificity: {result['spec']*100:.2f}% | "
                 f"AUROC: {result['auroc']*100:.2f}%")

        gt_ahis, pred_ahis = get_ahis(test_y, test_pred, config, FOLD)
        all_gt_ahis.append(gt_ahis)
        all_pred_ahis.append(pred_ahis)

    # Concatenate AHIs
    all_gt_ahis = np.concatenate(all_gt_ahis, axis=None)
    all_pred_ahis = np.concatenate(all_pred_ahis, axis=None)

    # Scatter 图保存到本地
    fig = plt.figure(figsize=(6, 6), dpi=180)
    plt.scatter(all_gt_ahis, all_pred_ahis, s=3, alpha=0.4)
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    fig.savefig(os.path.join(log_dir, "scatter_all.png"))
    plt.close(fig)

    # Bland-Altman 图保存到本地
    fig = bland_altman_plot(all_gt_ahis, all_pred_ahis, config)
    fig.savefig(os.path.join(log_dir, "bland_altman_all.png"))
    plt.close(fig)

    pearson = pearsonr(all_gt_ahis, all_pred_ahis)

    results = pd.DataFrame.from_records(results)
    mean = results.mean(axis=0).to_dict()
    std  = results.std(axis=0, ddof=0).to_dict()

    log.info("=" * 50)
    log.info("5-Fold Summary:")
    log.info(f"  Accuracy:    {mean['acc']*100:.2f}% ± {std['acc']*100:.2f}%")
    log.info(f"  F1:          {mean['f1']*100:.2f}% ± {std['f1']*100:.2f}%")
    log.info(f"  Sensitivity: {mean['sens']*100:.2f}% ± {std['sens']*100:.2f}%")
    log.info(f"  Specificity: {mean['spec']*100:.2f}% ± {std['spec']*100:.2f}%")
    log.info(f"  AUROC:       {mean['auroc']*100:.2f}% ± {std['auroc']*100:.2f}%")
    log.info(f"  Pearson r:   {pearson[0]:.4f} (p={pearson[1]:.4f})")
    log.info("=" * 50)


if __name__ == "__main__":
    main()
