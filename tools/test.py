import sys, os, shutil
import argparse, yaml
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

from ueps.datasets import build_data
from ueps.models import build_model
from ueps.utils import train_acc_setting
from ueps.utils import eval_plot_gt_pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/fastmri_UEPS.yaml")
    parser.add_argument("--ckpt_path", default="../checkpoints/ckpt_pick.pth")
    parser.add_argument("--data_dir", default="../demo_data", help="Path to the directory containing all example datasets")
    parser.add_argument("--output_dir", default="../model_export/name1_eval")

    args = parser.parse_args()
    ckpt_path = args.ckpt_path
    output_dir = args.output_dir

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.data_dir is not None:
        config["root"] = args.data_dir
    
    config["ckpt_path"] = ckpt_path
    config["output_dir"] = output_dir

    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # set device, model, loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Use device: {device}")

    model = build_model(config, device, None, ckpt_path, True)
    _, _, model, _ = train_acc_setting(
        model, False, config["use_compile"], config["precision_type"])

    # eval on each testset
    eval_use_amp = config.get("eval_use_amp", False)
    Naddtestset = config.get("Naddtestset", 1)

    for i in range(Naddtestset):
        config_i = config.get(f"addtestset_config_{i}")
        if args.data_dir is not None:
            config_i["root"] = args.data_dir
            
        name_i = f"addtestset{i}_"+config_i['which_dataset']+"-"+config_i['which_data']
        print(f"\n{name_i}")
        _, _, test_loader = build_data(config_i, is_train=False)

        eval_plot_gt_pred(test_loader, model, device,
                          os.path.join(output_dir, name_i),
                          config["escale"], eval_use_amp)

    print("All Done")
    return None

if __name__ == '__main__':
    main()
