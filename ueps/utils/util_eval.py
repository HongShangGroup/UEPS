import os, shutil
import torch

from .util_plot import plot_gt_pred_c1

f16_type = torch.bfloat16

def eval_plot_gt_pred(dataloader, model, device, output_dir, escale, use_amp=False):
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    model.eval()
    batch_idx = 0
    with torch.no_grad():
        for x, y, mask, max_value, shape_raw in dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type='cuda', dtype=f16_type):
                    pred = model(x, mask)
            else:
                pred = model(x, mask)

            plot_gt_pred_c1(y, pred, shape_raw, max_value, output_dir, batch_idx, escale)

            batch_idx += 1
    return None
