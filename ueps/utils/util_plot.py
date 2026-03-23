import os
import numpy as np
import matplotlib.pyplot as plt

from .util_metric import nmse, psnr, ssim
from .util_dataprep import img3_rm_black_border, center_crop

def plot_gt_pred_c1(gt, pred, shape_raw, max_value, output_dir, batch_idx, escale=1., cmap='gray', ecmap='jet'):
    """
    Plot ground truth image with predicted image, both with 1 channel.
    Image black border is cut if exists in GT. Metrics are saved in filename.

    Parameters:
        gt: tensor [Nbs, 1, H, W]
        pred: tensor [Nbs, 1, H, W]
        shape_raw: tensor [Nbs, 2], reference shape for center crop
        max_value: tensor [Nbs]
        output_dir: str, path to save figures
        escale: float, error display range scaling factor
        cmap: str, color map for gt and pred image
        ecmap: str, color map for error image
    """
    Nbs, Nc, _, _ = gt.shape
    assert Nc == 1
    assert gt.shape == pred.shape

    gt = gt.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    max_value = max_value.cpu().detach().numpy()
    shape_raw = shape_raw.cpu().detach().numpy()
    erro = np.abs(gt - pred)

    plt.rcParams.update({'font.size': 14})
    title_list = ["True", "Predict", "Error"]

    with plt.ioff():
        for s in range(Nbs):
            fig, axs = plt.subplot_mosaic([['a'], ['b'], ['c']], layout='constrained',
                                          figsize=(6, 15), dpi=300)

            gts = center_crop(gt[s,...], shape_raw[s,:])
            preds = center_crop(pred[s,...], shape_raw[s,:])
            erros = center_crop(erro[s,...], shape_raw[s,:])
            gts, preds, erros = img3_rm_black_border(gts, preds, erros)
            maxval = max_value[s]
            vr = [0.0, np.max(gts)]

            for i, (label, ax) in enumerate(axs.items()):
                if i == 0:
                    a = ax.imshow(gts[0,:,:], vmin=vr[0], vmax=vr[1], cmap=cmap)
                elif i == 1:
                    a = ax.imshow(preds[0,:,:], vmin=vr[0], vmax=vr[1], cmap=cmap)
                elif i == 2:
                    a = ax.imshow(erros[0,:,:], vmin=vr[0], vmax=vr[1]/escale, cmap=ecmap)
                ax.set_axis_off()
                ax.set_title(title_list[i])
                plt.colorbar(a, location='right')

            nmse_val = nmse(gts, preds)
            psnr_val = psnr(gts, preds, maxval)
            ssim_val = ssim(gts, preds, maxval)
            metric_str = f"nmse_{nmse_val:.5f}_psnr_{psnr_val:.4f}_ssim_{ssim_val:.4f}"
            idx = batch_idx * Nbs + s
            savename = os.path.join(output_dir, f"{idx}_{metric_str}.png")
            plt.savefig(savename, bbox_inches='tight')
            plt.close()

    return None
