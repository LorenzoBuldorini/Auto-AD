"""
Adattamento di Auto-AD al dataset ABU-Airport.
Mantiene identica la logica di training Deep Image Prior del repository
originale; cambia solo il caricamento dati e l'aggiunta del calcolo AUC.
"""
import time
import numpy as np
import torch
import torch.optim
from sklearn.metrics import roc_auc_score

from models.skip import skip
from utils.inpainting_utils import get_noise, get_params

# Disattivando benchmark (False) e attivando deterministic = True, cuDNN usa sempre lo stesso algoritmo, garantendo output identici a parità di seed
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def auto_ad_valuta_scena(cubo, gt, canali=64, layers=5, num_iter=1001,
                          thres=0.000015, lr=0.01, seed=42, verbose_ogni=0,
                          device='cuda'):
    """
    cubo: array (righe, colonne, bande)
    gt:   array (righe, colonne), binario
    Ritorna: metriche (dict), mappa_residuo (righe, colonne)
    """
    dtype = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor
    torch.manual_seed(seed)
    np.random.seed(seed)

    rows, cols, bands = cubo.shape
    img_np = cubo.transpose(2, 0, 1).astype(np.float32)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    img_var = torch.from_numpy(img_np).type(dtype)[None, :]

    net = skip(bands, bands,
               num_channels_down=[canali] * layers,
               num_channels_up=[canali] * layers,
               num_channels_skip=[canali] * layers,
               filter_size_up=3, filter_size_down=3,
               upsample_mode='nearest', filter_skip_size=1,
               need_sigmoid=True, need_bias=True, pad='reflection',
               act_fun='LeakyReLU').type(dtype)

    n_parametri = sum(np.prod(list(p.size())) for p in net.parameters())

    net_input = get_noise(bands, '2D', img_np.shape[1:]).type(dtype)
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    mse = torch.nn.MSELoss().type(dtype)
    mask_var = torch.ones(1, bands, rows, cols).to(device)
    residual_varr = torch.ones(rows, cols).to(device)

    def closure(iter_num, mask_varr, residual_varr):
        net_in = net_input_saved + (noise.normal_() * 0.1)
        out = net(net_in)
        mask_c, res_c = mask_varr.detach().clone(), residual_varr.detach().clone()

        if iter_num % 100 == 0 and iter_num != 0:
            temp = (out.detach()[0, :] - img_var[0, :]) ** 2
            residual_img = temp.sum(0)
            res_c = residual_img
            w = residual_img.max() - residual_img
            w = (w - w.min()) / (w.max() - w.min() + 1e-8)
            for i in range(mask_c.size(1)):
                mask_c[0, i, :] = w[:]

        loss = mse(out * mask_c, img_var * mask_c)
        loss.backward()
        return mask_c, res_c, loss

    optimizer = torch.optim.Adam(get_params('net', net, net_input), lr=lr)
    loss_np, loss_last, end_iter = np.zeros((1, 50), dtype=np.float32), 0, False
    inizio = time.time()

    for j in range(num_iter):
        optimizer.zero_grad()
        mask_var, residual_varr, loss = closure(j, mask_var, residual_varr)
        optimizer.step()
        if j >= 1:
            idx = j - int(j / 50) * 50
            loss_np[0][idx - 1] = abs(loss.item() - loss_last)
            if j % 50 == 0 and np.mean(loss_np) < thres:
                end_iter = True
        loss_last = loss.item()
        if verbose_ogni and j % verbose_ogni == 0:
            print(f"  iter {j}: loss={loss.item():.6f}")
        if j == num_iter - 1 or end_iter:
            break

    mappa_residuo = residual_varr.detach().cpu().numpy()
    auc = roc_auc_score(gt.flatten(), mappa_residuo.flatten())

    return {
        'iterazioni_effettive': j + 1,
        'tempo_secondi': round(time.time() - inizio, 1),
        'n_parametri': int(n_parametri),
        'auc': round(float(auc), 4),
    }, mappa_residuo
