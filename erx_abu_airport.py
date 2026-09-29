"""
Baseline ERX (Yong Ma et al., Exponential RX — libreria HyperAD:
https://github.com/WiseGamgee/HyperAD) sul dataset ABU-Airport.

Parametri chiave:
- buffer_ratio: frazione delle righe totali usata per la stima iniziale
  di media e covarianza del background, prima di iniziare a produrre
  punteggi di anomalia affidabili (comportamento streaming/line-scan).
  buffer_ratio=0.2 -> le prime 20% delle righe sono "warm-up"
  buffer_ratio=0.0 -> nessun warm-up, punteggi fin dalla prima riga
    (a costo di stime di background meno stabili all'inizio)
"""
import sys
sys.path.insert(0, '/content/HyperAD')

import numpy as np
from detectors.erx import ERX
from sklearn.metrics import roc_auc_score


def valuta_scena(cubo, gt, n_projdims=10, momentum=0.1, buffer_ratio=0.2,
                  seed=42, buffer_minimo=10):
    rows, cols, bands = cubo.shape
    n_pixels, n_lines = cols, rows
    buffer_len = max(buffer_minimo, int(n_lines * buffer_ratio))

    np.random.seed(seed)
    model = ERX(n_bands=bands, n_pixels=n_pixels, buffer_len=buffer_len,
                n_projdims=n_projdims, momentum=momentum)

    scores = []
    for i in range(n_lines):
        y = model.forward(cubo[i, :, :])
        scores.append(y if y is not None else np.zeros(n_pixels))
    scores = np.array(scores)

    auc_completo = roc_auc_score(gt.flatten(), scores.flatten())
    gt_v = gt[buffer_len:, :]
    sc_v = scores[buffer_len:, :]
    auc_valido = roc_auc_score(gt_v.flatten(), sc_v.flatten())

    return {
        'buffer_len': buffer_len,
        'auc_con_warmup': round(float(auc_completo), 4),
        'auc_senza_warmup': round(float(auc_valido), 4),
    }, scores
