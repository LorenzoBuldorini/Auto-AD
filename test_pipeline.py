"""
Test da superare prima degli esperimenti (sezione 6 della guida).

    python test_pipeline.py                      # tutti i test, su CPU
    python test_pipeline.py --solo 1 2 3 4 5     # solo i test veloci
    python test_pipeline.py --device cuda        # test 6-7 su GPU (Colab)

I test 1-5 controllano le banche e l'integrazione in models/common.py.
I test 6-7 addestrano la rete sulla scena 1 con un numero ridotto di iterazioni
(--iter); l'output completo viene salvato anche in risultati/test_pipeline/.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat

import models.common as common
from models.skip import skip
from models.ridge_bank import FreeBankConv2d, RidgeBankConv2d
from abu_airport_autoad import auto_ad_valuta_scena

BANCHE = {'free_bank': FreeBankConv2d, 'ridge_bank': RidgeBankConv2d}
PARAMETRI_ATTESI = {  # scena 1 (205 bande), 64 canali, 5 livelli
    3: {'dense': 887885, 'free_bank': 889100, 'ridge_bank': 889865},
    7: {'dense': 4525645, 'free_bank': 894500, 'ridge_bank': 889865},
}
CARTELLE_DATI = ['../ABU_Dataset', 'data/ABU_Airport']   # locale, Colab
CARTELLA_USCITA = 'risultati/test_pipeline'


def carica_scena(nome='abu-airport-1.mat'):
    for cartella in CARTELLE_DATI:
        percorso = os.path.join(cartella, nome)
        if os.path.exists(percorso):
            dati = loadmat(percorso)
            return dati['data'].astype(np.float32), dati['map']
    raise FileNotFoundError(f'{nome} non trovato in {CARTELLE_DATI}')


def costruisci_rete(bande, livelli, kernel, operatore, canali=64):
    """Stessa rete di abu_airport_autoad.py, con l'operatore scelto."""
    common.imposta_operatore(operatore)
    return skip(bande, bande,
                num_channels_down=[canali] * livelli,
                num_channels_up=[canali] * livelli,
                num_channels_skip=[canali] * livelli,
                filter_size_up=kernel, filter_size_down=kernel,
                upsample_mode='nearest', filter_skip_size=1,
                need_sigmoid=True, need_bias=True, pad='reflection',
                act_fun='LeakyReLU')


def conta_parametri(modulo):
    return sum(p.numel() for p in modulo.parameters())


# --------------------------------------------------------------------------
# 1. Forme
# --------------------------------------------------------------------------
def test_forme():
    torch.manual_seed(0)
    x = torch.randn(2, 8, 21, 21)
    ok = True
    for k in (3, 5, 7):
        for stride in (1, 2):
            for padding in (0, k // 2):
                atteso = nn.Conv2d(8, 6, k, stride, padding)(x).shape
                for nome, cls in BANCHE.items():
                    forma = cls(8, 6, k, stride=stride, padding=padding)(x).shape
                    if forma != atteso:
                        print(f'  {nome} k={k} stride={stride} padding={padding}: '
                              f'{tuple(forma)} invece di {tuple(atteso)}')
                        ok = False
    print('  forme uguali a nn.Conv2d per k=3,5,7, stride 1,2, padding 0 e k//2'
          if ok else '  forme diverse (vedi sopra)')
    return ok


# --------------------------------------------------------------------------
# 2. Modo A == modo B
# --------------------------------------------------------------------------
def test_modo_a_b():
    torch.manual_seed(0)
    C, O = 8, 6
    x = torch.randn(2, C, 20, 20)
    peggiore = 0.0
    for k in (3, 5, 7):
        for stride in (1, 2):
            for nome, cls in BANCHE.items():
                modulo = cls(C, O, k, stride=stride, padding=0)
                with torch.no_grad():
                    R, A, bias = modulo.banca(), modulo.A, modulo.bias
                    S = R.shape[0]
                    p = k // 2
                    xp = F.pad(x, (p, p, p, p), mode='reflect')
                    w_dw = R.unsqueeze(0).expand(C, S, k, k).reshape(C * S, 1, k, k)
                    g = F.conv2d(xp, w_dw, stride=stride, groups=C)   # canale c*S + s
                    out_B = F.conv2d(g, A.reshape(O, C * S, 1, 1), bias)
                    out_A = modulo(xp)
                diff = (out_A - out_B).abs().max().item()
                peggiore = max(peggiore, diff)
                print(f'  {nome:10s} k={k} stride={stride}: differenza massima {diff:.2e}')
    return peggiore < 1e-4


# --------------------------------------------------------------------------
# 3. Copertura a k=3
# --------------------------------------------------------------------------
def test_copertura_k3():
    torch.manual_seed(0)
    O, C = 6, 8
    modulo = FreeBankConv2d(C, O, 3)
    W_target = torch.randn(O, C, 3, 3)
    with torch.no_grad():
        R = modulo.banca().reshape(9, 9)
        A = W_target.reshape(O * C, 9) @ torch.linalg.pinv(R)
        modulo.A.copy_(A.reshape(O, C, 9))
        errore = (modulo.peso() - W_target).abs().max().item()
    print(f'  banca libera: errore massimo {errore:.2e}')

    # Solo informativo: rango della banca RF-KAN a k=3 (9 = completa)
    with torch.no_grad():
        rango = torch.linalg.matrix_rank(RidgeBankConv2d(C, O, 3).banca().reshape(9, 9)).item()
    print(f'  (info) rango della banca RF-KAN iniziale a k=3: {rango} su 9')
    return errore < 1e-4


# --------------------------------------------------------------------------
# 4. Gradienti
# --------------------------------------------------------------------------
def test_gradienti():
    torch.manual_seed(0)
    ok = True
    attesi = {'free_bank': ['A', 'bias', 'R'],
              'ridge_bank': ['A', 'bias', 'gen.theta', 'gen.a_ridge', 'gen.log_scales']}
    for nome, cls in BANCHE.items():
        modulo = cls(8, 6, 5, padding=2)
        x, target = torch.randn(2, 8, 16, 16), torch.randn(2, 6, 16, 16)
        F.mse_loss(modulo(x), target).backward()
        parametri = dict(modulo.named_parameters())
        if sorted(parametri) != sorted(attesi[nome]):
            print(f'  {nome}: parametri {sorted(parametri)}, attesi {sorted(attesi[nome])}')
            ok = False
        for p_nome in attesi[nome]:
            grad = parametri[p_nome].grad
            norma = 0.0 if grad is None else grad.norm().item()
            print(f'  {nome:10s} {p_nome:15s} |grad| = {norma:.3e}')
            ok = ok and norma > 0
    return ok


# --------------------------------------------------------------------------
# 5. Conteggio dei parametri
# --------------------------------------------------------------------------
def test_conteggio_parametri():
    cubo, _ = carica_scena()
    bande = cubo.shape[2]
    ok = bande == 205
    if not ok:
        print(f'  la scena 1 ha {bande} bande invece di 205')
    for k, attesi in PARAMETRI_ATTESI.items():
        for operatore, atteso in attesi.items():
            rete = costruisci_rete(bande, 5, k, operatore)
            n = conta_parametri(rete)
            n_banche = sum(isinstance(m, tuple(BANCHE.values())) for m in rete.modules())
            n_conv = sum(isinstance(m, nn.Conv2d) for m in rete.modules())
            esito = 'ok' if n == atteso else 'DIVERSO'
            print(f'  k={k} {operatore:10s} {n:>9,} (atteso {atteso:>9,}) {esito}'
                  f'   banche={n_banche} nn.Conv2d={n_conv}')
            ok = ok and n == atteso
    common.imposta_operatore('dense')
    return ok


# --------------------------------------------------------------------------
# 6. Determinismo
# --------------------------------------------------------------------------
def test_determinismo(device, iterazioni, curve_ogni=50):
    cubo, gt = carica_scena()
    for operatore in common.OPERATORI:
        risultati = [auto_ad_valuta_scena(cubo, gt, num_iter=iterazioni, seed=1, device=device,
                                          operatore=operatore, curve_ogni=curve_ogni)
                     for _ in range(2)]
        (m1, r1, c1), (m2, r2, c2) = risultati
        diff = np.abs(r1 - r2).max()
        print(f'  {operatore:10s} AUC {m1["auc"]:.4f} vs {m2["auc"]:.4f}, iterazioni '
              f'{m1["iterazioni_effettive"]} vs {m2["iterazioni_effettive"]}, '
              f'differenza massima mappa residui {diff:.2e}')
        # Curve: differenza massima su tutti i punti registrati, per ciascuna grandezza
        if len(c1) != len(c2):
            print(f'  {"":10s} curve di lunghezza diversa: {len(c1)} vs {len(c2)} punti')
            continue
        diff_curve = {campo: max(abs(p1[campo] - p2[campo]) for p1, p2 in zip(c1, c2))
                      for campo in ('auc', 'errore_sfondo', 'errore_aerei')}
        print(f'  {"":10s} curve ({len(c1)} punti, ogni {curve_ogni} iter), differenza massima: '
              + ', '.join(f'{campo} {d:.2e}' for campo, d in diff_curve.items()))
    common.imposta_operatore('dense')
    # Sulla GPU piccole differenze sono normali: il test registra, non boccia
    return True


# --------------------------------------------------------------------------
# 7. Run di prova
# --------------------------------------------------------------------------
def test_run_prova(device, iterazioni):
    cubo, gt = carica_scena()
    os.makedirs(CARTELLA_USCITA, exist_ok=True)
    righe = []
    for operatore in common.OPERATORI:
        metriche, mappa, _ = auto_ad_valuta_scena(cubo, gt, num_iter=iterazioni, seed=1,
                                                  device=device, operatore=operatore)
        np.save(os.path.join(CARTELLA_USCITA, f'residui_{operatore}.npy'), mappa)
        righe.append({'operatore': operatore, **metriche})
        print(f'  {operatore:10s} {metriche}')
    common.imposta_operatore('dense')

    percorso = os.path.join(CARTELLA_USCITA, 'run_prova.csv')
    with open(percorso, 'w', newline='') as f:
        scrittore = csv.DictWriter(f, fieldnames=list(righe[0]))
        scrittore.writeheader()
        scrittore.writerows(righe)
    attesi = [percorso] + [os.path.join(CARTELLA_USCITA, f'residui_{o}.npy')
                           for o in common.OPERATORI]
    mancanti = [p for p in attesi if not os.path.exists(p)]
    print(f'  file in uscita: {CARTELLA_USCITA}/' + (f' (mancano {mancanti})' if mancanti else ''))
    return not mancanti


class Tee:
    """Scrive l'output sia a schermo sia su file."""

    def __init__(self, *flussi):
        self.flussi = flussi

    def write(self, testo):
        for f in self.flussi:
            f.write(testo)

    def flush(self):
        for f in self.flussi:
            f.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--solo', type=int, nargs='+', default=list(range(1, 8)))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--iter', type=int, default=101,
                        help='iterazioni per i test 6-7 (101 = un aggiornamento dei residui)')
    args = parser.parse_args()

    os.makedirs(CARTELLA_USCITA, exist_ok=True)
    log = open(os.path.join(CARTELLA_USCITA, 'output.txt'), 'w', encoding='utf-8')
    sys.stdout = Tee(sys.__stdout__, log)
    print(f'torch {torch.__version__}, device {args.device}, iterazioni test 6-7: {args.iter}')

    test = {
        1: ('Forme', test_forme),
        2: ('Modo A == modo B', test_modo_a_b),
        3: ('Copertura a k=3', test_copertura_k3),
        4: ('Gradienti', test_gradienti),
        5: ('Conteggio parametri', test_conteggio_parametri),
        6: ('Determinismo', lambda: test_determinismo(args.device, args.iter)),
        7: ('Run di prova', lambda: test_run_prova(args.device, args.iter)),
    }
    esiti = {}
    for numero in args.solo:
        nome, funzione = test[numero]
        print(f'\n[{numero}] {nome}')
        inizio = time.time()
        esiti[numero] = funzione()
        print(f'  -> {"SUPERATO" if esiti[numero] else "FALLITO"} ({time.time() - inizio:.1f} s)')

    print('\nRiepilogo')
    for numero, esito in esiti.items():
        print(f'  [{numero}] {test[numero][0]:22s} {"SUPERATO" if esito else "FALLITO"}')
    sys.stdout = sys.__stdout__
    log.close()
    sys.exit(0 if all(esiti.values()) else 1)


if __name__ == '__main__':
    main()
