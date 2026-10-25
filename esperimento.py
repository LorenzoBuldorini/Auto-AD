"""
Un singolo run di Auto-AD su una scena ABU, con l'operatore scelto.

    python esperimento.py --scena abu-airport-1 --seed 1 --operatore ridge_bank \
        --kernel 7 --livelli 5                           # run principale
    python esperimento.py --scena abu-airport-1 --seed 1 --operatore ridge_bank \
        --kernel 7 --livelli 5 --soglia 0                # diagnostica (1000 iterazioni)

Per ogni run salva in <uscita>/<id_run>/:
    curve.csv    AUC, errore medio sfondo ed errore medio aerei ogni --curve_ogni iterazioni
    residui.npy  mappa dei residui finale (quella usata per l'AUC)
e aggiunge una riga a <uscita>/riepilogo.csv. La riga si scrive per ultima: se c'e',
il run e' completo (il lanciatore usa questo per saltare i run gia' fatti).
"""
import argparse
import csv
import glob
import os
import time

import numpy as np
import torch
from scipy.io import loadmat

import models.common as common
from abu_airport_autoad import auto_ad_valuta_scena

CARTELLE_DATI = ['../ABU_Dataset', 'data']   # locale, Colab (cerca anche nelle sottocartelle)
USCITA = 'risultati/esperimenti'
CAMPI_RIEPILOGO = ['id_run', 'scena', 'righe', 'colonne', 'bande', 'operatore', 'kernel',
                   'livelli', 'w0', 'soglia', 'seed', 'iter_max', 'iterazioni_effettive',
                   'tempo_secondi', 'n_parametri', 'auc', 'device', 'data']


def carica_scena(scena):
    for cartella in CARTELLE_DATI:
        trovati = glob.glob(os.path.join(cartella, '**', scena + '.mat'), recursive=True)
        if trovati:
            dati = loadmat(trovati[0])
            return dati['data'].astype(np.float32), dati['map']
    raise FileNotFoundError(f'{scena}.mat non trovato in {CARTELLE_DATI}')


def id_run(scena, operatore, kernel, livelli, w0, soglia, seed, iter_max):
    return (f'{scena}_L{livelli}_k{kernel}_{operatore}_w0_{w0:g}_soglia_{soglia:g}'
            f'_it{iter_max}_s{seed}')


def run_completati(uscita=USCITA):
    """Insieme degli id_run gia' presenti nel riepilogo."""
    percorso = os.path.join(uscita, 'riepilogo.csv')
    if not os.path.exists(percorso):
        return set()
    with open(percorso, newline='') as f:
        return {riga['id_run'] for riga in csv.DictReader(f)}


def esegui_run(scena, seed, operatore, kernel, livelli, w0=3.0, soglia=1.5e-5,
               iter_max=1001, curve_ogni=50, device='cuda', uscita=USCITA):
    nome = id_run(scena, operatore, kernel, livelli, w0, soglia, seed, iter_max)
    cubo, gt = carica_scena(scena)
    righe, colonne, bande = cubo.shape

    metriche, mappa, curve = auto_ad_valuta_scena(
        cubo, gt, layers=livelli, num_iter=iter_max, thres=soglia, seed=seed,
        device=device, operatore=operatore, kernel=kernel, w0=w0, curve_ogni=curve_ogni)
    common.imposta_operatore('dense')

    cartella = os.path.join(uscita, nome)
    os.makedirs(cartella, exist_ok=True)
    np.save(os.path.join(cartella, 'residui.npy'), mappa)
    with open(os.path.join(cartella, 'curve.csv'), 'w', newline='') as f:
        scrittore = csv.DictWriter(f, fieldnames=['iterazione', 'auc', 'errore_sfondo',
                                                  'errore_aerei'])
        scrittore.writeheader()
        scrittore.writerows(curve)

    riga = {'id_run': nome, 'scena': scena, 'righe': righe, 'colonne': colonne,
            'bande': bande, 'operatore': operatore, 'kernel': kernel, 'livelli': livelli,
            'w0': w0, 'soglia': soglia, 'seed': seed, 'iter_max': iter_max,
            **metriche, 'device': device, 'data': time.strftime('%Y-%m-%d %H:%M:%S')}
    percorso = os.path.join(uscita, 'riepilogo.csv')
    nuovo = not os.path.exists(percorso)
    with open(percorso, 'a', newline='') as f:
        scrittore = csv.DictWriter(f, fieldnames=CAMPI_RIEPILOGO)
        if nuovo:
            scrittore.writeheader()
        scrittore.writerow(riga)
    return riga


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scena', required=True, help='es. abu-airport-1 (senza .mat)')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--operatore', required=True, choices=common.OPERATORI)
    parser.add_argument('--kernel', type=int, required=True)
    parser.add_argument('--livelli', type=int, required=True)
    parser.add_argument('--w0', type=float, default=3.0)
    parser.add_argument('--soglia', type=float, default=1.5e-5,
                        help='early stopping; 0 = diagnostica senza early stopping')
    parser.add_argument('--iter_max', type=int, default=1001)
    parser.add_argument('--curve_ogni', type=int, default=50)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--uscita', default=USCITA)
    args = parser.parse_args()

    riga = esegui_run(**vars(args))
    print(riga)


if __name__ == '__main__':
    main()
