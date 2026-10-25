"""
Lancia la griglia di esperimenti, saltando i run gia' presenti nel riepilogo.

    python lanciatore.py --modo principale                 # tutte le 13 scene
    python lanciatore.py --modo diagnostica                # solo le 4 scene Airport
    python lanciatore.py --modo principale --seed 1 2 --prova   # elenca soltanto

Modi:
    principale   early stopping originale (soglia 1.5e-5, max 1001 iterazioni)
    diagnostica  soglia 0, 1000 iterazioni, curve ogni 50 (solo scene Airport)
Griglia: configurazioni (livelli, kernel) x 3 bracci x seed x scene. I bracci della
stessa scena/configurazione/seed girano uno dopo l'altro, cosi' anche una griglia
interrotta lascia confronti completi.
I run che falliscono vengono annotati in <uscita>/errori.txt e ritentati al lancio dopo.
"""
import argparse
import os
import time
import traceback

import torch

import models.common as common
from esperimento import USCITA, esegui_run, id_run, run_completati

CONFIGURAZIONI = [(5, 3), (5, 7), (3, 3), (3, 7), (3, 11), (3, 15), (3, 21)]  # (livelli, kernel)
SEED = [1, 2, 3, 4, 5]
SCENE = {
    'airport': [f'abu-airport-{i}' for i in range(1, 5)],
    'urban': [f'abu-urban-{i}' for i in range(1, 6)],
    'beach': [f'abu-beach-{i}' for i in range(1, 5)],
}
SOGLIA = {'principale': 1.5e-5, 'diagnostica': 0.0}


def espandi_scene(nomi):
    """Accetta gruppi (airport, urban, beach) o nomi di scena (abu-urban-2)."""
    scene = []
    for nome in nomi:
        for scena in SCENE.get(nome, [nome]):
            if scena not in scene:
                scene.append(scena)
    return scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modo', required=True, choices=list(SOGLIA))
    parser.add_argument('--scene', nargs='+', default=list(SCENE),
                        help='gruppi (airport urban beach) o nomi (abu-urban-2)')
    parser.add_argument('--seed', type=int, nargs='+', default=SEED)
    parser.add_argument('--operatori', nargs='+', default=common.OPERATORI,
                        choices=common.OPERATORI)
    parser.add_argument('--livelli', type=int, nargs='+', help='filtra le configurazioni')
    parser.add_argument('--kernel', type=int, nargs='+', help='filtra le configurazioni')
    parser.add_argument('--w0', type=float, default=3.0)
    parser.add_argument('--iter_max', type=int, default=1001)
    parser.add_argument('--curve_ogni', type=int, default=50)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--uscita', default=USCITA)
    parser.add_argument('--prova', action='store_true', help='elenca i run senza eseguirli')
    args = parser.parse_args()

    scene = espandi_scene(args.scene)
    if args.modo == 'diagnostica':
        scartate = [s for s in scene if not s.startswith('abu-airport')]
        if scartate:
            print(f'Diagnostica solo sulle scene Airport: ignoro {scartate}')
        scene = [s for s in scene if s.startswith('abu-airport')]
    configurazioni = [(l, k) for l, k in CONFIGURAZIONI
                      if (not args.livelli or l in args.livelli)
                      and (not args.kernel or k in args.kernel)]
    soglia = SOGLIA[args.modo]

    griglia = [dict(scena=scena, seed=seed, operatore=operatore, kernel=k, livelli=l)
               for seed in args.seed for scena in scene for l, k in configurazioni
               for operatore in args.operatori]
    fatti = run_completati(args.uscita)
    da_fare = [r for r in griglia
               if id_run(w0=args.w0, soglia=soglia, iter_max=args.iter_max, **r) not in fatti]
    print(f'Modo {args.modo}: {len(griglia)} run nella griglia, {len(griglia) - len(da_fare)} '
          f'gia\' completati, {len(da_fare)} da fare (device {args.device}).')
    if args.prova:
        for r in da_fare:
            print('  ' + id_run(w0=args.w0, soglia=soglia, iter_max=args.iter_max, **r))
        return

    os.makedirs(args.uscita, exist_ok=True)
    inizio, falliti = time.time(), 0
    for n, r in enumerate(da_fare, 1):
        nome = id_run(w0=args.w0, soglia=soglia, iter_max=args.iter_max, **r)
        try:
            riga = esegui_run(w0=args.w0, soglia=soglia, iter_max=args.iter_max,
                              curve_ogni=args.curve_ogni, device=args.device,
                              uscita=args.uscita, **r)
            esito = (f'AUC {riga["auc"]:.4f}, {riga["iterazioni_effettive"]} iter, '
                     f'{riga["tempo_secondi"]} s')
        except Exception as errore:
            falliti += 1
            common.imposta_operatore('dense')
            esito = f'ERRORE: {errore}'
            with open(os.path.join(args.uscita, 'errori.txt'), 'a', encoding='utf-8') as f:
                f.write(f'--- {time.strftime("%Y-%m-%d %H:%M:%S")} {nome}\n')
                f.write(traceback.format_exc())
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        trascorso = time.time() - inizio
        restante = trascorso / n * (len(da_fare) - n)
        print(f'[{n}/{len(da_fare)}] {nome}: {esito} '
              f'(trascorsi {trascorso / 60:.1f} min, stima restante {restante / 60:.1f} min)',
              flush=True)

    print(f'Finito: {len(da_fare) - falliti} run completati, {falliti} falliti.')


if __name__ == '__main__':
    main()
