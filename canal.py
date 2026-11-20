# canal.py
# El "canal" del disco: aquí se introducen los daños de forma controlada.

import numpy as np


def rafaga(datos, inicio, largo, semilla=None):
    # Rayón: 'largo' bytes SEGUIDOS quedan dañados (se les hace XOR con un valor al azar
    # distinto de cero, así cada byte del rayón queda realmente cambiado).
    rng = np.random.default_rng(semilla)
    d = np.array(datos, dtype=np.uint8)          # copia
    fin = min(inicio + largo, len(d))
    if inicio < fin:
        ruido = rng.integers(1, 256, size=fin - inicio, dtype=np.uint8)
        d[inicio:fin] ^= ruido
    return d


def errores_bytes(datos, p, semilla=None):
    # Cada byte se daña con probabilidad p, de forma independiente (errores aislados).
    rng = np.random.default_rng(semilla)
    d = np.array(datos, dtype=np.uint8)
    cuales = rng.random(len(d)) < p
    ruido = rng.integers(1, 256, size=len(d), dtype=np.uint8)
    d[cuales] ^= ruido[cuales]
    return d


def errores_bits(datos, p, semilla=None):
    # Canal binario simétrico: cada BIT se invierte con probabilidad p.
    rng = np.random.default_rng(semilla)
    d = np.array(datos, dtype=np.uint8)
    for b in range(8):
        voltear = (rng.random(len(d)) < p).astype(np.uint8)
        d ^= (voltear << b).astype(np.uint8)
    return d


def bytes_distintos(a, b):
    a = np.asarray(a, dtype=np.uint8)
    b = np.asarray(b, dtype=np.uint8)
    return int(np.count_nonzero(a != b))
