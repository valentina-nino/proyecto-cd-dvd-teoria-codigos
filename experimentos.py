# experimentos.py
# Genera los resultados del proyecto (tablas CSV y gráficas en salida/):
#   A) probabilidad de recuperar el disco exacto vs. largo del rayón, para varios D
#      (comparado con la predicción teórica de circ.rafaga_teorica)
#   B) efecto de que C1 corrija 1 o 2 errores (falsas correcciones)
#   C) errores aislados: tasa de error antes y después del código
#
# Uso:  python experimentos.py            (completo, tarda unos minutos)
#       python experimentos.py --rapido   (menos repeticiones)

import sys
import csv
import time
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import circ
import canal

RAPIDO = "--rapido" in sys.argv
N = 300                      # tramas de datos (7200 bytes)
PRUEBAS = 12 if RAPIDO else 40
random.seed(11)
rng = random.Random(11)

datos = bytes(rng.randrange(256) for _ in range(24 * N))


def probar_rafaga(disco, D, largo, t1):
    # un rayón de 'largo' bytes en un lugar al azar (lejos de los bordes del disco)
    lo = 32 * (27 * D + 2)
    hi = 32 * (N - 1) - largo
    ini = rng.randrange(lo, hi)
    rx = canal.rafaga(disco, ini, largo, rng.randrange(10 ** 9))
    rec, dud, info = circ.decodificar_circ(rx, len(datos), D, t1)
    return bytes(rec) == datos


def curva_rafaga(D, t1, largos):
    disco = circ.codificar_circ(datos, D)
    prob = []
    for largo in largos:
        ok = sum(probar_rafaga(disco, D, largo, t1) for _ in range(PRUEBAS))
        prob.append(ok / PRUEBAS)
    return prob


def experimento_A():
    print("A) ráfaga vs. probabilidad de recuperación (C1 corrige 1)")
    resultados = {}
    filas = []
    for D in [0, 1, 2, 4, 8]:
        garantizada, maxima = circ.rafaga_teorica(D, 1)
        if D == 0:
            largos = list(range(1, 11))
        else:
            largos = sorted(set(int(x) for x in np.linspace(0.2 * maxima, 1.15 * maxima, 22)))
            largos += [garantizada, maxima]
            largos = sorted(set(largos))
        t0 = time.time()
        prob = curva_rafaga(D, 1, largos)
        resultados[D] = (largos, prob)
        for l, p in zip(largos, prob):
            filas.append([D, l, p])
        print(f"  D={D}: garantizada={garantizada}, máxima={maxima}  [{time.time() - t0:.0f} s]")
        # comprobación: dentro de lo garantizado debe ser 100 %
        for l, p in zip(largos, prob):
            if l <= garantizada and p < 1:
                print(f"    !! largo {l} <= garantizada pero p={p}")
    with open("salida/resultados_rafaga.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["D", "largo_rafaga_bytes", "prob_recuperacion_exacta"])
        w.writerows(filas)
    return resultados


def experimento_B():
    print("B) C1 corrige 1 vs 2 errores (D=4)")
    D = 4
    g1, m1 = circ.rafaga_teorica(D, 1)
    largos = sorted(set(int(x) for x in np.linspace(0.6 * m1, 1.05 * m1, 16)))
    p1 = curva_rafaga(D, 1, largos)
    p2 = curva_rafaga(D, 2, largos)
    with open("salida/resultados_c1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["largo_rafaga_bytes", "prob_C1_corrige_1", "prob_C1_corrige_2"])
        for l, a, b in zip(largos, p1, p2):
            w.writerow([l, a, b])
    return largos, p1, p2


def experimento_C():
    print("C) errores aislados (cada byte se daña con probabilidad p)")
    D = 4
    disco = circ.codificar_circ(datos, D)
    ps = [0.001, 0.003, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]
    filas = []
    sin = []
    con = []
    reps = 3 if RAPIDO else 8
    for p in ps:
        malos_sin = malos_con = 0
        for _ in range(reps):
            sem = rng.randrange(10 ** 9)
            rx_sin = canal.errores_bytes(np.frombuffer(datos, dtype=np.uint8), p, sem)
            malos_sin += canal.bytes_distintos(np.frombuffer(datos, dtype=np.uint8), rx_sin)
            rx = canal.errores_bytes(disco, p, sem)
            rec, dud, info = circ.decodificar_circ(rx, len(datos), D, 2)
            malos_con += canal.bytes_distintos(np.frombuffer(datos, dtype=np.uint8), rec)
        total = reps * len(datos)
        sin.append(malos_sin / total)
        con.append(malos_con / total)
        filas.append([p, malos_sin / total, malos_con / total])
        print(f"  p={p}: sin código {malos_sin / total:.5f}   con CIRC {malos_con / total:.6f}")
    with open("salida/resultados_aleatorio.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p_byte", "tasa_error_sin_codigo", "tasa_error_con_CIRC"])
        w.writerows(filas)
    return ps, sin, con


def graficar(resA, resB, resC):
    colores = {0: "#d62728", 1: "#ff7f0e", 2: "#2ca02c", 4: "#1f77b4", 8: "#9467bd"}

    # --- figura 1: ráfaga ---
    fig, ejes = plt.subplots(1, 2, figsize=(13, 4.6))
    eje = ejes[0]
    for D, (largos, prob) in resA.items():
        eje.plot(largos, prob, "o-", color=colores[D], markersize=3, label=f"D = {D}")
        g, m = circ.rafaga_teorica(D, 1)
        eje.axvspan(g, m, color=colores[D], alpha=0.12)
        eje.axvline(m, color=colores[D], linestyle=":", linewidth=1)
    eje.set_xscale("log")
    eje.set_xlabel("largo del rayón (bytes seguidos dañados)")
    eje.set_ylabel("probabilidad de recuperar todo")
    eje.set_title("Rayón: efecto del intercalado\n(zona sombreada = teoría: garantizada a máxima)")
    eje.legend()
    eje.grid(alpha=0.3)

    eje = ejes[1]
    largos, p1, p2 = resB
    g1, m1 = circ.rafaga_teorica(4, 1)
    eje.plot(largos, p1, "o-", color="#1f77b4", label="C1 corrige 1 error")
    eje.plot(largos, p2, "s-", color="#d62728", label="C1 corrige 2 errores")
    eje.axvspan(g1, m1, color="gray", alpha=0.15, label="teoría (garantizada a máxima)")
    eje.set_xlabel("largo del rayón (bytes)")
    eje.set_ylabel("probabilidad de recuperar todo")
    eje.set_title("D = 4: falsas correcciones de C1\n(corregir 2 arriesga errores no detectados)")
    eje.legend(loc="lower left")
    eje.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("salida/grafica_rafaga.png", dpi=140)
    plt.close(fig)

    # --- figura 2: errores aislados ---
    ps, sin, con = resC
    fig, eje = plt.subplots(figsize=(6.4, 4.6))
    eje.loglog(ps, sin, "o-", color="#d62728", label="sin código")
    piso = 0.5 / (len(datos) * (3 if RAPIDO else 8))        # valor para poner los ceros
    con_graf = [c if c > 0 else piso for c in con]
    eje.loglog(ps, con_graf, "s-", color="#1f77b4", label="con CIRC")
    for p, c in zip(ps, con):
        if c == 0:
            eje.annotate("0", (p, piso), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    eje.set_xlabel("probabilidad de que un byte se dañe (p)")
    eje.set_ylabel("tasa de error de bytes después de decodificar")
    eje.set_title("Errores aislados: antes y después del código")
    eje.legend()
    eje.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig("salida/grafica_aleatorio.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    t0 = time.time()
    resA = experimento_A()
    resB = experimento_B()
    resC = experimento_C()
    graficar(resA, resB, resC)
    print(f"\nListo en {time.time() - t0:.0f} s. Resultados en la carpeta salida/")
