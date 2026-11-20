# circ.py
# Mini-CIRC (Cross-Interleaved Reed-Solomon Code), versión simplificada del que usa el CD.
#
#   datos (24 bytes por trama) --> C2 = RS(28,24) --> intercalado con retardos
#                              --> C1 = RS(32,28) --> trama de 32 bytes al disco
#
# Los dos códigos tienen d = n-k+1 = 5, o sea corrigen 2 errores o 4 borrados.
# Tasa total = 24/32 = 3/4.
#
# Intercalado: el byte i (0..27) de cada palabra de C2 se retrasa i*D tramas.
# En el CD real D = 4. Con D = 0 no hay intercalado.
#
# Al decodificar: primero C1. Si C1 no puede corregir una trama, se marcan sus 28 bytes
# como BORRADOS (sabemos que están mal, y dónde). Se deshace el intercalado y C2 corrige
# usando esas marcas. Un borrado "cuesta" la mitad que un error: 2e + s <= 4.
#
# Simplificaciones respecto al CD real: no hay modulación EFM, ni el retardo extra de
# 1 trama en los bytes pares/impares, ni la inversión de paridad.

import numpy as np
from reed_solomon import ReedSolomon

C2 = ReedSolomon(28, 24)
C1 = ReedSolomon(32, 28)


def tramas_de_datos(num_bytes):
    # cuántas tramas de 24 bytes se necesitan (se rellena con ceros)
    return (num_bytes + 23) // 24


def codificar_circ(datos, D=4):
    # datos: bytes. Devuelve un array de bytes (uint8) con todo lo que se graba.
    datos = bytes(datos)
    N = tramas_de_datos(len(datos))
    relleno = datos + bytes(N * 24 - len(datos))

    # 1) C2 sobre cada trama de 24 bytes -> 28 bytes
    c2 = np.zeros((N, 28), dtype=np.uint8)
    for j in range(N):
        mensaje = list(relleno[j * 24:(j + 1) * 24])
        c2[j] = C2.codificar(mensaje)

    # 2) intercalado: el byte i de la palabra j se manda en la trama j + i*D
    T = N + 27 * D                      # tramas totales (las últimas vacían los retardos)
    inter = np.zeros((T, 28), dtype=np.uint8)
    for i in range(28):
        inter[i * D: i * D + N, i] = c2[:, i]

    # 3) C1 sobre cada trama de 28 bytes -> 32 bytes
    disco = np.zeros((T, 32), dtype=np.uint8)
    for t in range(T):
        disco[t] = C1.codificar(inter[t].tolist())

    return disco.reshape(-1)


def decodificar_circ(recibido, num_bytes, D=4, max_errores_c1=1):
    # recibido: lo que se leyó del disco (con daños). num_bytes: largo de los datos originales.
    # max_errores_c1: cuántos errores corrige C1 (la distancia permite 2). Se usa 1 por defecto
    #   porque C1 con 2 "corrige" mal (falsa corrección) una trama con 3 o más errores con
    #   probabilidad |B_2|/256^4 = 0.75%, mientras que con 1 es |B_1|/256^4 = 0.0002%.
    #   Los lectores de CD reales hacen algo parecido.
    # Devuelve (datos, dudosos, info):
    #   datos   : array uint8 con los datos recuperados
    #   dudosos : array bool, True en los bytes que se sabe que no se pudieron recuperar
    #   info    : estadísticas del decodificador
    N = tramas_de_datos(num_bytes)
    T = N + 27 * D
    recibido = np.asarray(recibido, dtype=np.uint8)
    tramas = recibido.reshape(T, 32)

    # --- decodificador C1 ---
    inter = np.zeros((T, 28), dtype=np.uint8)
    borrado = np.zeros((T, 28), dtype=bool)
    c1_fallidas = 0
    for t in range(T):
        palabra, ok = C1.decodificar(tramas[t].tolist(), max_errores=max_errores_c1)
        inter[t] = palabra[:28]
        if not ok:
            borrado[t, :] = True        # toda la trama queda marcada como borrada
            c1_fallidas += 1

    # --- deshacer el intercalado y decodificador C2 ---
    datos = np.zeros(N * 24, dtype=np.uint8)
    dudosos = np.zeros(N * 24, dtype=bool)
    c2_fallidas = 0
    for j in range(N):
        palabra = [int(inter[j + i * D, i]) for i in range(28)]
        marcas = [i for i in range(28) if borrado[j + i * D, i]]
        salida, ok = C2.decodificar(palabra, borrados=marcas)
        if ok:
            datos[j * 24:(j + 1) * 24] = salida[:24]
        else:
            c2_fallidas += 1
            datos[j * 24:(j + 1) * 24] = palabra[:24]
            for i in marcas:
                if i < 24:
                    dudosos[j * 24 + i] = True

    info = {"tramas": T, "c1_fallidas": c1_fallidas, "c2_fallidas": c2_fallidas}
    return datos[:num_bytes], dudosos[:num_bytes], info


def rafaga_teorica(D, t1=1):
    # Predicción teórica de la longitud de ráfaga (en bytes del disco) que se corrige.
    # t1 = errores que C1 se atreve a corregir (2 por la distancia, pero se puede usar 1).
    # Devuelve (garantizada, maxima):
    #   garantizada: toda ráfaga de hasta ese largo se corrige, caiga donde caiga.
    #   maxima: ráfagas más largas solo se corrigen si caen "bien alineadas" con las tramas.
    #
    # Razonamiento: C1 falla en una trama si le dañan más de t1 bytes. Una ráfaga de B bytes
    # cubre por completo f tramas y deja 2 tramas parciales en los extremos (a y b bytes).
    # Cada trama donde C1 falla se vuelve un borrado en la palabra de C2 que cruza por ahí;
    # con retardos de D tramas, m tramas malas seguidas dejan a lo más ceil(m/D) borrados
    # por palabra de C2, y C2 corrige 4 borrados. Entonces se necesita m <= 4*D.
    #   - Mejor caso (extremos con t1 bytes, C1 los corrige): B = t1 + 32*(4D) + t1.
    #   - Peor caso (extremos con t1+1 bytes, C1 falla ahí también): hay f+2 tramas malas, y se
    #     exige f+2 <= 4D, lo que da  B <= 2(t1+1) + 32(4D-1) - 1.
    # Sin intercalado (D = 0) basta una trama mala para perder la palabra de C2.
    if D == 0:
        return t1, 2 * t1
    maxima = 128 * D + 2 * t1
    garantizada = 128 * D + 2 * t1 - 31
    return garantizada, maxima
