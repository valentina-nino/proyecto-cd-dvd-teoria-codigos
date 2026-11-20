# metrica.py
# Métrica del aplicativo: ¿cuál es el rayón más largo que se recupera EXACTO?
# Se sube el largo del rayón hasta que la recuperación falla en alguna de varias posiciones.

import numpy as np
import circ
import canal


def tolera(grabado, contenido, D, t1, largo, fracciones, semilla):
    # True si el rayón de 'largo' bytes se recupera exacto en TODAS las posiciones probadas
    for f in fracciones:
        inicio = int(f * (len(grabado) - largo))
        recibido = canal.rafaga(grabado, inicio, largo, semilla)
        recuperado = circ.decodificar_circ(recibido, len(contenido), D, t1)[0]
        if canal.bytes_distintos(contenido, recuperado) != 0:
            return False
    return True


def rayon_maximo(contenido, D, t1, pruebas=10, semilla=1, paso=16, tope=2000):
    # Primero sube de 'paso' en 'paso' hasta el primer fallo y luego afina de 1 en 1.
    grabado = circ.codificar_circ(contenido, D)
    fracciones = np.random.default_rng(semilla).random(pruebas)
    bien = 0
    largo = 1
    while largo <= tope and tolera(grabado, contenido, D, t1, largo, fracciones, semilla):
        bien = largo
        largo += paso
    largo = bien + 1
    while largo <= tope and tolera(grabado, contenido, D, t1, largo, fracciones, semilla):
        bien = largo
        largo += 1
    return bien, len(grabado)


def mostrar_barrido(contenido, D, t1, pruebas=10):
    print(f"\nMétrica: rayón más largo recuperado EXACTO en {pruebas} posiciones distintas del disco")
    print("(se sube el rayón hasta que falla; puede tardar unos segundos)")
    for nombre, d in [("sin intercalado (D=0)", 0), (f"con intercalado (D={D})", D)]:
        largo, tam_disco = rayon_maximo(contenido, d, t1, pruebas)
        garantizada, maxima = circ.rafaga_teorica(d, t1)
        print(f"  {nombre:<24} {largo:>4} bytes = {8 * largo:>5} bits = {100 * largo / tam_disco:6.3f} % del disco"
              f"   (teoría: siempre {garantizada}, máximo {maxima})")
    print("  Son cifras de NUESTRO simulador. Las del CD físico (~4 000 bits, ~2,5 mm) son del estándar.")
