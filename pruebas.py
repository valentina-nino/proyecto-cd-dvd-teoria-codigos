# pruebas.py
# Pruebas del núcleo. Se ejecuta con:  python pruebas.py   (o: python -m unittest pruebas)

import random
import unittest
import numpy as np
import gf256
import circ
import canal
import disco
import metrica
from reed_solomon import ReedSolomon

random.seed(7)


def prueba_cuerpo():
    # todo elemento no nulo tiene inverso y alfa tiene orden 255
    for a in range(1, 256):
        assert gf256.mult(a, gf256.inverso(a)) == 1
    assert gf256.EXP[255] == 1
    assert len(set(gf256.EXP[:255])) == 255        # alfa es primitivo
    # distributividad en algunos casos al azar
    for _ in range(500):
        a, b, c = [random.randrange(256) for _ in range(3)]
        assert gf256.mult(a, b ^ c) == gf256.mult(a, b) ^ gf256.mult(a, c)
    print("cuerpo GF(256): OK")


def prueba_parametros_rs():
    for n, k in [(28, 24), (32, 28)]:
        rs = ReedSolomon(n, k)
        # toda palabra de código tiene síndromes 0
        for _ in range(50):
            m = [random.randrange(256) for _ in range(k)]
            c = rs.codificar(m)
            assert max(rs.sindromes(c)) == 0
            assert c[:k] == m                       # es sistemático
        # d = n-k+1 = 5: dos mensajes que difieren en UN símbolo dan palabras que difieren en >= 5
        for _ in range(200):
            m1 = [random.randrange(256) for _ in range(k)]
            m2 = list(m1)
            m2[random.randrange(k)] ^= random.randint(1, 255)
            c1, c2 = rs.codificar(m1), rs.codificar(m2)
            dist = sum(1 for x, y in zip(c1, c2) if x != y)
            assert dist >= rs.d
        print(f"RS({n},{k}): d = {rs.d} (MDS, Singleton con igualdad)  OK")


def dañar(palabra, posiciones):
    r = list(palabra)
    for p in posiciones:
        r[p] ^= random.randint(1, 255)
    return r


def prueba_correccion():
    rs = ReedSolomon(28, 24)
    n = 28
    casos = [("2 errores", 2, 0), ("1 error", 1, 0), ("4 borrados", 0, 4),
             ("1 error + 2 borrados", 1, 2), ("3 borrados", 0, 3), ("2 borrados", 0, 2)]
    for nombre, e, s in casos:
        for _ in range(300):
            m = [random.randrange(256) for _ in range(24)]
            c = rs.codificar(m)
            pos = random.sample(range(n), e + s)
            r = dañar(c, pos)
            salida, ok = rs.decodificar(r, borrados=pos[:s])
            assert ok and salida == c, nombre
        print(f"corrige {nombre}: OK  (2e+s = {2 * e + s} <= 4)")

    # con 3 errores NO se garantiza nada: nunca debe devolver "ok" con la palabra original,
    # y casi siempre detecta el fallo (aunque a veces cae en otra palabra de código)
    detectados = 0
    pruebas = 2000
    for _ in range(pruebas):
        m = [random.randrange(256) for _ in range(24)]
        c = rs.codificar(m)
        r = dañar(c, random.sample(range(n), 3))
        salida, ok = rs.decodificar(r)
        if not ok:
            detectados += 1
        else:
            assert salida != c
    print(f"3 errores (más que t=2): detecta que no puede corregir en {detectados}/{pruebas} casos")


def prueba_matrices():
    # G (k x n) y H ((n-k) x n) del Reed-Solomon, y su relación con codificar() y sindromes()
    for n, k in [(28, 24), (32, 28)]:
        rs = ReedSolomon(n, k)
        G = rs.matriz_G()
        H = rs.matriz_H()
        assert len(G) == k and len(G[0]) == n
        assert len(H) == n - k and len(H[0]) == n
        # G * H^T = 0
        for fila in G:
            assert rs.producto_vector_matriz(fila, [[H[j][c] for j in range(n - k)] for c in range(n)]) == [0] * (n - k)
        for _ in range(100):
            m = [random.randrange(256) for _ in range(k)]
            c = rs.producto_vector_matriz(m, G)
            assert c == rs.codificar(m)                              # c = m G
            Hc = [rs.producto_vector_matriz(c, [[H[j][pos]] for pos in range(n)])[0] for j in range(n - k)]
            assert Hc == [0] * (n - k)                               # H c^T = 0
            r = dañar(c, random.sample(range(n), 2))
            Hr = [rs.producto_vector_matriz(r, [[H[j][pos]] for pos in range(n)])[0] for j in range(n - k)]
            assert Hr == rs.sindromes(r)                             # H r^T = síndrome
        print(f"RS({n},{k}): G es {k}x{n}, H es {n - k}x{n}, c = mG, G*H^T = 0, H r = síndrome: OK")


def prueba_circ():
    N = 200
    datos = bytes(random.randrange(256) for _ in range(24 * N))
    for D in [0, 1, 2, 4]:
        disco = circ.codificar_circ(datos, D)
        # sin daño
        rec, dud, info = circ.decodificar_circ(disco, len(datos), D)
        assert bytes(rec) == datos and not dud.any()
        # ráfagas dentro de lo garantizado: siempre se recupera
        garantizada, maxima = circ.rafaga_teorica(D, 1)
        for _ in range(25):
            ini = random.randrange(32 * (27 * D + 2), 32 * (N - 1) - garantizada)
            rx = canal.rafaga(disco, ini, garantizada, random.randrange(10 ** 6))
            rec, dud, info = circ.decodificar_circ(rx, len(datos), D, 1)
            assert bytes(rec) == datos, (D, garantizada)
        # ráfagas más largas que el máximo teórico: no se puede recuperar todo
        if D > 0:
            largo = maxima + 40
            ini = 32 * (27 * D + 5)
            rx = canal.rafaga(disco, ini, largo, 5)
            rec, dud, info = circ.decodificar_circ(rx, len(datos), D, 1)
            assert bytes(rec) != datos
        print(f"CIRC D={D}: sin daño y ráfagas de hasta {garantizada} bytes recuperadas: OK")


def prueba_disco():
    # encabezado: se escribe y se vuelve a leer igual
    img = np.arange(30, dtype=np.uint8)
    aud = np.arange(50, dtype=np.uint8)
    cont = np.frombuffer(disco.empaquetar(img, aud, 5, 2, 3, 44100, 2, 16), dtype=np.uint8)
    h = disco.leer_encabezado(cont)
    assert h["ancho"] == 5 and h["alto"] == 2 and h["canales_img"] == 3
    assert h["fs"] == 44100 and h["canales_audio"] == 2 and h["bits_audio"] == 16
    assert h["largo_imagen"] == 30 and h["largo_audio"] == 50
    im2, au2 = disco.partir(cont, 30, 50)
    assert (im2 == img).all() and (au2 == aud).all()
    # interpolación de audio de 16 bits estéreo: una senoide con un hueco queda casi igual
    fs = 44100
    t = np.arange(4000) / fs
    izq = np.round(8000 * np.sin(2 * np.pi * 440 * t)).astype("<i2")
    der = np.round(8000 * np.sin(2 * np.pi * 330 * t)).astype("<i2")
    est = np.stack([izq, der], axis=1)
    audio = np.frombuffer(est.tobytes(), dtype=np.uint8).copy()
    dud = np.zeros(len(audio), dtype=bool)
    dud[4000:4040] = True                      # 40 bytes dudosos seguidos
    roto = audio.copy()
    roto[4000:4040] = 0
    arreglado = disco.interpolar_audio(roto, dud, 16, 2)
    assert disco.snr_db(audio, arreglado, 16) > disco.snr_db(audio, roto, 16) + 10
    print("disco.py (encabezado, color, audio 16 bits estéreo, interpolación): OK")


def prueba_tamanos():
    # el CIRC funciona con archivos de cualquier tamaño (no solo múltiplos de 24 bytes)
    for tam in [1, 23, 24, 25, 1000, 5001]:
        datos = bytes(random.randrange(256) for _ in range(tam))
        for D in [0, 4]:
            disco_cod = circ.codificar_circ(datos, D)
            rec, dud, info = circ.decodificar_circ(disco_cod, len(datos), D)
            assert bytes(rec) == datos and len(rec) == tam
    print("CIRC con archivos de tamaño arbitrario (1, 23, 24, 25, 1000, 5001 bytes): OK")


def prueba_metrica():
    # el rayón máximo medido coincide con la teoría (D=4, C1 corrige 1: 483 bytes; D=0: 1 byte)
    datos = bytes(random.randrange(256) for _ in range(3000))
    datos = np.frombuffer(datos, dtype=np.uint8)
    con, _ = metrica.rayon_maximo(datos, 4, 1, pruebas=6)
    sin, _ = metrica.rayon_maximo(datos, 0, 1, pruebas=6)
    assert con == circ.rafaga_teorica(4, 1)[0], con
    assert sin == circ.rafaga_teorica(0, 1)[0], sin
    print(f"metrica.py: rayón máximo medido = {con} bytes (D=4) y {sin} byte (D=0), igual que la teoría: OK")


class Pruebas(unittest.TestCase):
    # cada prueba_* de arriba es un test; si algún assert falla, unittest dice cuál
    def test_cuerpo(self):
        prueba_cuerpo()

    def test_parametros_rs(self):
        prueba_parametros_rs()

    def test_correccion(self):
        prueba_correccion()

    def test_matrices(self):
        prueba_matrices()

    def test_circ(self):
        prueba_circ()

    def test_disco(self):
        prueba_disco()

    def test_tamanos(self):
        prueba_tamanos()

    def test_metrica(self):
        prueba_metrica()


if __name__ == "__main__":
    unittest.main(verbosity=2)
