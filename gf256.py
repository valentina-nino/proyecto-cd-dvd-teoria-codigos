# gf256.py
# Aritmética del cuerpo finito GF(2^8) = GF(256).
# Los elementos son bytes (0 a 255). Se construye con el polinomio primitivo
# x^8 + x^4 + x^3 + x^2 + 1  (0x11d), el mismo que usan los QR y los CD.
# alfa = 2 es un elemento primitivo: sus potencias recorren los 255 elementos no nulos.

POLI = 0x11d

# tablas de exponenciales y logaritmos
EXP = [0] * 512   # EXP[i] = alfa^i  (se repite para no tener que hacer mod 255)
LOG = [0] * 256   # LOG[x] = i tal que alfa^i = x

x = 1
for i in range(255):
    EXP[i] = x
    LOG[x] = i
    x = x << 1          # multiplicar por alfa
    if x >= 256:
        x = x ^ POLI    # reducir módulo el polinomio primitivo
for i in range(255, 512):
    EXP[i] = EXP[i - 255]


def suma(a, b):
    # en característica 2 sumar y restar es lo mismo: XOR
    return a ^ b


def mult(a, b):
    if a == 0 or b == 0:
        return 0
    return EXP[LOG[a] + LOG[b]]


def inverso(a):
    if a == 0:
        raise ZeroDivisionError("0 no tiene inverso")
    return EXP[255 - LOG[a]]


def div(a, b):
    if b == 0:
        raise ZeroDivisionError("division por 0")
    if a == 0:
        return 0
    return EXP[LOG[a] + 255 - LOG[b]]


def potencia(a, n):
    if a == 0:
        return 0 if n > 0 else 1
    return EXP[(LOG[a] * n) % 255]


# ---- polinomios con coeficientes en GF(256) ----
# Se guardan como lista [c0, c1, c2, ...] donde ci es el coeficiente de x^i
# (el término independiente va primero).

def poly_mult(p, q):
    res = [0] * (len(p) + len(q) - 1)
    for i in range(len(p)):
        for j in range(len(q)):
            res[i + j] ^= mult(p[i], q[j])
    return res


def poly_eval(p, x):
    # regla de Horner
    resultado = 0
    for coef in reversed(p):
        resultado = mult(resultado, x) ^ coef
    return resultado
