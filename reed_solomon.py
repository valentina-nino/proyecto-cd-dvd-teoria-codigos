# reed_solomon.py
# Código Reed-Solomon RS(n, k) sobre GF(256), hecho por el grupo (sin librerías de RS).
#
# Parámetros: n <= 255, k mensajes, n-k = nsym símbolos de redundancia.
# Es un código MDS: d = n - k + 1 (cota de Singleton con igualdad).
# Con d = n-k+1 se pueden corregir e errores y s borrados si  2e + s <= n-k.
#
# Convención: la palabra es una lista de n bytes; palabra[0] es el coeficiente de x^(n-1)
# y palabra[n-1] el de x^0. Es sistemático: los primeros k bytes son el mensaje y los
# últimos n-k bytes son la redundancia (paridad).
# Las raíces del generador son alfa^0, alfa^1, ..., alfa^(n-k-1).

import gf256


class ReedSolomon:

    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.nsym = n - k
        self.d = n - k + 1
        # polinomio generador g(x) = (x - alfa^0)(x - alfa^1)...(x - alfa^(nsym-1))
        # (escrito con el coeficiente de mayor grado primero)
        g = [1]
        for i in range(self.nsym):
            g = gf256.poly_mult(g, [1, gf256.EXP[i]])
        self.generador = g

    # ---------- codificador ----------
    def codificar(self, mensaje):
        # mensaje: lista de k bytes -> palabra de código de n bytes
        if len(mensaje) != self.k:
            raise ValueError("el mensaje debe tener k bytes")
        # dividimos m(x) * x^nsym entre g(x); el residuo es la redundancia
        resto = list(mensaje) + [0] * self.nsym
        for i in range(self.k):
            coef = resto[i]
            if coef != 0:
                for j in range(1, len(self.generador)):
                    resto[i + j] ^= gf256.mult(self.generador[j], coef)
        redundancia = resto[self.k:]
        return list(mensaje) + redundancia

    # ---------- síndromes ----------
    def evaluar(self, palabra, x):
        # evalúa la palabra (mayor grado primero) en x con la regla de Horner
        r = 0
        for c in palabra:
            r = gf256.mult(r, x) ^ c
        return r

    def sindromes(self, palabra):
        # S_j = r(alfa^j), j = 0 .. nsym-1. Si todos son 0, r es palabra de código.
        return [self.evaluar(palabra, gf256.EXP[j]) for j in range(self.nsym)]

    # ---------- matrices G y H ----------
    # El código es lineal sobre GF(256), así que tiene matriz generadora G (k x n) y matriz de
    # control H ((n-k) x n). El decodificador no las usa directamente (codifica con la división
    # polinómica y calcula síndromes evaluando), pero son equivalentes:
    #   - fila j de H: (alfa^(j*(n-1)), ..., alfa^j, 1), así  H*r = (r(alfa^0), ..., r(alfa^(n-k-1)))
    #     que es exactamente el vector de síndromes de la función sindromes().
    #   - fila i de G: la palabra de código del mensaje con un 1 en la posición i (forma sistemática
    #     G = [I | P]), así  c = m*G  da la misma palabra que codificar(m).
    # Además  G * H^T = 0  (toda palabra generada cumple las restricciones de H).
    def matriz_H(self):
        H = []
        for j in range(self.nsym):
            H.append([gf256.potencia(gf256.EXP[1], j * (self.n - 1 - pos)) for pos in range(self.n)])
        return H

    def matriz_G(self):
        G = []
        for i in range(self.k):
            unitario = [0] * self.k
            unitario[i] = 1
            G.append(self.codificar(unitario))
        return G

    def producto_vector_matriz(self, v, M):
        # v (largo f) por M (f x c) en GF(256): suma = XOR, producto = gf256.mult
        res = [0] * len(M[0])
        for i in range(len(v)):
            if v[i] != 0:
                for j in range(len(M[0])):
                    res[j] ^= gf256.mult(v[i], M[i][j])
        return res

    # ---------- decodificador ----------
    def decodificar(self, recibida, borrados=None, max_errores=None):
        # recibida: lista de n bytes
        # borrados: posiciones (índices 0..n-1) que se sabe que están dañadas
        # max_errores: tope de errores desconocidos que se aceptan corregir (por defecto
        #              todos los que permite la distancia)
        # devuelve (palabra_corregida, ok). Si ok es False no se pudo corregir.
        if borrados is None:
            borrados = []
        n = self.n
        nsym = self.nsym
        palabra = list(recibida)

        S = self.sindromes(palabra)
        if max(S) == 0:
            return palabra, True            # no hay errores

        s = len(borrados)
        if s > nsym:
            return palabra, False           # demasiados borrados

        # 1) polinomio localizador de borrados  Gamma(x) = prod (1 - X_i x)
        #    con X_i = alfa^(n-1-posicion). (coeficiente independiente primero)
        gamma = [1]
        for pos in borrados:
            X = gf256.EXP[n - 1 - pos]
            gamma = gf256.poly_mult(gamma, [1, X])

        # 2) síndromes modificados: se multiplica S(x) por Gamma(x) y nos quedamos con los
        #    coeficientes de s a nsym-1. Así los borrados "desaparecen" del problema.
        producto = gf256.poly_mult(S, gamma)
        T = producto[s:nsym]

        # 3) Berlekamp-Massey sobre T para hallar el localizador de errores Lambda(x)
        Lambda, L = self.berlekamp_massey(T)
        if 2 * L > nsym - s:
            return palabra, False           # hay más errores de los que se pueden corregir
        if max_errores is not None and L > max_errores:
            return palabra, False

        # 4) localizador total y búsqueda de raíces (se prueba cada posición)
        psi = gf256.poly_mult(gamma, Lambda)
        grado = len(psi) - 1
        while grado > 0 and psi[grado] == 0:
            grado -= 1
        posiciones = []
        for pos in range(n):
            x = gf256.EXP[(255 - (n - 1 - pos)) % 255]   # alfa^-(n-1-pos)
            if gf256.poly_eval(psi, x) == 0:
                posiciones.append(pos)
        if len(posiciones) != grado:
            return palabra, False           # el localizador no tiene todas sus raíces

        # 5) valores de error: se resuelve el sistema  sum_i e_i * X_i^j = S_j
        m = len(posiciones)
        if m > nsym:
            return palabra, False
        A = []
        for j in range(m):
            fila = []
            for pos in posiciones:
                X = gf256.EXP[n - 1 - pos]
                fila.append(gf256.potencia(X, j))
            A.append(fila)
        b = S[:m]
        valores = self.resolver(A, b)
        if valores is None:
            return palabra, False
        for i in range(m):
            palabra[posiciones[i]] ^= valores[i]

        # 6) comprobación final: la palabra corregida debe tener síndromes nulos
        if max(self.sindromes(palabra)) != 0:
            return palabra, False
        return palabra, True

    def berlekamp_massey(self, T):
        # Devuelve (Lambda, L): polinomio localizador (coef. independiente primero)
        # y su grado L, el que genera la secuencia T.
        C = [1]       # polinomio actual
        B = [1]       # polinomio anterior
        L = 0
        m = 1
        b = 1
        for i in range(len(T)):
            # discrepancia
            d = T[i]
            for j in range(1, L + 1):
                if j < len(C):
                    d ^= gf256.mult(C[j], T[i - j])
            if d == 0:
                m += 1
            else:
                # C(x) = C(x) - (d/b) x^m B(x)
                factor = gf256.div(d, b)
                nuevo = list(C)
                while len(nuevo) < len(B) + m:
                    nuevo.append(0)
                for j in range(len(B)):
                    nuevo[j + m] ^= gf256.mult(factor, B[j])
                if 2 * L <= i:
                    B = list(C)
                    L = i + 1 - L
                    b = d
                    m = 1
                else:
                    m += 1
                C = nuevo
        return C, L

    def resolver(self, A, b):
        # Eliminación de Gauss en GF(256). A es m x m. Devuelve la solución o None.
        m = len(A)
        M = [list(A[i]) + [b[i]] for i in range(m)]
        for col in range(m):
            # buscar pivote
            piv = None
            for fila in range(col, m):
                if M[fila][col] != 0:
                    piv = fila
                    break
            if piv is None:
                return None
            M[col], M[piv] = M[piv], M[col]
            inv = gf256.inverso(M[col][col])
            M[col] = [gf256.mult(inv, v) for v in M[col]]
            for fila in range(m):
                if fila != col and M[fila][col] != 0:
                    f = M[fila][col]
                    M[fila] = [M[fila][c] ^ gf256.mult(f, M[col][c]) for c in range(m + 1)]
        return [M[i][m] for i in range(m)]
