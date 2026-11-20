# hamming_intercalado.py
# Demo previa: ¿por qué sirve el intercalado?
# Usamos Hamming (7,4) (corrige 1 error por palabra, d = 3) y mandamos un texto.
# Se compara qué pasa con una ráfaga de errores con y sin intercalado.

import numpy as np

# Matriz generadora G = [I4 | P] y de control H = [P^T | I3]
P = np.array([[1, 1, 0],
              [1, 0, 1],
              [0, 1, 1],
              [1, 1, 1]])
G = np.concatenate([np.eye(4, dtype=int), P], axis=1)
H = np.concatenate([P.T, np.eye(3, dtype=int)], axis=1)


def texto_a_bits(texto):
    bits = []
    for byte in texto.encode("utf-8"):
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def bits_a_texto(bits):
    bytes_ = []
    for i in range(0, len(bits) - 7, 8):
        v = 0
        for b in bits[i:i + 8]:
            v = v * 2 + b
        bytes_.append(v)
    return bytes(bytes_).decode("utf-8", errors="replace")


def codificar_hamming(bits):
    # cada 4 bits -> palabra de 7 bits. Devuelve lista de palabras (arrays).
    palabras = []
    for i in range(0, len(bits), 4):
        m = np.array(bits[i:i + 4])
        palabras.append(m.dot(G) % 2)
    return palabras


def decodificar_hamming(palabras):
    bits = []
    for r in palabras:
        r = r.copy()
        s = H.dot(r) % 2                    # síndrome
        if s.any():
            # la columna de H que coincide con el síndrome indica la posición del error
            for pos in range(7):
                if (H[:, pos] == s).all():
                    r[pos] ^= 1
                    break
        bits.extend(r[:4].tolist())        # los 4 primeros bits son el mensaje
    return bits


def intercalar(palabras, profundidad):
    # Se escriben las palabras por filas en bloques de 'profundidad' filas x 7 columnas
    # y se transmite por columnas.
    palabras = list(palabras)
    while len(palabras) % profundidad != 0:
        palabras.append(np.zeros(7, dtype=int))     # relleno
    salida = []
    for b in range(0, len(palabras), profundidad):
        bloque = np.array(palabras[b:b + profundidad])   # profundidad x 7
        for col in range(7):
            salida.extend(bloque[:, col].tolist())
    return salida


def desintercalar(bits, profundidad):
    palabras = []
    tam_bloque = profundidad * 7
    for b in range(0, len(bits), tam_bloque):
        trozo = np.array(bits[b:b + tam_bloque])
        bloque = trozo.reshape(7, profundidad).T         # se invierte lo que hicimos
        for fila in range(profundidad):
            palabras.append(bloque[fila])
    return palabras


def rafaga(bits, inicio, largo):
    # invierte 'largo' bits seguidos (un rayón)
    bits = list(bits)
    for i in range(inicio, min(inicio + largo, len(bits))):
        bits[i] ^= 1
    return bits


if __name__ == "__main__":
    mensaje = "Teoria de Codigos: el CD usa Reed-Solomon"
    bits = texto_a_bits(mensaje)
    palabras = codificar_hamming(bits)
    profundidad = 8
    largo_rafaga = 6     # 6 bits seguidos dañados
    inicio = 40

    print("mensaje original  :", mensaje)
    print("bits de datos     :", len(bits), "  bits codificados:", 7 * len(palabras))
    print("ráfaga de", largo_rafaga, "bits seguidos, profundidad de intercalado =", profundidad)
    print()

    # 1) sin código: se daña el mensaje directamente
    dano = rafaga(bits, inicio, largo_rafaga)
    print("sin código         :", bits_a_texto(dano))

    # 2) Hamming sin intercalado
    transmitido = []
    for p in palabras:
        transmitido.extend(p.tolist())
    recibido = rafaga(transmitido, inicio, largo_rafaga)
    pal_rx = [np.array(recibido[i:i + 7]) for i in range(0, len(recibido), 7)]
    print("Hamming            :", bits_a_texto(decodificar_hamming(pal_rx)))

    # 3) Hamming con intercalado
    tx = intercalar(palabras, profundidad)
    rx = rafaga(tx, inicio, largo_rafaga)
    pal_rx2 = desintercalar(rx, profundidad)
    resultado = decodificar_hamming(pal_rx2)[:len(bits)]
    print("Hamming+intercalado:", bits_a_texto(resultado))
    print()
    print("Con intercalado una ráfaga de hasta", profundidad,
          "bits cae en palabras distintas (a lo más 1 error por palabra) y Hamming la corrige.")
