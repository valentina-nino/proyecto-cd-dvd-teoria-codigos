# disco.py
# Convierte imagen y audio en una cadena de bytes (el "contenido del disco") y viceversa.
#
# Formato del contenido:
#   encabezado (23 bytes) | imagen (ancho*alto*canales bytes) | audio (bytes PCM)
# Encabezado: magia "CDTC", ancho, alto, canales de la imagen (1 gris, 3 color),
#             frecuencia de muestreo, canales del audio, bits del audio, largo imagen, largo audio.
#
# Se usan formatos SIN comprimir a propósito: en un PNG o JPG un solo byte dañado puede
# arruinar todo lo que sigue y no se vería el efecto del código. (Para archivos comprimidos
# como mp3 o mp4 se usa demo_archivo.py, que graba el archivo tal cual.)

import struct
import wave
import numpy as np
from PIL import Image, ImageOps

FORMATO_ENCABEZADO = ">4sHHBIBBII"
TAM_ENCABEZADO = struct.calcsize(FORMATO_ENCABEZADO)    # 23


# ------------------------------------------------------------------ imagen
def cargar_imagen(ruta, ancho=128, alto=128, color=False):
    # gris: 1 byte por píxel. color: 3 bytes por píxel (R, G, B).
    img = Image.open(ruta).convert("RGB" if color else "L")
    img = ImageOps.fit(img, (ancho, alto))        # recorta para no deformar
    return np.array(img, dtype=np.uint8).reshape(-1)


def guardar_imagen(ruta, datos, ancho, alto, canales=1):
    forma = (alto, ancho) if canales == 1 else (alto, ancho, canales)
    img = np.array(datos, dtype=np.uint8).reshape(forma)
    Image.fromarray(img).save(ruta)


def pixeles_distintos(a, b, canales=1):
    # cuenta píxeles donde algún canal es distinto
    a = np.asarray(a, dtype=np.uint8).reshape(-1, canales)
    b = np.asarray(b, dtype=np.uint8).reshape(-1, canales)
    return int(np.count_nonzero((a != b).any(axis=1)))


# ------------------------------------------------------------------ audio
def cargar_audio(ruta, fs=8000, canales=1, bits=8):
    # Lee un WAV y lo convierte a la calidad pedida:
    #   fs Hz, 'canales' (1 o 2), 'bits' (8 sin signo o 16 con signo).
    w = wave.open(ruta, "rb")
    c_orig = w.getnchannels()
    ancho = w.getsampwidth()
    fs_orig = w.getframerate()
    crudo = w.readframes(w.getnframes())
    w.close()
    if ancho == 1:
        x = (np.frombuffer(crudo, dtype=np.uint8).astype(float) - 128) / 128
    elif ancho == 2:
        x = np.frombuffer(crudo, dtype="<i2").astype(float) / 32768
    else:
        raise ValueError("solo se aceptan WAV de 8 o 16 bits")
    x = x.reshape(-1, c_orig)
    # canales
    if canales == 1:
        x = x.mean(axis=1, keepdims=True)
    elif c_orig == 1:
        x = np.repeat(x, 2, axis=1)
    else:
        x = x[:, :2]
    # remuestreo lineal sencillo
    if fs_orig != fs:
        n_nuevo = int(len(x) * fs / fs_orig)
        pos = np.linspace(0, len(x) - 1, n_nuevo)
        x = np.stack([np.interp(pos, np.arange(len(x)), x[:, c]) for c in range(x.shape[1])], axis=1)
    # bits
    if bits == 8:
        y = np.clip(np.round(x * 127) + 128, 0, 255).astype(np.uint8)
        return y.reshape(-1)
    y = np.clip(np.round(x * 32767), -32768, 32767).astype("<i2")
    return np.frombuffer(y.tobytes(), dtype=np.uint8)


def guardar_audio(ruta, datos, fs=8000, canales=1, bits=8):
    w = wave.open(ruta, "wb")
    w.setnchannels(canales)
    w.setsampwidth(bits // 8)
    w.setframerate(fs)
    datos = bytes(np.array(datos, dtype=np.uint8))
    datos = datos[:len(datos) // (canales * bits // 8) * (canales * bits // 8)]
    w.writeframes(datos)
    w.close()


def a_muestras(audio, bits=8):
    # bytes -> muestras numéricas (8 bits centradas en 0, 16 bits con signo)
    a = np.array(audio, dtype=np.uint8)
    if bits == 8:
        return a.astype(float) - 128
    n = len(a) // 2 * 2
    return np.frombuffer(a[:n].tobytes(), dtype="<i2").astype(float)


def snr_db(original, recuperado, bits=8):
    # relación señal-ruido en dB
    o = a_muestras(original, bits)
    r = a_muestras(recuperado, bits)
    ruido = np.sum((o - r) ** 2)
    if ruido == 0:
        return float("inf")
    return 10 * np.log10(np.sum(o ** 2) / ruido)


def interpolar_audio(audio, dudosos, bits=8, canales=1):
    # Cuando ni C2 pudo recuperar unas muestras, un reproductor de CD las interpola.
    # Aquí: interpolación lineal, canal por canal, entre la última muestra buena y la siguiente.
    audio = np.array(audio, dtype=np.uint8)
    dudosos = np.array(dudosos, dtype=bool)
    if not dudosos.any():
        return audio
    bytes_muestra = bits // 8
    n_muestras = len(audio) // bytes_muestra // canales * canales   # muestras completas
    n_bytes = n_muestras * bytes_muestra
    m = a_muestras(audio[:n_bytes], bits)
    # una muestra es dudosa si alguno de sus bytes lo es
    mal = dudosos[:n_bytes].reshape(n_muestras, bytes_muestra).any(axis=1)
    m = m.reshape(-1, canales)
    mal = mal.reshape(-1, canales)
    for c in range(canales):
        malos = np.where(mal[:, c])[0]
        buenos = np.where(~mal[:, c])[0]
        if len(malos) > 0 and len(buenos) > 0:
            m[malos, c] = np.interp(malos, buenos, m[buenos, c])
    m = m.reshape(-1)
    salida = audio.copy()
    if bits == 8:
        salida[:n_bytes] = np.clip(np.round(m) + 128, 0, 255).astype(np.uint8)
    else:
        y = np.clip(np.round(m), -32768, 32767).astype("<i2")
        salida[:n_bytes] = np.frombuffer(y.tobytes(), dtype=np.uint8)
    return salida


# ------------------------------------------------------------------ el contenido del disco
def empaquetar(imagen=None, audio=None, ancho=0, alto=0, canales_img=1, fs=0, canales_audio=1, bits_audio=8):
    # imagen y audio son arrays uint8 o None. Devuelve el contenido del disco en bytes.
    if imagen is None:
        imagen = np.zeros(0, dtype=np.uint8)
        ancho = alto = 0
    if audio is None:
        audio = np.zeros(0, dtype=np.uint8)
        fs = 0
    enc = struct.pack(FORMATO_ENCABEZADO, b"CDTC", ancho, alto, canales_img, fs,
                      canales_audio, bits_audio, len(imagen), len(audio))
    return enc + bytes(imagen) + bytes(audio)


def leer_encabezado(datos):
    # Devuelve un diccionario con los datos del encabezado, o None si está dañado.
    magia, ancho, alto, ci, fs, ca, ba, li, la = struct.unpack(
        FORMATO_ENCABEZADO, bytes(datos[:TAM_ENCABEZADO]))
    if magia != b"CDTC":
        return None
    return {"ancho": ancho, "alto": alto, "canales_img": ci, "fs": fs,
            "canales_audio": ca, "bits_audio": ba, "largo_imagen": li, "largo_audio": la}


def partir(datos, largo_imagen, largo_audio):
    # separa imagen y audio usando los largos (conocidos por el que grabó el disco)
    datos = np.asarray(datos, dtype=np.uint8)
    ini = TAM_ENCABEZADO
    imagen = datos[ini:ini + largo_imagen]
    audio = datos[ini + largo_imagen: ini + largo_imagen + largo_audio]
    return imagen, audio
