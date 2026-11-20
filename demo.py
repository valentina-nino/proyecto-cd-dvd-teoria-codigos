# demo.py
# Aplicativo del proyecto: "rayar" un disco con imagen y/o audio y recuperarlo con CIRC.
#
# Uso:
#   python demo.py                  -> menú interactivo (para la demostración en vivo)
#   python demo.py --modo ambos --rafaga 480 --D 4
#   python demo.py --modo ambos --color --resolucion 256 --calidad alta
#   (para mp3, mp4 y cualquier otro archivo: python demo_archivo.py)
#
# Se comparan tres casos con el MISMO rayón (misma longitud y posición en el disco):
#   1) sin código
#   2) CIRC sin intercalado (D = 0): los dos Reed-Solomon pero sin reordenar
#   3) CIRC con intercalado (D elegido, en el CD real D = 4)

import os
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")           # las figuras se guardan en archivos; --mostrar las abre
import matplotlib.pyplot as plt

import circ
import canal
import disco
import medios
import metrica

# calidades de audio: (frecuencia, canales, bits)
CALIDADES = {"baja": (8000, 1, 8), "alta": (44100, 2, 16)}


def preguntar(texto, defecto, tipo=str):
    r = input(f"{texto} [{defecto}]: ").strip()
    if r == "":
        return defecto
    return tipo(r)


def leer_parametros():
    p = argparse.ArgumentParser(description="Demo CIRC: rayar un disco con imagen y/o audio")
    p.add_argument("--modo", choices=["imagen", "audio", "ambos"], default="ambos")
    p.add_argument("--imagen", default=None, help="archivo de imagen (png, jpg, ...)")
    p.add_argument("--audio", default=None, help="archivo WAV")
    p.add_argument("--color", action="store_true", help="imagen a color (3 bytes por píxel)")
    p.add_argument("--resolucion", type=int, default=128, help="lado de la imagen en píxeles")
    p.add_argument("--calidad", choices=["baja", "alta"], default="baja",
                   help="baja: 8 kHz 8 bits mono. alta: 44,1 kHz 16 bits estéreo")
    p.add_argument("--rafaga", type=int, default=480, help="largo del rayón en bytes")
    p.add_argument("--posicion", type=float, default=12, help="dónde cae el rayón (0 a 100 %% del disco)")
    p.add_argument("--D", type=int, default=4, help="profundidad del intercalado (CD real: 4)")
    p.add_argument("--t1", type=int, default=1, choices=[1, 2], help="errores que corrige C1")
    p.add_argument("--semilla", type=int, default=1)
    p.add_argument("--mostrar", action="store_true", help="abrir las gráficas al terminar")
    p.add_argument("--barrido", action="store_true",
                   help="calcular el rayón máximo recuperado (con y sin intercalado)")
    if len(sys.argv) == 1:
        # sin argumentos: menú interactivo
        a = p.parse_args([])
        print("=== Demo CD: Reed-Solomon con intercalado (CIRC) ===")
        resp = preguntar("Contenido del disco (1 imagen, 2 audio, 3 ambos)", "3")
        a.modo = {"1": "imagen", "2": "audio"}.get(resp, "ambos")      # cualquier otra cosa = ambos
        if a.modo != "audio":
            a.color = preguntar("Imagen a color (s/n)", "n").lower().startswith("s")
            a.resolucion = preguntar("Resolución de la imagen (lado en píxeles)", a.resolucion, int)
        if a.modo != "imagen":
            resp = preguntar("Calidad del audio (1 baja 8 kHz mono, 2 alta 44,1 kHz estéreo)", "1")
            a.calidad = "alta" if resp == "2" else "baja"       # cualquier otra cosa = baja
        a.rafaga = preguntar("Largo del rayón en bytes", a.rafaga, int)
        a.posicion = preguntar("Posición del rayón en el disco (0-100 %)", a.posicion, float)
        a.D = preguntar("Profundidad del intercalado D (0 = sin intercalado)", a.D, int)
        a.t1 = preguntar("Errores que corrige C1 (1 o 2)", a.t1, int)
        a.mostrar = True            # en el menú se abren las gráficas solas
        return a
    return p.parse_args()


def resumen(nombre, original, recuperado, extra=""):
    malos = canal.bytes_distintos(original, recuperado)
    print(f"  {nombre:<28} bytes malos: {malos:>7} de {len(original)}  "
          f"({100 * malos / len(original):6.2f} %)  {extra}")
    return malos


def envolvente(muestras, cubos=1500):
    # para dibujar audio largo: mínimo y máximo de cada tramo
    n = len(muestras) // cubos * cubos
    m = muestras[:n].reshape(cubos, -1)
    return m.min(axis=1), m.max(axis=1)


def armar_contenido(a):
    # Arma el contenido del disco (encabezado + imagen + audio) a partir de los parámetros.
    # Devuelve el contenido y un diccionario 'f' con el formato, que usan los pasos siguientes.
    f = {"ancho": a.resolucion, "alto": a.resolucion, "canales_img": 3 if a.color else 1}
    f["fs"], f["canales_audio"], f["bits"] = CALIDADES[a.calidad]
    ruta_imagen = a.imagen or ("datos/imagen_color.png" if a.color else "datos/imagen_ejemplo.png")
    ruta_audio = a.audio or ("datos/audio_alta.wav" if a.calidad == "alta" else "datos/audio_ejemplo.wav")

    imagen = audio = None
    if a.modo in ("imagen", "ambos"):
        imagen = disco.cargar_imagen(ruta_imagen, f["ancho"], f["alto"], a.color)
    if a.modo in ("audio", "ambos"):
        audio = disco.cargar_audio(ruta_audio, f["fs"], f["canales_audio"], f["bits"])
    contenido = np.frombuffer(
        disco.empaquetar(imagen, audio, f["ancho"], f["alto"], f["canales_img"],
                         f["fs"], f["canales_audio"], f["bits"]), dtype=np.uint8)
    f["li"] = 0 if imagen is None else len(imagen)
    f["la"] = 0 if audio is None else len(audio)
    print(f"\nContenido: {len(contenido)} bytes  (imagen {f['li']}, audio {f['la']})")
    return contenido, imagen, audio, f


def grabar_y_rayar(a, contenido):
    # Codifica el contenido (con intercalado D y sin intercalado) y le aplica el MISMO rayón
    # a los tres casos. Devuelve lo que "se lee" del disco en cada caso.
    t0 = time.time()
    grabado_D = circ.codificar_circ(contenido, a.D)
    grabado_0 = circ.codificar_circ(contenido, 0)
    print(f"Disco codificado: {len(grabado_D)} bytes  (tasa 24/32 = 3/4 por RS, más {27 * a.D} tramas "
          f"de retardo)  [{time.time() - t0:.1f} s]")

    B = a.rafaga
    garantizada, maxima = circ.rafaga_teorica(a.D, a.t1)
    print(f"\nRayón de {B} bytes seguidos al {a.posicion:.0f} % del disco.")
    print(f"Teoría con D={a.D}, C1 corrige {a.t1}: siempre se recupera hasta {garantizada} bytes; "
          f"entre {garantizada} y {maxima} depende de la alineación; más de {maxima} no se recupera.")

    # el rayón empieza en el mismo porcentaje del disco en cada caso
    rx_sin = canal.rafaga(contenido, int(a.posicion / 100 * len(contenido)), B, a.semilla)
    rx_0 = canal.rafaga(grabado_0, int(a.posicion / 100 * len(grabado_0)), B, a.semilla)
    rx_D = canal.rafaga(grabado_D, int(a.posicion / 100 * len(grabado_D)), B, a.semilla)
    return rx_sin, rx_0, rx_D


def decodificar_casos(a, contenido, rx_0, rx_D):
    t0 = time.time()
    rec_0, dud_0, info_0 = circ.decodificar_circ(rx_0, len(contenido), 0, a.t1)
    rec_D, dud_D, info_D = circ.decodificar_circ(rx_D, len(contenido), a.D, a.t1)
    print(f"Decodificación terminada [{time.time() - t0:.1f} s]\n")
    return rec_0, info_0, rec_D, dud_D, info_D


def mostrar_metricas(a, contenido, rx_sin, rec_0, info_0, rec_D, info_D):
    print("Resultados (bytes del contenido que quedaron mal):")
    resumen("sin código", contenido, rx_sin)
    resumen("CIRC sin intercalado (D=0)", contenido, rec_0,
            f"tramas C1 fallidas: {info_0['c1_fallidas']}, palabras C2 fallidas: {info_0['c2_fallidas']}")
    m_D = resumen(f"CIRC con intercalado (D={a.D})", contenido, rec_D,
                  f"tramas C1 fallidas: {info_D['c1_fallidas']}, palabras C2 fallidas: {info_D['c2_fallidas']}")
    if m_D == 0:
        print("\n  -> con intercalado el disco se recuperó EXACTAMENTE (0 bytes malos).")
    hdr = disco.leer_encabezado(rec_D)
    print("  encabezado del disco recuperado:", "OK" if hdr is not None else "DAÑADO")


def figura_imagen(a, f, imagen, rx_sin, rec_0, rec_D):
    li, la = f["li"], f["la"]
    paneles = [("Original", imagen),
               ("Sin código", disco.partir(rx_sin, li, la)[0]),
               ("CIRC sin intercalado", disco.partir(rec_0, li, la)[0]),
               (f"CIRC con intercalado D={a.D}", disco.partir(rec_D, li, la)[0])]
    canales = f["canales_img"]
    forma = (f["alto"], f["ancho"]) if canales == 1 else (f["alto"], f["ancho"], canales)
    fig, ejes = plt.subplots(1, 4, figsize=(14, 3.8))
    for eje, (titulo, datos) in zip(ejes, paneles):
        eje.imshow(np.array(datos).reshape(forma), cmap="gray" if canales == 1 else None,
                   vmin=0, vmax=255, interpolation="nearest")
        if titulo == "Original":
            eje.set_title(titulo, fontsize=10)
        else:
            malos_img = disco.pixeles_distintos(imagen, datos, canales)
            eje.set_title(f"{titulo}\n{malos_img} píxeles mal", fontsize=10)
        eje.axis("off")
    fig.suptitle(f"Rayón de {a.rafaga} bytes  ({f['ancho']}x{f['alto']}, {'color' if a.color else 'gris'})",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("salida/comparacion_imagen.png", dpi=130)
    print("\nGuardado: salida/comparacion_imagen.png")
    plt.close(fig)
    if a.mostrar:
        medios.abrir_archivo("salida/comparacion_imagen.png")


def salida_audio(a, f, audio, rx_sin, rec_0, rec_D, dud_D):
    # separa el audio de cada caso, interpola lo que C2 no pudo recuperar, guarda los WAV
    # y calcula la relación señal-ruido. Devuelve los cuatro audios (original, sin código, D=0, D).
    li, la, fs, bits, canales = f["li"], f["la"], f["fs"], f["bits"], f["canales_audio"]
    a_sin = disco.partir(rx_sin, li, la)[1]
    a_0 = disco.partir(rec_0, li, la)[1]
    a_D = disco.partir(rec_D, li, la)[1]
    # muestras que ni C2 pudo recuperar -> se interpolan (como hace un reproductor)
    ini = disco.TAM_ENCABEZADO + li
    dud_audio = dud_D[ini:ini + la]
    a_D_interp = disco.interpolar_audio(a_D, dud_audio, bits, canales)
    if dud_audio.any():
        print(f"\nAudio: {int(dud_audio.sum())} bytes no recuperados por C2, se interpolan.")

    disco.guardar_audio("salida/audio_original.wav", audio, fs, canales, bits)
    disco.guardar_audio("salida/audio_sin_codigo.wav", a_sin, fs, canales, bits)
    disco.guardar_audio("salida/audio_circ_sin_intercalado.wav", a_0, fs, canales, bits)
    disco.guardar_audio("salida/audio_circ.wav", a_D_interp, fs, canales, bits)

    print(f"\nAudio ({fs} Hz, {bits} bits, {'estéreo' if canales == 2 else 'mono'}) "
          f"- relación señal-ruido, más alto es mejor:")
    for nombre, x in [("sin código", a_sin), ("CIRC sin intercalado", a_0),
                      (f"CIRC con intercalado D={a.D}", a_D_interp)]:
        s = disco.snr_db(audio, x, bits)
        txt = "idéntico al original" if s == float("inf") else f"{s:.1f} dB"
        print(f"  {nombre:<28} {txt}")
    return a_sin, a_0, a_D_interp


def figura_audio(a, f, audio, a_sin, a_0, a_D_interp):
    fs, bits, canales = f["fs"], f["bits"], f["canales_audio"]
    fig, ejes = plt.subplots(4, 1, figsize=(10, 6), sharex=True, sharey=True)
    for eje, (titulo, x) in zip(ejes, [("Original", audio), ("Sin código", a_sin),
                                       ("CIRC sin intercalado", a_0),
                                       (f"CIRC con intercalado D={a.D}", a_D_interp)]):
        m = disco.a_muestras(x, bits)
        m = m[:len(m) // canales * canales].reshape(-1, canales)[:, 0]
        lo, hi = envolvente(m)
        t = np.linspace(0, len(m) / fs, len(lo))
        eje.fill_between(t, lo, hi, color="black", linewidth=0)
        eje.set_title(titulo, fontsize=9, loc="left")
    ejes[-1].set_xlabel("tiempo (s)")
    fig.suptitle(f"Audio con un rayón de {a.rafaga} bytes", fontsize=12)
    fig.tight_layout()
    fig.savefig("salida/comparacion_audio.png", dpi=130)
    print("Guardado: salida/comparacion_audio.png y los .wav en salida/")
    plt.close(fig)
    if a.mostrar:
        medios.abrir_archivo("salida/comparacion_audio.png")


def main():
    a = leer_parametros()
    os.makedirs("salida", exist_ok=True)

    contenido, imagen, audio, f = armar_contenido(a)                 # 1) contenido del disco
    rx_sin, rx_0, rx_D = grabar_y_rayar(a, contenido)                # 2) codificar y 3) rayón
    rec_0, info_0, rec_D, dud_D, info_D = decodificar_casos(a, contenido, rx_0, rx_D)   # 4) decodificar
    mostrar_metricas(a, contenido, rx_sin, rec_0, info_0, rec_D, info_D)                # 5) métrica
    if a.barrido:
        metrica.mostrar_barrido(contenido, a.D, a.t1)                # 5b) rayón máximo

    if imagen is not None:
        figura_imagen(a, f, imagen, rx_sin, rec_0, rec_D)            # 6) salidas de imagen
    if audio is not None:
        a_sin, a_0, a_D_interp = salida_audio(a, f, audio, rx_sin, rec_0, rec_D, dud_D)   # 7) audio
        figura_audio(a, f, audio, a_sin, a_0, a_D_interp)


if __name__ == "__main__":
    main()
