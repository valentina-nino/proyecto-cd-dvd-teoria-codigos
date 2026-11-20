# demo_archivo.py
# Graba CUALQUIER archivo (mp3, mp4, jpg, png, wav, pdf, zip...) en el disco simulado,
# lo raya, lo recupera con CIRC y comprueba si quedó idéntico (SHA-256).
#
# Idea: Reed-Solomon trabaja con bytes y no sabe qué contienen, así que el formato da igual.
# Lo interesante es lo que pasa SIN código: en un formato comprimido (mp3, mp4, jpg) un byte
# dañado no afecta solo a un píxel o una muestra, sino que se propaga y a veces el archivo
# deja de abrirse. Con CIRC el archivo se recupera bit por bit.
#
# Uso:
#   python demo_archivo.py datos/video_ejemplo.mp4
#   python demo_archivo.py datos/audio_ejemplo.mp3 --rafaga 480 --posicion 40 --D 4
#   python demo_archivo.py mi_archivo.pdf --rapido        (omite el caso D=0, más veloz)

import os
import io
import time
import hashlib
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True      # que muestre lo que alcance a decodificar, como un visor real

import circ
import canal
import medios

EXT_IMAGEN = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
EXT_AUDIO = {".mp3", ".wav", ".aac", ".m4a", ".ogg", ".flac", ".opus"}
EXT_VIDEO = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}


def sha(datos):
    return hashlib.sha256(bytes(datos)).hexdigest()


def guardar(ruta, datos):
    with open(ruta, "wb") as f:
        f.write(bytes(np.asarray(datos, dtype=np.uint8)))


# ---------------------------------------------------------------- revisión según el tipo
def estado_ffmpeg(ruta):
    # ffmpeg solo avisa cuando el flujo de datos es inconsistente; un archivo puede decodificar
    # "sin errores" y aun así tener fallas visibles o audibles. La prueba real es el SHA-256.
    abre, errores = medios.revisar_con_ffmpeg(ruta)
    if abre and errores == 0:
        return "ffmpeg: sin mensajes de error"
    if abre:
        return f"ffmpeg: {errores} mensajes de error"
    return f"ffmpeg: NO decodifica ({errores} mensajes)"


def abrir_imagen(datos):
    # devuelve un array RGB o None si la imagen no abre
    try:
        img = Image.open(io.BytesIO(bytes(datos)))
        img.load()
        return np.array(img.convert("RGB"))
    except Exception:
        return None


def figura_imagen(original, versiones, B, ruta_png):
    ref = abrir_imagen(original)
    fig, ejes = plt.subplots(1, 1 + len(versiones), figsize=(3.6 * (1 + len(versiones)), 4))
    ejes[0].imshow(ref)
    ejes[0].set_title("Original", fontsize=10)
    for eje, (nombre, datos) in zip(ejes[1:], versiones):
        img = abrir_imagen(datos)
        if img is None:
            eje.text(0.5, 0.5, "Pillow no puede abrirla", ha="center", va="center", fontsize=11)
            eje.set_title(nombre, fontsize=10)
        else:
            if img.shape == ref.shape:
                malos = int(np.count_nonzero((img != ref).any(axis=2)))
                eje.set_title(f"{nombre}\n{malos} píxeles mal", fontsize=10)
            else:
                eje.set_title(f"{nombre}\n(tamaño distinto)", fontsize=10)
            eje.imshow(img)
    for eje in ejes:
        eje.axis("off")
    fig.suptitle(f"Rayón de {B} bytes", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(ruta_png, dpi=130)
    plt.close(fig)


def figura_audio(rutas, B, ruta_png):
    # rutas: lista de (nombre, ruta_archivo); la primera es el original
    fs = 8000
    senales = []
    for nombre, ruta in rutas:
        x = medios.decodificar_audio(ruta, fs)
        senales.append((nombre, x))
    ref = senales[0][1]
    print("\nAudio decodificado (relación señal-ruido contra el original, más alto es mejor):")
    fig, ejes = plt.subplots(len(senales), 1, figsize=(10, 1.6 * len(senales) + 1), sharex=True, sharey=True)
    for eje, (nombre, x) in zip(ejes, senales):
        if x is None:
            eje.text(0.5, 0.5, "no se puede decodificar", ha="center", va="center", transform=eje.transAxes)
            eje.set_title(nombre, fontsize=9, loc="left")
            if nombre != "Original":
                print(f"  {nombre:<28} no se puede decodificar")
            continue
        if nombre != "Original":
            n = min(len(ref), len(x))
            ruido = np.sum((ref[:n].astype(float) - x[:n].astype(float)) ** 2)
            if ruido == 0 and len(x) == len(ref):
                txt = "idéntico al original"
            else:
                txt = f"{10 * np.log10(np.sum(ref[:n].astype(float) ** 2) / max(ruido, 1e-9)):.1f} dB"
                if len(x) != len(ref):
                    txt += f"  (dura {len(x)} muestras en vez de {len(ref)}: se desfasa)"
            print(f"  {nombre:<28} {txt}")
        cubos = 1500
        n = len(x) // cubos * cubos
        m = x[:n].reshape(cubos, -1)
        t = np.linspace(0, len(x) / fs, cubos)
        eje.fill_between(t, m.min(axis=1), m.max(axis=1), color="black", linewidth=0)
        eje.set_title(nombre, fontsize=9, loc="left")
    ejes[-1].set_xlabel("tiempo (s)")
    fig.suptitle(f"Audio decodificado, rayón de {B} bytes", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(ruta_png, dpi=130)
    plt.close(fig)


def figura_video(ruta_original, versiones, B, ruta_png):
    # versiones: lista de (nombre, ruta). Se compara cuadro a cuadro (en gris y tamaño reducido).
    orig = medios.decodificar_video_pequeno(ruta_original)
    curvas = []
    for nombre, ruta in versiones:
        v = medios.decodificar_video_pequeno(ruta)
        if v is None:
            print(f"\nVideo '{nombre}': no se pudo decodificar.")
            curvas.append((nombre, ruta, None))
            continue
        n = min(len(orig), len(v))
        err = np.abs(orig[:n].astype(float) - v[:n].astype(float)).mean(axis=(1, 2))
        curvas.append((nombre, ruta, err))
        print(f"\nVideo '{nombre}': {len(v)} de {len(orig)} cuadros; "
              f"cuadros con error: {int(np.count_nonzero(err > 0))}; error medio máximo {err.max():.1f} niveles de gris")
    validas = [c for c in curvas if c[2] is not None]
    if not validas:
        return
    # cuadro más dañado entre todas las versiones
    peor = int(np.argmax(max((c[2] for c in validas), key=lambda e: e.max())))
    carpeta = "salida"
    paneles = [("Original", ruta_original)] + [(c[0], c[1]) for c in curvas if c[2] is not None]
    fig = plt.figure(figsize=(3.6 * len(paneles), 7))
    for i, (titulo, ruta) in enumerate(paneles):
        eje = fig.add_subplot(2, len(paneles), i + 1)
        tmp = f"{carpeta}/_cuadro.png"
        if medios.extraer_cuadro(ruta, peor, tmp):
            eje.imshow(Image.open(tmp))
            os.remove(tmp)
        eje.set_title(f"{titulo}\n(cuadro {peor})", fontsize=9)
        eje.axis("off")
    eje = fig.add_subplot(2, 1, 2)
    colores = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]
    for k, (nombre, ruta, err) in enumerate(validas):
        eje.plot(err, color=colores[k % 4], label=nombre)
    eje.set_xlabel("cuadro")
    eje.set_ylabel("error medio vs original")
    eje.legend()
    eje.grid(alpha=0.3)
    fig.suptitle(f"Video con un rayón de {B} bytes", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(ruta_png, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- programa principal
def leer_parametros():
    p = argparse.ArgumentParser(description="Grabar un archivo cualquiera, rayarlo y recuperarlo con CIRC")
    p.add_argument("archivo")
    p.add_argument("--rafaga", type=int, default=480, help="largo del rayón en bytes")
    p.add_argument("--posicion", type=float, default=40, help="dónde cae el rayón (0 a 100 %% del archivo)")
    p.add_argument("--D", type=int, default=4)
    p.add_argument("--t1", type=int, default=1, choices=[1, 2])
    p.add_argument("--semilla", type=int, default=1)
    p.add_argument("--rapido", action="store_true", help="omitir el caso CIRC sin intercalado")
    p.add_argument("--mostrar", action="store_true", help="abrir la gráfica al terminar")
    return p.parse_args()


def leer_archivo(ruta):
    with open(ruta, "rb") as f:
        contenido = np.frombuffer(f.read(), dtype=np.uint8)
    mb = len(contenido) / 1e6
    print(f"Archivo: {ruta}  ({len(contenido)} bytes = {mb:.2f} MB)")
    if mb > 30:
        print("  (es grande: Reed-Solomon en Python puro tarda ~3-5 s por MB; puede demorar)")
    return contenido


def grabar_y_rayar(a, contenido):
    # codifica (con D y, si no es --rapido, también con D=0) y aplica el mismo rayón a cada caso
    t0 = time.time()
    grabado_D = circ.codificar_circ(contenido, a.D)
    grabado_0 = None if a.rapido else circ.codificar_circ(contenido, 0)
    print(f"Disco codificado: {len(grabado_D)} bytes  (tasa 3/4) [{time.time() - t0:.1f} s]")

    B = a.rafaga
    garantizada, maxima = circ.rafaga_teorica(a.D, a.t1)
    print(f"\nRayón de {B} bytes seguidos al {a.posicion:.0f} % del archivo.")
    print(f"Teoría con D={a.D}: siempre se recupera hasta {garantizada} bytes, máximo {maxima}.")
    rx_sin = canal.rafaga(contenido, int(a.posicion / 100 * len(contenido)), B, a.semilla)
    rx_D = canal.rafaga(grabado_D, int(a.posicion / 100 * len(grabado_D)), B, a.semilla)
    rx_0 = None if a.rapido else canal.rafaga(grabado_0, int(a.posicion / 100 * len(grabado_0)), B, a.semilla)
    return rx_sin, rx_0, rx_D


def decodificar_casos(a, contenido, rx_0, rx_D):
    t0 = time.time()
    rec_D = circ.decodificar_circ(rx_D, len(contenido), a.D, a.t1)[0]
    rec_0 = None
    if not a.rapido:
        rec_0 = circ.decodificar_circ(rx_0, len(contenido), 0, a.t1)[0]
    print(f"Decodificación terminada [{time.time() - t0:.1f} s]")
    return rec_0, rec_D


def guardar_resultados(base, ext, contenido, rx_sin, rec_0, rec_D):
    guardar(base + "original" + ext, contenido)
    guardar(base + "sin_codigo" + ext, rx_sin)
    if rec_0 is not None:
        guardar(base + "circ_sin_intercalado" + ext, rec_0)
    guardar(base + "recuperado" + ext, rec_D)


def lista_de_casos(a, base, ext, rx_sin, rec_0, rec_D):
    # cada caso: (nombre, bytes, ruta del archivo guardado)
    casos = [("sin código", rx_sin, base + "sin_codigo" + ext)]
    if rec_0 is not None:
        casos.append(("CIRC sin intercalado (D=0)", rec_0, base + "circ_sin_intercalado" + ext))
    casos.append((f"CIRC con intercalado (D={a.D})", rec_D, base + "recuperado" + ext))
    return casos


def tipo_de_archivo(ext):
    if ext in EXT_IMAGEN:
        return "imagen"
    if ext in EXT_AUDIO:
        return "audio"
    if ext in EXT_VIDEO:
        return "video"
    return "otro"


def mostrar_metricas(tipo, usar_ffmpeg, contenido, casos, rec_D):
    print(f"\nResultados (tipo de archivo: {tipo}):")
    h0 = sha(contenido)
    for nombre, datos, ruta in casos:
        malos = canal.bytes_distintos(contenido, datos)
        igual = "SHA-256 IDÉNTICO" if sha(datos) == h0 else "SHA-256 distinto"
        extra = ""
        if usar_ffmpeg:
            extra = " | " + estado_ffmpeg(ruta)
        elif tipo == "imagen":
            extra = " | " + ("abre" if abrir_imagen(datos) is not None else "NO abre")
        print(f"  {nombre:<28} bytes malos: {malos:>7}  {igual}{extra}")
    if sha(rec_D) == h0:
        print("\n  -> el archivo recuperado con intercalado es IDÉNTICO al original, bit por bit.")


def hacer_figura(tipo, usar_ffmpeg, B, base, ext, contenido, casos):
    png = "salida/comparacion_archivo.png"
    if tipo == "imagen":
        figura_imagen(contenido, [(n, d) for n, d, r in casos], B, png)
        print("\nGuardado: " + png)
    elif tipo == "audio" and usar_ffmpeg:
        figura_audio([("Original", base + "original" + ext)] + [(n, r) for n, d, r in casos], B, png)
        print("Guardado: " + png)
    elif tipo == "video" and usar_ffmpeg:
        figura_video(base + "original" + ext, [(n, r) for n, d, r in casos], B, png)
        print("Guardado: " + png)


def main():
    a = leer_parametros()
    ext = os.path.splitext(a.archivo)[1].lower()
    contenido = leer_archivo(a.archivo)
    os.makedirs("salida", exist_ok=True)

    rx_sin, rx_0, rx_D = grabar_y_rayar(a, contenido)
    rec_0, rec_D = decodificar_casos(a, contenido, rx_0, rx_D)

    base = "salida/archivo_"
    guardar_resultados(base, ext, contenido, rx_sin, rec_0, rec_D)
    casos = lista_de_casos(a, base, ext, rx_sin, rec_0, rec_D)

    tipo = tipo_de_archivo(ext)
    usar_ffmpeg = tipo in ("audio", "video") and medios.hay_ffmpeg()
    mostrar_metricas(tipo, usar_ffmpeg, contenido, casos, rec_D)
    hacer_figura(tipo, usar_ffmpeg, a.rafaga, base, ext, contenido, casos)

    if a.mostrar and os.path.exists("salida/comparacion_archivo.png"):
        medios.abrir_archivo("salida/comparacion_archivo.png")
    print("Archivos en salida/: " + ", ".join(os.path.basename(r) for n, d, r in casos) + ", archivo_original" + ext)


if __name__ == "__main__":
    main()
