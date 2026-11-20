# medios.py
# Funciones AUXILIARES para revisar archivos de audio y video después de dañarlos y recuperarlos.
# No son parte del núcleo del código corrector: solo usan ffmpeg (si está instalado) para
# comprobar si el archivo todavía se puede decodificar.

import os
import sys
import shutil
import subprocess
import numpy as np


def hay_ffmpeg():
    return shutil.which("ffmpeg") is not None


def revisar_con_ffmpeg(ruta):
    # Decodifica el archivo completo. Devuelve (abre, mensajes_de_error).
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", ruta, "-f", "null", "-"],
                       capture_output=True, text=True, timeout=300)
    lineas = [l for l in r.stderr.splitlines() if l.strip()]
    return r.returncode == 0, len(lineas)


def decodificar_audio(ruta, fs=8000):
    # Devuelve el audio decodificado como muestras int16 mono, o None si no se pudo.
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", ruta, "-vn", "-ac", "1", "-ar", str(fs),
                        "-f", "s16le", "-"], capture_output=True, timeout=300)
    if len(r.stdout) < 2:
        return None
    n = len(r.stdout) // 2 * 2
    return np.frombuffer(r.stdout[:n], dtype="<i2")


def decodificar_video_pequeno(ruta, ancho=96, alto=54):
    # Devuelve los cuadros del video en gris y tamaño reducido: array (cuadros, alto, ancho).
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", ruta, "-an", "-vf", f"scale={ancho}:{alto}",
                        "-pix_fmt", "gray", "-f", "rawvideo", "-"], capture_output=True, timeout=600)
    n = len(r.stdout) // (ancho * alto)
    if n == 0:
        return None
    return np.frombuffer(r.stdout[:n * ancho * alto], dtype=np.uint8).reshape(n, alto, ancho)


def extraer_cuadro(ruta, indice, salida_png):
    # Guarda el cuadro número 'indice' del video como imagen. Devuelve True si lo logró.
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", ruta, "-vf", f"select=eq(n\\,{indice})",
                        "-frames:v", "1", "-fps_mode", "passthrough", salida_png],
                       capture_output=True, timeout=300)
    return r.returncode == 0


def abrir_archivo(ruta):
    # Abre un archivo (por ejemplo una gráfica .png) con el programa predeterminado del sistema.
    # Si no hay pantalla o no se puede, no pasa nada: el archivo igual queda guardado.
    try:
        if sys.platform.startswith("win"):
            os.startfile(os.path.abspath(ruta))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", ruta])
        elif shutil.which("xdg-open"):
            subprocess.Popen(["xdg-open", ruta], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
