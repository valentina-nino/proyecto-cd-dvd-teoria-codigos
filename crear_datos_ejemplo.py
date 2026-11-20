# crear_datos_ejemplo.py
# Genera archivos de prueba en la carpeta datos/ (todo sintético, sin derechos de autor).
# Se puede cambiar por cualquier imagen, WAV, mp3, mp4... propios.
#
#   imagen_ejemplo.png   128x128 gris
#   imagen_color.png     512x512 color      (y imagen_color.jpg, comprimida)
#   audio_ejemplo.wav    8 kHz, 8 bits, mono
#   audio_alta.wav       44,1 kHz, 16 bits, estéreo   (y audio_ejemplo.mp3, comprimido)
#   video_ejemplo.mp4    video H.264 + audio AAC (solo si hay ffmpeg instalado)

import os
import shutil
import subprocess
import numpy as np
from PIL import Image, ImageDraw
import disco

os.makedirs("datos", exist_ok=True)

# ---------------------------------------------------------------- imagen gris 128 x 128
ancho, alto = 128, 128
img = Image.new("L", (ancho, alto), 0)
d = ImageDraw.Draw(img)
for y in range(alto):                      # fondo con degradado
    d.line([(0, y), (ancho, y)], fill=int(40 + 120 * y / alto))
d.ellipse([20, 20, 108, 108], outline=255, width=5)          # disco
d.ellipse([54, 54, 74, 74], fill=0, outline=255, width=3)     # hueco central
d.rectangle([8, 100, 60, 120], fill=230)                      # figuras
d.polygon([(90, 120), (120, 120), (105, 95)], fill=20)
d.text((70, 8), "CD", fill=255)
img.save("datos/imagen_ejemplo.png")

# ---------------------------------------------------------------- imagen a color 512 x 512
N = 512
x = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, x)
rgb = np.zeros((N, N, 3))
rgb[..., 0] = 0.2 + 0.8 * X                  # rojo crece hacia la derecha
rgb[..., 1] = 0.2 + 0.8 * Y                  # verde crece hacia abajo
rgb[..., 2] = 0.8 - 0.6 * X * Y              # azul
img = Image.fromarray(np.uint8(rgb * 255))
d = ImageDraw.Draw(img)
d.ellipse([90, 90, 420, 420], outline=(255, 255, 255), width=14)       # disco
d.ellipse([210, 210, 300, 300], fill=(15, 15, 40), outline=(255, 255, 255), width=8)
d.ellipse([60, 330, 200, 470], fill=(250, 210, 40))                    # sol
d.rectangle([330, 400, 470, 480], fill=(200, 30, 60))
d.polygon([(380, 120), (470, 120), (425, 40)], fill=(30, 160, 90))
img.save("datos/imagen_color.png")
img.save("datos/imagen_color.jpg", quality=85)

# ---------------------------------------------------------------- audio: una melodía sencilla
notas = [262, 262, 392, 392, 440, 440, 392, 349, 349, 330, 330, 294, 294, 262]  # Hz
duraciones = [0.4] * 6 + [0.8] + [0.4] * 6 + [0.8]


def melodia(fs):
    senal = []
    for f, dur in zip(notas, duraciones):
        t = np.arange(int(fs * dur)) / fs
        onda = np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * 2 * f * t)
        envolvente = np.minimum(1, np.minimum(t / 0.02, (dur - t) / 0.05))
        senal.append(onda * envolvente)
    senal = np.concatenate(senal)
    return senal / np.max(np.abs(senal))


# 8 kHz, 8 bits, mono
s = melodia(8000) * 100
audio = np.clip(np.round(s) + 128, 0, 255).astype(np.uint8)
disco.guardar_audio("datos/audio_ejemplo.wav", audio, 8000, 1, 8)

# 44,1 kHz, 16 bits, estéreo (la melodía se mueve un poco entre izquierda y derecha)
fs = 44100
s = melodia(fs)
t = np.arange(len(s)) / fs
pan = 0.5 + 0.4 * np.sin(2 * np.pi * 0.3 * t)
izq = s * (1 - pan) * 2 * 0.6
der = s * pan * 2 * 0.6
est = np.stack([izq, der], axis=1)
pcm = np.clip(np.round(est * 32767), -32768, 32767).astype("<i2")
disco.guardar_audio("datos/audio_alta.wav", np.frombuffer(pcm.tobytes(), dtype=np.uint8), fs, 2, 16)

# ---------------------------------------------------------------- mp3 y mp4 (necesitan ffmpeg)
if shutil.which("ffmpeg"):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", "datos/audio_alta.wav",
                    "-b:a", "128k", "datos/audio_ejemplo.mp3"], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error",
                    "-f", "lavfi", "-i", "testsrc2=size=480x270:rate=24:duration=6",
                    "-i", "datos/audio_alta.wav", "-shortest",
                    "-c:v", "libx264", "-crf", "30", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "96k", "datos/video_ejemplo.mp4"], check=True)
else:
    print("(no hay ffmpeg: no se crearon el mp3 ni el mp4; puedes usar tus propios archivos)")

for f in sorted(os.listdir("datos")):
    print(f"{f:<22} {os.path.getsize('datos/' + f):>9} bytes")
