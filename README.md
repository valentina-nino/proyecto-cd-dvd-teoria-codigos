# Proyecto final – Teoría de Códigos (MAT 4215)
**Tema 4: CD y DVD – Reed-Solomon con intercalado (CIRC)**

**Integrantes:** Santiago Barrios Guerrero · Jorge Duarte Martinez · Valentina Niño Solano

## Qué hace el aplicativo
Simula un disco, lo "raya" (bytes consecutivos dañados) y lo recupera con un mini-CIRC: C2 = RS(28,24), intercalado con retardos de D tramas y C1 = RS(32,28), sobre GF(256), d = 5 (corrigen 2 errores o 4 borrados). Compara el mismo rayón **sin código**, con **CIRC sin intercalado (D=0)** y con **CIRC con intercalado**. Funciona con imagen (gris o color), audio WAV y **cualquier archivo** (mp3, mp4, jpg...). Las 4 partes: codificador `circ.codificar_circ`, canal `canal.py`, decodificador `circ.decodificar_circ`, métrica en `demo.py` y `experimentos.py`.

## Cómo ejecutarlo (Python 3 con `pip install -r requirements.txt`)
```
python crear_datos_ejemplo.py          # una vez: crea los archivos de prueba en datos/
python demo.py                         # menú para la demostración en vivo
python demo.py --modo imagen --barrido # además calcula el rayón máximo recuperado
python demo_archivo.py datos/audio_ejemplo.mp3 --rafaga 480     # cualquier archivo
python hamming_intercalado.py          # intercalado con Hamming (7,4)
python pruebas.py                      # pruebas del núcleo
python experimentos.py                 # gráficas y tablas en salida/ (~1,5 min)
```
Archivos: `gf256.py`, `reed_solomon.py`, `circ.py` (núcleo propio), `canal.py`, `disco.py`, `medios.py`, `demo.py`, `demo_archivo.py`. ffmpeg es opcional (solo revisa mp3/mp4).

## Resultados principales
- **Teoría:** C2 corrige 4 borrados y los retardos reparten una ráfaga de m tramas en ⌈m/D⌉ borrados por palabra, entonces m ≤ 4D. Con C1 corrigiendo 1 error, siempre se recupera una ráfaga de **128D − 29 bytes** (D=4: 483; D=8: 995) y a lo más 128D + 2.
- **Medido (40 rayones al azar por punto):** 100 % de recuperación hasta ese largo. **Sin intercalado (D=0) solo sobrevive 1 byte.** 483 bytes = 3864 bits, del orden de los ~4000 bits (2,5 mm) que citan las fuentes para el CD real.
- **Errores aislados** (p = probabilidad por byte): con p = 0,02 la tasa de error pasa de 2 % a 0 %; con p = 0,05, de 5,0 % a 2,1 %.
- **Formatos comprimidos** (rayón de 480 bytes): una imagen sin comprimir pierde 160 píxeles, el mismo dibujo en JPG pierde el 61 %, un mp4 queda con 82 de 144 cuadros corruptos. Con CIRC (D=4) los tres quedan idénticos (SHA-256 igual).
- C1 corrige 1 error por defecto (con 2 hay falsas correcciones con probabilidad ≈ 0,75 %; con 1, ≈ 0,0002 %).

## Herramientas utilizadas
- **Claude (IA de Anthropic):** ayudó a escribir y depurar el código del aplicativo, a preparar los experimentos y a redactar este README.
- **`reedsolo`:** solo durante el desarrollo, para comprobar que nuestro codificador da las mismas palabras de código; no se usa en el aplicativo.
- **Auxiliares:** numpy, Pillow, matplotlib, `wave` y ffmpeg. GF(256), Reed-Solomon y CIRC son propios.

## Fuentes
Immink (1999), "Reed–Solomon Codes and the Compact Disc", en Wicker y Bhargava (eds.), *Reed-Solomon Codes and Their Applications*, Wiley · Wikipedia, "Cross-interleaved Reed–Solomon coding" · Notas ELEC 405/511, Univ. of Victoria, https://ece.uvic.ca/~agullive/reedsolomon405-511-2016.pdf · Stereophile, "Analog Compact Disc", p. 5 · Material del curso (Gómez, U. del Norte).
