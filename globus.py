#!/usr/bin/env python3
"""Zeichnet einen Globus (orthografische Ansicht, ganze Erdkugel sichtbar), zentriert auf den Ort des Tages.
Aufruf:  python3 globus.py [news.json] [ort-globus.png]
Liest ort.lat / ort.lon aus news.json und schreibt ein PNG (512 × 512, transparenter Hintergrund).
Benötigt nur numpy und Pillow sowie land-maske.png (Natural Earth 110m, gemeinfrei) im selben Ordner."""
import json, sys, os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HIER = os.path.dirname(os.path.abspath(__file__))
quelle = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HIER, "news.json")
ziel = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HIER, "ort-globus.png")
ort = json.load(open(quelle, encoding="utf-8"))["ort"]
lat0, lon0 = np.radians(float(ort["lat"])), np.radians(float(ort["lon"]))

N, S = 512, 2                      # Bildgröße, Überabtastung für weiche Kanten
G = N * S; R = G * 0.46; C = G / 2
maske = np.asarray(Image.open(os.path.join(HIER, "land-maske.png")).convert("L"), dtype=np.float32) / 255
MH, MW = maske.shape

# Farben laut Designrichtlinie (Hellblau = Wasser, Sand = Land, Weiß-Gitter, Pastellrot = Ort)
MEER = np.array([36, 62, 78]); MEER_HELL = np.array([70, 120, 145])
LAND = np.array([200, 170, 128]); LAND_HELL = np.array([228, 205, 168])

y, x = np.mgrid[0:G, 0:G].astype(np.float32)
px, py = (x - C) / R, (C - y) / R
rho = np.sqrt(px ** 2 + py ** 2); innen = rho <= 1
c = np.arcsin(np.clip(rho, 0, 1))
with np.errstate(invalid="ignore", divide="ignore"):
    lat = np.arcsin(np.cos(c) * np.sin(lat0) + np.where(rho > 0, py * np.sin(c) * np.cos(lat0) / rho, 0))
    lon = lon0 + np.arctan2(px * np.sin(c), rho * np.cos(c) * np.cos(lat0) - py * np.sin(c) * np.sin(lat0))
lon = (lon + np.pi) % (2 * np.pi) - np.pi
mx = ((lon + np.pi) / (2 * np.pi) * MW).astype(int).clip(0, MW - 1)
my = ((np.pi / 2 - lat) / np.pi * MH).astype(int).clip(0, MH - 1)
land = maske[my, mx][..., None]

# Beleuchtung: Licht von links oben, Rand dunkler (Kugeleindruck)
pz = np.sqrt(np.clip(1 - px ** 2 - py ** 2, 0, 1))
licht = np.clip(0.55 + 0.45 * (-0.45 * px + 0.5 * py + 0.74 * pz), 0, 1)[..., None]
farbe = (MEER + (MEER_HELL - MEER) * licht) * (1 - land) + (LAND + (LAND_HELL - LAND) * licht) * land

# Gradnetz alle 30°
deg_lat, deg_lon = np.degrees(lat), np.degrees(lon)
breite = 0.35 + 0.0 * rho
netz = ((np.abs(((deg_lat + 15) % 30) - 15) < breite) | (np.abs(((deg_lon + 15) % 30) - 15) < breite / np.maximum(np.cos(lat), 0.2)))
farbe = np.where(netz[..., None], farbe * 0.82 + 255 * 0.18, farbe)

rgba = np.zeros((G, G, 4), dtype=np.uint8)
rgba[..., :3] = np.clip(farbe, 0, 255).astype(np.uint8)
rgba[..., 3] = (innen * 255).astype(np.uint8)
bild = Image.fromarray(rgba, "RGBA")

# Atmosphärenring
ring = Image.new("RGBA", (G, G), (0, 0, 0, 0)); d = ImageDraw.Draw(ring)
d.ellipse([C - R - 3 * S, C - R - 3 * S, C + R + 3 * S, C + R + 3 * S], outline=(120, 210, 240, 150), width=4 * S)
ring = ring.filter(ImageFilter.GaussianBlur(3 * S))
bild = Image.alpha_composite(ring, bild)

# Markierung des Ortes (Mitte des Globus)
for r_, f_ in [(24, (242, 139, 130, 90)), (12, (255, 255, 255, 255)), (8, (242, 139, 130, 255))]:
    lage = Image.new("RGBA", (G, G), (0, 0, 0, 0))
    ImageDraw.Draw(lage).ellipse([C - r_ * S, C - r_ * S, C + r_ * S, C + r_ * S], fill=f_)
    bild = Image.alpha_composite(bild, lage)

bild.resize((N, N), Image.LANCZOS).save(ziel, optimize=True)
print("geschrieben:", ziel, ort.get("name"))
