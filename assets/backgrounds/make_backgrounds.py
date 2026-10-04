"""Generates two simple 1280x720 background PNGs using only the standard library.

Run from this folder:  python make_backgrounds.py
"""
import struct, zlib

W, H = 1280, 720

def png(path, pixel):
    raw = bytearray()
    for y in range(H):
        raw.append(0)
        for x in range(W):
            raw += bytes(pixel(x, y))
    def chunk(t, d):
        c = t + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
    data += chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b"")
    open(path, "wb").write(data)

def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

# 1) Blue-to-teal diagonal gradient.
png("gradient_blue.png",
    lambda x, y: lerp((30, 60, 140), (40, 170, 160), (x / W + y / H) / 2))

# 2) Simple "room": warm wall, darker floor, a light window rectangle.
def room(x, y):
    if 160 < x < 520 and 100 < y < 380:
        return lerp((200, 225, 245), (150, 190, 230), (y - 100) / 280)  # window
    if y > 520:
        return lerp((120, 85, 60), (80, 55, 40), (y - 520) / 200)       # floor
    return lerp((235, 220, 195), (210, 190, 160), y / 520)               # wall
png("simple_room.png", room)
