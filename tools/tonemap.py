from os import environ

environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2 as cv
import numpy as np


def tonemapping_aces(x):
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return x * (a * x + b) / (x * (c * x + d) + e)


def tonemapping_uncharted2(x):
    def F(x):
        A = 0.22
        B = 0.30
        C = 0.10
        D = 0.20
        E = 0.01
        F = 0.30
        return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
    WHITE = 11.2
    return F(1.6 * x) / F(WHITE)


def hdr2ldr(image, tonemap):
    if tonemap == "aces":
        image = tonemapping_aces(image)
    elif tonemap == "uncharted2":
        image = tonemapping_uncharted2(image)
    elif tonemap is not None:
        print("Unknown tonemap method:", tonemap)
    return np.uint8(np.round(np.clip(np.where(
        image <= 0.00304,
        12.92 * image,
        1.055 * np.power(image, 1.0 / 2.4) - 0.055
    ) * 255, 0, 255)))


def read_image(path: str, exposure=0, tonemap=True):
    image = cv.imread(path, cv.IMREAD_COLOR | cv.IMREAD_ANYDEPTH)
    assert image is not None and image.dtype == np.float32
    return hdr2ldr(np.maximum(image[..., :3] * (2 ** exposure), 0.0), tonemap)

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    assert path.endswith(".exr")
    op = sys.argv[2] if len(sys.argv) > 2 else None
    exposure = float(sys.argv[3]) if len(sys.argv) > 3 else 0.
    cv.imwrite(f"{path[:-4]}.png", read_image(path, exposure, op))
