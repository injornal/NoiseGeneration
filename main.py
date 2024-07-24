import math

import numpy as np
import numpy.typing as npt
from PIL import Image


def mix(a0: float, a1: float, w: float) -> float:
    return (a1 - a0) * (3.0 - w * 2.0) * w * w + a0


def perlin_noise(w: int, size: int) -> npt.NDArray:
    image = np.zeros((w, w))
    gradients = np.array(
        [
            [np.array([math.sin(x), math.cos(x)]) for x in y]
            for y in np.random.uniform(0, 1, (size + 1, size + 1))
        ]
    )
    for row in range(w):
        for column in range(w):
            tl = np.array([(row % size), (column % size)]) / size
            tr = np.array([(row % size), (column % size) - size]) / size
            bl = np.array([(row % size - size), (column % size)]) / size
            br = np.array([(row % size - size), (column % size) - size]) / size

            dotTL = np.dot(tl, gradients[row // size][column // size])
            dotTR = np.dot(tr, gradients[row // size][column // size + 1])
            dotBL = np.dot(bl, gradients[row // size + 1][column // size])
            dotBR = np.dot(br, gradients[row // size + 1][column // size + 1])

            image[row][column] = mix(
                mix(dotTL, dotTR, column % size / size),
                mix(dotBL, dotBR, column % size / size),
                row % size / size,
            )

            # print(f"{dotTL=}\n{dotTR=}\n{dotBL=}\n{dotBR=}\n{image[row][column]=}\n")

    return image


if __name__ == "__main__":
    width = 1000
    size = 250
    # img = Inp.random.uniform(0, 1, (w // size + 1, w // size + 1, 2))
    img = Image.fromarray(
        np.uint8(
            np.minimum(
                np.maximum(
                    1 - abs(perlin_noise(width, size) * 3), np.zeros((width, width))
                ),
                np.ones((width, width)),
            )
            * 255
        )
    )
    img.show()
