import numpy as np
import gdsfactory as gf
from math import pi

gf.gpdk.PDK.activate()

# ---- Parameters (microns) ----
circle_diameter = 4.0
rect_width = 0.3
rect_length = 50.0

dr = 0.775
r0 = dr/2

GRID = 0.001  # 1 nm


def snap(v):
    return np.round(v / GRID) * GRID


# ---- Rotated element (polygon-based, grid-safe) ----
@gf.cell
def radial_element(angle_deg: float):
    c = gf.Component()

    theta = np.radians(angle_deg - 90)

    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ])

    # Rectangle
    rect_pts = np.array([
        [-rect_width/2, 0],
        [ rect_width/2, 0],
        [ rect_width/2, rect_length],
        [-rect_width/2, rect_length],
    ])
    rect_pts = snap(rect_pts @ R.T)
    #c.add_polygon(rect_pts, layer=(1, 0))

    # Circle (polygon approx)
    n_pts = 64
    angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    circle_pts = np.column_stack([
        (circle_diameter/2) * np.cos(angles),
        (circle_diameter/2) * np.sin(angles),
    ])
    circle_pts = snap(circle_pts @ R.T)
    c.add_polygon(circle_pts, layer=(1, 0))

    return c


# ---- Circular array with manual control ----
@gf.cell
def circular_array(n_per_layer, offset_per_layer):
    """
    n_per_layer: list[int]
    offset_per_layer: list[float] (degrees)
    """
    c = gf.Component()

    assert len(n_per_layer) == len(offset_per_layer), "Lists must match length"

    for i, (n, offset_deg) in enumerate(zip(n_per_layer, offset_per_layer)):
        r = snap(r0 + i * dr)

        for k in range(n):
            theta = 2 * pi * k / n + np.radians(offset_deg)
            theta = theta % (2 * pi)

            angle_deg = np.degrees(theta)

            x = snap(r * np.cos(theta))
            y = snap(r * np.sin(theta))

            elem = radial_element(angle_deg)
            ref = c.add_ref(elem)
            ref.dmove((x, y))

    return c


# ---- Example usage ----
if __name__ == "__main__":
    n_per_layer = [4,10,16,22,29,35,41,48,54,60,66,73,79,85,92,98,104,110,117,123,129,136,142,148,154,161,167,173,180,186,192,198]
    offset_per_layer = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]  # degrees

    c = circular_array(n_per_layer, offset_per_layer)
    c.write_gds("test.gds")