"""This script is used to generate the color palette table in the API documentation
for `draw.poses` node.
"""

from typing import Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import ImageColor

COLOR_MAP = {
    "champagne*": (156, 223, 244),
    "blizzard*": (241, 232, 164),
    "violet blue*": (188, 118, 119),
    "tomato*": (77, 103, 255),
    "pkdbrown*": (96, 109, 167),
    "aliceblue": (255, 248, 240),
    "antiquewhite": (215, 235, 250),
    "aqua": (255, 255, 0),
    "aquamarine": (212, 255, 127),
    "azure": (255, 255, 240),
    "beige": (220, 245, 245),
    "bisque": (196, 228, 255),
    "black": (0, 0, 0),
    "blanchedalmond": (205, 235, 255),
    "blue": (255, 0, 0),
    "blueviolet": (226, 43, 138),
    "brown": (42, 42, 165),
    "burlywood": (135, 184, 222),
    "cadetblue": (160, 158, 95),
    "chartreuse": (0, 255, 127),
    "chocolate": (30, 105, 210),
    "coral": (80, 127, 255),
    "cornflowerblue": (237, 149, 100),
    "cornsilk": (220, 248, 255),
    "crimson": (60, 20, 220),
    "cyan": (255, 255, 0),
    "darkblue": (139, 0, 0),
    "darkcyan": (139, 139, 0),
    "darkgoldenrod": (11, 134, 184),
    "darkgray": (169, 169, 169),
    "darkgreen": (0, 100, 0),
    "darkgrey": (169, 169, 169),
    "darkkhaki": (107, 183, 189),
    "darkmagenta": (139, 0, 139),
    "darkolivegreen": (47, 107, 85),
    "darkorange": (0, 140, 255),
    "darkorchid": (204, 50, 153),
    "darkred": (0, 0, 139),
    "darksalmon": (122, 150, 233),
    "darkseagreen": (143, 188, 143),
    "darkslateblue": (139, 61, 72),
    "darkslategray": (79, 79, 47),
    "darkslategrey": (79, 79, 47),
    "darkturquoise": (209, 206, 0),
    "darkviolet": (211, 0, 148),
    "deeppink": (147, 20, 255),
    "deepskyblue": (255, 191, 0),
    "dimgray": (105, 105, 105),
    "dimgrey": (105, 105, 105),
    "dodgerblue": (255, 144, 30),
    "firebrick": (34, 34, 178),
    "floralwhite": (240, 250, 255),
    "forestgreen": (34, 139, 34),
    "fuchsia": (255, 0, 255),
    "gainsboro": (220, 220, 220),
    "ghostwhite": (255, 248, 248),
    "gold": (0, 215, 255),
    "goldenrod": (32, 165, 218),
    "gray": (128, 128, 128),
    "green": (0, 128, 0),
    "greenyellow": (47, 255, 173),
    "grey": (128, 128, 128),
    "honeydew": (240, 255, 240),
    "hotpink": (180, 105, 255),
    "indianred": (92, 92, 205),
    "indigo": (130, 0, 75),
    "ivory": (240, 255, 255),
    "khaki": (140, 230, 240),
    "lavender": (250, 230, 230),
    "lavenderblush": (245, 240, 255),
    "lawngreen": (0, 252, 124),
    "lemonchiffon": (205, 250, 255),
    "lightblue": (230, 216, 173),
    "lightcoral": (128, 128, 240),
    "lightcyan": (255, 255, 224),
    "lightgoldenrodyellow": (210, 250, 250),
    "lightgray": (211, 211, 211),
    "lightgreen": (144, 238, 144),
    "lightgrey": (211, 211, 211),
    "lightpink": (193, 182, 255),
    "lightsalmon": (122, 160, 255),
    "lightseagreen": (170, 178, 32),
    "lightskyblue": (250, 206, 135),
    "lightslategray": (153, 136, 119),
    "lightslategrey": (153, 136, 119),
    "lightsteelblue": (222, 196, 176),
    "lightyellow": (224, 255, 255),
    "lime": (0, 255, 0),
    "limegreen": (50, 205, 50),
    "linen": (230, 240, 250),
    "magenta": (255, 0, 255),
    "maroon": (0, 0, 128),
    "mediumaquamarine": (170, 205, 102),
    "mediumblue": (205, 0, 0),
    "mediumorchid": (211, 85, 186),
    "mediumpurple": (219, 112, 147),
    "mediumseagreen": (113, 179, 60),
    "mediumslateblue": (238, 104, 123),
    "mediumspringgreen": (154, 250, 0),
    "mediumturquoise": (204, 209, 72),
    "mediumvioletred": (133, 21, 199),
    "midnightblue": (112, 25, 25),
    "mintcream": (250, 255, 245),
    "mistyrose": (225, 228, 255),
    "moccasin": (181, 228, 255),
    "navajowhite": (173, 222, 255),
    "navy": (128, 0, 0),
    "oldlace": (230, 245, 253),
    "olive": (0, 128, 128),
    "olivedrab": (35, 142, 107),
    "orange": (0, 165, 255),
    "orangered": (0, 69, 255),
    "orchid": (214, 112, 218),
    "palegoldenrod": (170, 232, 238),
    "palegreen": (152, 251, 152),
    "paleturquoise": (238, 238, 175),
    "palevioletred": (147, 112, 219),
    "papayawhip": (213, 239, 255),
    "peachpuff": (185, 218, 255),
    "peru": (63, 133, 205),
    "pink": (203, 192, 255),
    "plum": (221, 160, 221),
    "powderblue": (230, 224, 176),
    "purple": (128, 0, 128),
    "rebeccapurple": (153, 51, 102),
    "red": (0, 0, 255),
    "rosybrown": (143, 143, 188),
    "royalblue": (225, 105, 65),
    "saddlebrown": (19, 69, 139),
    "salmon": (114, 128, 250),
    "sandybrown": (96, 164, 244),
    "seagreen": (87, 139, 46),
    "seashell": (238, 245, 255),
    "sienna": (45, 82, 160),
    "silver": (192, 192, 192),
    "skyblue": (235, 206, 135),
    "slateblue": (205, 90, 106),
    "slategray": (144, 128, 112),
    "slategrey": (144, 128, 112),
    "snow": (250, 250, 255),
    "springgreen": (127, 255, 0),
    "steelblue": (180, 130, 70),
    "tan": (140, 180, 210),
    "teal": (128, 128, 0),
    "thistle": (216, 191, 216),
    # "tomato": (71, 99, 255),
    "turquoise": (208, 224, 64),
    "violet": (238, 130, 238),
    "wheat": (179, 222, 245),
    "white": (255, 255, 255),
    "whitesmoke": (245, 245, 245),
    "yellow": (0, 255, 255),
    "yellowgreen": (50, 205, 154),
}


def hex2rgb(hex: str, rgb: bool = True) -> Tuple[int, ...]:
    """Convert hex color to rgb/bgr color.

    Reference:
        https://pillow.readthedocs.io/en/stable/reference/ImageColor.html
    """
    if rgb:
        return ImageColor.getrgb(hex)
    return ImageColor.getrgb(hex)[::-1]


def plot_colortable(colors, sort_colors=True, emptycols=0):
    """Plots a color table.

    Reference:
        Code taken from https://matplotlib.org/stable/gallery/color/named_colors.html
    """
    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        by_hsv = sorted(
            (tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))), name)
            for name, color in colors.items()
        )

        names = [name for hsv, name in by_hsv]
    else:
        names = list(colors)

    n = len(names)
    ncols = 4 - emptycols
    nrows = n // ncols + int(n % ncols > 0)

    width = cell_width * 4 + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(
        margin / width,
        margin / height,
        (width - margin) / width,
        (height - margin) / height,
    )
    ax.set_xlim(0, cell_width * 4)
    ax.set_ylim(cell_height * (nrows - 0.5), -cell_height / 2.0)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(
            text_pos_x,
            y,
            name,
            fontsize=14,
            horizontalalignment="left",
            verticalalignment="center",
        )
        ax.add_patch(
            Rectangle(
                xy=(swatch_start_x, y - 9),
                width=swatch_width,
                height=18,
                facecolor=colors[name],
                edgecolor="0.7",
            )
        )

    return fig


def rgb2hex(rgb: Tuple[int, int, int]) -> str:
    """Convert rgb color to hex color."""
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


if __name__ == "__main__":
    # step 1: take colors from matplotlib
    # css colors superset of base colors
    # {"key": value} = {"aliceblue": "#F0F8FF", ...}
    css4_colors = mcolors.CSS4_COLORS
    css4_colors_hex2rgb = {
        color_name: hex2rgb(color_hex, rgb=False)
        for color_name, color_hex in css4_colors.items()
    }
    # after this step, add in custom colors to css4_colors_hex2rgb
    # this forms our COLOR_MAP.

    # step 2: convert rgb to hex as matplotlib expects it
    COLOR_MAP = {k: rgb2hex(v[::-1]) for k, v in COLOR_MAP.items()}
    plot_colortable(COLOR_MAP, sort_colors=True)
    plt.show()
