# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from IPython.display import SVG, display
from PIL import Image
from binascii import b2a_base64

PNGS_PATH = (Path(__file__).parent / '../CoDraw/Pngs').resolve()
EMBED_PNGS_PATH = '../../CoDraw/Pngs'
DEPTH_SCALE = [1.0, 0.7, 0.49]
IMAGE_NAMES = [
 's_0s.png',
 's_1s.png',
 's_2s.png',
 's_3s.png',
 's_4s.png',
 's_5s.png',
 's_6s.png',
 's_7s.png',
 'p_0s.png',
 'p_1s.png',
 'p_2s.png',
 'p_3s.png',
 'p_4s.png',
 'p_5s.png',
 'p_6s.png',
 'p_7s.png',
 'p_8s.png',
 'p_9s.png',
 'hb0_0s.png',
 'hb0_1s.png',
 'hb0_2s.png',
 'hb0_3s.png',
 'hb0_4s.png',
 'hb0_5s.png',
 'hb0_6s.png',
 'hb0_7s.png',
 'hb0_8s.png',
 'hb0_9s.png',
 'hb0_10s.png',
 'hb0_11s.png',
 'hb0_12s.png',
 'hb0_13s.png',
 'hb0_14s.png',
 'hb0_15s.png',
 'hb0_16s.png',
 'hb0_17s.png',
 'hb0_18s.png',
 'hb0_19s.png',
 'hb0_20s.png',
 'hb0_21s.png',
 'hb0_22s.png',
 'hb0_23s.png',
 'hb0_24s.png',
 'hb0_25s.png',
 'hb0_26s.png',
 'hb0_27s.png',
 'hb0_28s.png',
 'hb0_29s.png',
 'hb0_30s.png',
 'hb0_31s.png',
 'hb0_32s.png',
 'hb0_33s.png',
 'hb0_34s.png',
 'hb1_0s.png',
 'hb1_1s.png',
 'hb1_2s.png',
 'hb1_3s.png',
 'hb1_4s.png',
 'hb1_5s.png',
 'hb1_6s.png',
 'hb1_7s.png',
 'hb1_8s.png',
 'hb1_9s.png',
 'hb1_10s.png',
 'hb1_11s.png',
 'hb1_12s.png',
 'hb1_13s.png',
 'hb1_14s.png',
 'hb1_15s.png',
 'hb1_16s.png',
 'hb1_17s.png',
 'hb1_18s.png',
 'hb1_19s.png',
 'hb1_20s.png',
 'hb1_21s.png',
 'hb1_22s.png',
 'hb1_23s.png',
 'hb1_24s.png',
 'hb1_25s.png',
 'hb1_26s.png',
 'hb1_27s.png',
 'hb1_28s.png',
 'hb1_29s.png',
 'hb1_30s.png',
 'hb1_31s.png',
 'hb1_32s.png',
 'hb1_33s.png',
 'hb1_34s.png',
 'a_0s.png',
 'a_1s.png',
 'a_2s.png',
 'a_3s.png',
 'a_4s.png',
 'a_5s.png',
 'c_0s.png',
 'c_1s.png',
 'c_2s.png',
 'c_3s.png',
 'c_4s.png',
 'c_5s.png',
 'c_6s.png',
 'c_7s.png',
 'c_8s.png',
 'c_9s.png',
 'e_0s.png',
 'e_1s.png',
 'e_2s.png',
 'e_3s.png',
 'e_4s.png',
 'e_5s.png',
 'e_6s.png',
 't_0s.png',
 't_1s.png',
 't_2s.png',
 't_3s.png',
 't_4s.png',
 't_5s.png',
 't_6s.png',
 't_7s.png',
 't_8s.png',
 't_9s.png',
 't_10s.png',
 't_11s.png',
 't_12s.png',
 't_13s.png',
 't_14s.png',
 ]

def get_image_name(clipart):
    if clipart.idx < 18:
        return IMAGE_NAMES[clipart.idx]
    elif clipart.idx < 18 + 2:
        return IMAGE_NAMES[18 + (clipart.idx - 18) * 35 + clipart.subtype]
    else:
        return IMAGE_NAMES[clipart.idx + 34*2]


def snippet_from_clipart(clipart, inline_images=True):
    img_name = get_image_name(clipart)
    img_path = PNGS_PATH / img_name
    img_pil = Image.open(img_path)
    width, height = img_pil.width, img_pil.height
    if inline_images:
        data = b2a_base64(img_path.read_bytes()).decode('ascii')

    scale = DEPTH_SCALE[clipart.depth]
    width = width * scale
    height = height * scale

    flip = -1 if bool(clipart.flip) else 1
    x = clipart.x - width / 2.0
    y = clipart.y - height / 2.0

    flipped_sub_x = (-width) if clipart.flip else 0

    if inline_images:
        href = f"data:image/png;base64,{data}"
    else:
        href = f"{EMBED_PNGS_PATH}/{img_name}"

    return f"""
    <g transform="translate({x}, {y})">
        <image href="{href}" x="{flipped_sub_x}" y="0" width="{width}" height="{height}"
         transform="scale({flip}, 1)"/>
    </g>
    """

def svg_from_cliparts(cliparts, color=None, label=None, inline_images=True, scale=1.0):
    img_path = PNGS_PATH / 'background.png'
    if inline_images:
        data = b2a_base64(img_path.read_bytes()).decode('ascii')
        href = f"data:image/png;base64,{data}"
    else:
        href = f"{EMBED_PNGS_PATH}/background.png"
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{int(500*scale)}px" height="{int(400*scale)}px" viewBox="0 0 500 400">
        <image href="{href}" x="0" y="0" width="100%" height="100%"/>
    """
    if color:
        svg += f"""
        <rect fill="{color}" opacity="0.2" x="0" y="0" width="100%" height="100%"/>
    """

    # Sun (idx=3) is always in the back; this hack is also in Abs.js
    # All sky objects (idx < 8) are behind any non-sky objects
    # Past that, objects are sorted by depth and then by index
    for clipart in sorted(cliparts, key=lambda c: c.render_order_key):
        svg += snippet_from_clipart(clipart, inline_images=inline_images)

    if label:
        svg += f"""<text x="95%" y="8%" style="text-anchor: end">{label}</text>"""

    svg += "</svg>"
    return svg

def display_cliparts(cliparts, color=None, label=None, scale=1.0):
    display(SVG(svg_from_cliparts(cliparts, color, label, scale=scale)))
