# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import numpy as np

def hex_to_rgb(hex):
    y = tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))
    return (y[0]/255,y[1]/255,y[2]/255)

# Define colors for the demo
color = ['0047AB', # cobaltblue
        '6495ED', # cornerblue
        'FF9999', 'FF9933', '00CC66', '66B2FF', 'FF6666', 'FF3333', 'C0C0C0', '9933FF'] # rosé - orange - green - blue - red - grey - violet
color = [ hex_to_rgb(x) for x in color]

for i in range(200):
        color_i = list(np.random.choice(range(256), size=3))
        color.append((color_i[0]/225, color_i[1]/225, color_i[2]/225))

demo_color = color

