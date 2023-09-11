import sys
import pathlib
import numpy as np
from PIL import Image, ImageOps

from eidynamics import pattern_index

'''
1. Import camera image
2. generate polygon frame grid
    2a. read polygon map file or make a calibration history file
    2b. get offset and scaling from 2a
3. Generate spot grid (all spots by default, or any sqSet)
4. overlay 3 over 2
5. overlay 4 over 1
6. save
'''

def main(brightfield_file, fluor_file='', fluor_channel='tdTomato', patternID_to_draw=999, draw_grid=True):
    brightfieldfile = pathlib.Path(brightfield_file)
    fileDir = brightfieldfile.parent
    outputfile = fileDir / "output_image.png"

    # bright field image
    bf_image = Image.open(brightfieldfile).copy().convert('RGBA')        

    # fluorescence image
    if fluor_file:
        fluorfile = pathlib.Path(fluor_file)
        flimage   = Image.open(fluorfile).copy()
        # flimage   = make_background_transparent(flimage)
        fl_image  = reduce_alpha(flimage,new_alpha=150)
        fl_image  = apply_channel_color(flimage,channel=fluor_channel)
        bf_composite_image = Image.blend(fl_image, bf_image, 0.3)
        bf_composite_image2 = bf_composite_image.copy()

    # make main canvas
    print(pattern_index.polygon_frame_properties)
    x0 = int(pattern_index.polygon_frame_properties['offsetx'])
    y0 = int(pattern_index.polygon_frame_properties['offsety'])
    w  = int(pattern_index.polygon_frame_properties['width m'  ])
    h  = int(pattern_index.polygon_frame_properties['height m' ])

    left   = abs( min (x0,                    0) )
    right  = abs( max (x0+w-bf_image.size[0], 0) )
    top    = abs( min (y0,                    0) )
    bottom = abs( max (y0+h-bf_image.size[1], 0) )
    background = expand_canvas(bf_composite_image,left, right, top, bottom)

    # polygon frame
    polygon_image = draw_polygon_frame(patternID=patternID_to_draw, grid_size=pattern_index.gridSize, draw_grid=draw_grid)
    polygon_image_mask = reduce_alpha(polygon_image)

    # make a composite
    # polygon_frame_mask = np.zeros(polygon_image.size)
    background.paste(polygon_image, (x0, y0), mask=polygon_image)
    
    background.save(outputfile)
    
    return background

def make_background_transparent(input_image,background_color=(0,0,0)):
        image = input_image.convert('RGBA')
        bgR,bgG,bgB = background_color
        # Transparency
        newImage = []
        for item in image.getdata():
            if item[:3] == background_color:
                new_color = (bgR,bgG,bgB,0)
                newImage.append(new_color)
            else:
                new_color = (item[0],item[1],item[2],255)
                newImage.append(new_color)

        image.putdata(newImage)

        return image

def reduce_alpha(input_image,new_alpha=128):
    image = input_image.copy().convert('RGBA')
    # Transparency
    newImage = []
    for item in image.getdata():
        new_color = (item[0], item[1], item[2], new_alpha)
        newImage.append(new_color)
    image.putdata(newImage)
    return image
        
def apply_channel_color(input_image,channel='tdTomato'):
    channel_color = get_channel_lut(channel)[:3]
    im1 = input_image.copy().convert('L')
    im2 = ImageOps.colorize(im1,black=(0,0,0),white=(255,255,255),mid=channel_color)
    im2 = im2.convert('RGBA')
    
    return im2

def get_channel_lut(channel):
    LUT = {'tdTomato':(255,174,127,255),
            'eYFP'   :(234,255,127,255),
            'GFP'    :(127,255,127,255),
            'IR'     :(200,200,200,255),
            'grey'   :(200,200,200,255)}
    return LUT[channel]

def draw_polygon_frame(patternID=999, grid_size=24, draw_grid=True):
    pixels        = np.zeros((grid_size*grid_size,1))
    sqSet         = pattern_index.patternID[patternID]
    pixels[sqSet] = 1
    frameArray    = np.reshape(pixels,(grid_size,grid_size))
    frameImage    = Image.fromarray(frameArray*255).convert('L')

    scaling       = pattern_index.polygon_frame_properties['scaling']
    frameLarge    = frameImage.resize(np.multiply(frameImage.size,scaling),0)
    polygon_image = make_background_transparent(frameLarge)

    # if draw_grid:
    #     grid = np.zeros(frameLarge.size).T
    #     grid[:,::scaling[0]] = 1
    #     grid[::scaling[1],:] = 1
    #     gridImage = Image.fromarray(grid*255).convert('RGBA')

        # polygon_frame_image = Image.blend(frameLarge,gridImage,0.3)
    # else:
        # polygon_frame_image = frameLarge

    # output_image = replace_color_with_transparency(polygon_frame_image)

    return polygon_image

def expand_canvas(input_image, left, right, top, bottom, color=(0,0,0)):
    width, height = input_image.size
    new_width = width + right + left
    new_height = height + top + bottom
    print(new_width,new_height)
    canvas = Image.new(input_image.mode, (new_width, new_height), color)
    canvas.paste(input_image, (left, top))
    return canvas

if __name__ == '__main__':
    import sys
    file1 = sys.argv[1]
    file2 = sys.argv[2]

    output_image = main(file1, fluor_file=file2)
    output_image.show()