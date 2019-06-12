import imageio
import sys

images = []
path = sys.argv[1]
filenames = [path + '/' + str(i) + '.png' for i in range(1000)]

imageio.plugins.freeimage.download()

for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave(path + '/movie.gif', images, format='GIF-FI', duration=0.001)
