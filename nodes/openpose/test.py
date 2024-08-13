from . import OpenposeDetector
obj = OpenposeDetector()
obj.load_model()

import numpy
from PIL import Image

imarray = numpy.random.rand(1024,1024,3) * 255
image = numpy.asarray(Image.fromarray(imarray.astype('uint8')).convert('RGB'))

print(obj(image).shape)