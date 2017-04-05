from time import time

from wavenet.utils import make_batch
from wavenet.models import Model, Generator

from scipy.io import wavfile

inputs, targets = make_batch('assets/sample_1.wav')


num_time_samples = inputs.shape[1]
num_channels = 1
gpu_fraction = 1.0

model = Model(num_time_samples=num_time_samples,
              num_channels=num_channels,
              gpu_fraction=gpu_fraction)

tic = time()
model.train(inputs, targets,10)
toc = time()

print('Training took {} seconds.'.format(toc-tic))

generator = Generator(model)

# Get first sample of input
input_ = inputs[:, 0:1, 0]

tic = time()
predictions = generator.run(input_, inputs.shape[0])
toc = time()
print('Generating took {} seconds.'.format(toc-tic))

wavfile.write("output_pred.wav",16000,predictions.T)
