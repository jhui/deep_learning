from progress.bar import Bar
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class ProgressBar(Bar):
    message = 'Loading'
    fill = '#'
    suffix = '%(percent).1f%% | ETA: %(eta)ds'


def plot(samples):
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
           ax = plt.subplot(gs[i])
           plt.axis('off')
           ax.set_xticklabels([])
           ax.set_yticklabels([])
           ax.set_aspect('equal')
           plt.imshow(sample.reshape([28, 28]), cmap=plt.get_cmap('gray'))

        return fig