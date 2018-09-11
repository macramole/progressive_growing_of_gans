#!/usr/bin/env python
#%%
import pickle
import numpy as np
import tensorflow as tf
import PIL.Image
import scipy

SNAPSHOT_NUM = "006008"

#%%

# Initialize TensorFlow session.
tf.InteractiveSession()

# Import official CelebA-HQ networks.
with open('/media/macramole/stuff/Data/pgan/network-snapshot-' + SNAPSHOT_NUM + '.pkl', 'rb') as file:
    G, D, Gs = pickle.load(file)

#%%

# Generate latent vectors.
latents = np.random.RandomState(1000).randn(1000, *Gs.input_shapes[0][1:]) # 1000 random latents
latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]] # hand-picked top-10

# Generate dummy labels (not used by the official networks).
labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

# Run the generator to produce a set of images.
images = Gs.run(latents, labels)

# Convert images to PIL-compatible format.
images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
images = images.transpose(0, 2, 3, 1) # NCHW => NHWC

# Save images as PNG.
for idx in range(images.shape[0]):
    PIL.Image.fromarray(images[idx], 'RGB').save('/media/macramole/stuff/Data/pgan/' + SNAPSHOT_NUM + '/img%d.png' % idx)

#%% generate_interpolation_video
    
#    def (run_id, snapshot=None, grid_size=[1,1], image_shrink=1, image_zoom=1, duration_sec=60.0, smoothing_sec=1.0, mp4=None, mp4_fps=30, mp4_codec='libx265', mp4_bitrate='16M', random_seed=1000, minibatch_size=8):
#network_pkl = misc.locate_network_pkl(run_id, snapshot)

minibatch_size=8
smoothing_sec=1.0
grid_size=[1,1]
random_seed=1000
duration_sec=10.0
mp4_fps=30
mp4_bitrate='16M'
image_shrink=1

mp4 = '/media/macramole/stuff/Data/pgan/' + SNAPSHOT_NUM + '/video.mp4'
num_frames = int(np.rint(duration_sec * mp4_fps))
random_state = np.random.RandomState(random_seed)

print('Generating latent vectors...')
shape = [num_frames, np.prod(grid_size)] + Gs.input_shape[1:] # [frame, image, channel, component]
all_latents = random_state.randn(*shape).astype(np.float32)
all_latents = scipy.ndimage.gaussian_filter(all_latents, [smoothing_sec * mp4_fps] + [0] * len(Gs.input_shape), mode='wrap')
all_latents /= np.sqrt(np.mean(np.square(all_latents)))

def create_image_grid(images, grid_size=None):
    assert images.ndim == 3 or images.ndim == 4
    num, img_w, img_h = images.shape[0], images.shape[-1], images.shape[-2]

    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    grid = np.zeros(list(images.shape[1:-2]) + [grid_h * img_h, grid_w * img_w], dtype=images.dtype)
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[..., y : y + img_h, x : x + img_w] = images[idx]
    return grid

# Frame generation func for moviepy.
def make_frame(t):
    frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
    latents = all_latents[frame_idx]
    labels = np.zeros([latents.shape[0], 0], np.float32)
    images = Gs.run(latents, labels, minibatch_size=minibatch_size, num_gpus=1, out_mul=127.5, out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8)
    grid = create_image_grid(images, grid_size).transpose(1, 2, 0) # HWC
    
    if grid.shape[2] == 1:
        grid = grid.repeat(3, 2) # grayscale => RGB
    return grid

# Generate video.
import moviepy.editor # pip install moviepy
#result_subdir = misc.create_result_subdir(config.result_dir, config.desc)
moviepy.editor.VideoClip(make_frame, duration=duration_sec).write_videofile(mp4, fps=mp4_fps, codec='libx264', bitrate=mp4_bitrate)
#open(os.path.join(result_subdir, '_done.txt'), 'wt').close()