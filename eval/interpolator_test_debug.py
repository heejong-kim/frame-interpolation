# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""A test script for mid frame interpolation from two input frames.

Usage example:
 python3 -m frame_interpolation.eval.interpolator_test \
   --frame1 <filepath of the first frame> \
   --frame2 <filepath of the second frame> \
   --model_path <The filepath of the TF2 saved model to use>

The output is saved to <the directory of the input frames>/output_frame.png. If
`--output_frame` filepath is provided, it will be used instead.
"""
import os
from typing import Sequence

from eval import interpolator as interpolator_lib
from eval import util
from absl import app
from absl import flags
import numpy as np
import sys

# Controls TF_CCP log level.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Clear any previously defined flags (important in Jupyter)
flags.FLAGS.unparse_flags()
flags.FLAGS.__dict__['__parsed'] = False

# Define flags
_FRAME1 = flags.DEFINE_string('frame1', None, 'The filepath of the first input frame.', required=True)
_FRAME2 = flags.DEFINE_string('frame2', None, 'The filepath of the second input frame.', required=True)
_MODEL_PATH = flags.DEFINE_string('model_path', None, 'The path of the TF2 saved model to use.')
_OUTPUT_FRAME = flags.DEFINE_string('output_frame', None, 'The output filepath of the interpolated mid-frame.')
_ALIGN = flags.DEFINE_integer('align', 64, 'If >1, pad the input size so it is evenly divisible by this value.')
_BLOCK_HEIGHT = flags.DEFINE_integer('block_height', 1, 'Number of patches along height.')
_BLOCK_WIDTH = flags.DEFINE_integer('block_width', 1, 'Number of patches along width.')

# Provide the arguments
sys.argv = [
    'debug',
    '--frame1=photos/one.png',
    '--frame2=photos/two.png',
    '--model_path=pretrained_models/film_net/Style/saved_model',
    '--output_frame=photos/output_middle.png',
    '--align=64',
    '--block_height=1',
    '--block_width=1'
]

# Parse them
flags.FLAGS(sys.argv)


def _run_interpolator() -> None:
  """Writes interpolated mid frame from a given two input frame filepaths."""

  interpolator = interpolator_lib.Interpolator(
      model_path=_MODEL_PATH.value,
      align=_ALIGN.value,
      block_shape=[_BLOCK_HEIGHT.value, _BLOCK_WIDTH.value])

  # First batched image.
  image_1 = util.read_image(_FRAME1.value)
  image_batch_1 = np.expand_dims(image_1, axis=0)

  # Second batched image.
  image_2 = util.read_image(_FRAME2.value)
  image_batch_2 = np.expand_dims(image_2, axis=0)

  # Batched time.
  batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)

  # Invoke the model for one mid-frame interpolation.
  mid_frame = interpolator(image_batch_1, image_batch_2, batch_dt)[0]

  # Write interpolated mid-frame.
  mid_frame_filepath = _OUTPUT_FRAME.value
  if not mid_frame_filepath:
    mid_frame_filepath = f'{os.path.dirname(_FRAME1.value)}/output_frame.png'
  util.write_image(mid_frame_filepath, mid_frame)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  _run_interpolator()


if __name__ == '__main__':
    app.run(main)


    # debugging
    model = tf.keras.models.load_model(_MODEL_PATH.value)

    from eval.interpolator import image_to_patches
    # First batched image.
    image_1 = util.read_image(_FRAME1.value)
    image_batch_1 = np.expand_dims(image_1, axis=0)

    # Second batched image.
    image_2 = util.read_image(_FRAME2.value)
    image_batch_2 = np.expand_dims(image_2, axis=0)

    _block_shape = [_BLOCK_HEIGHT.value, _BLOCK_WIDTH.value]

    image_0 = image_to_patches(image_batch_1, _block_shape)
    image_1 = image_to_patches(image_batch_2, _block_shape)
    batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)

    inputs = {'x0': image_0, 'x1': image_1, 'time': batch_dt[..., np.newaxis]}
    result = model(inputs, training=False)
    image = result['image']

    for var in model.variables:
      print(f"{var.name} — {var.shape}")

    from tensorflow.keras import Model
    target_layer = model.get_layer('feat_net')
    feature_model = Model(inputs=model.inputs, outputs=target_layer.output)
    features = feature_model({'x0': image_0, 'x1': image_1, 'time': batch_dt[..., None]})

    '''
    input: (1, 768, 1024, 3)

    (1, 768, 1024, 64) # 3->64
    (1, 384, 512, 192) # 64 + (64 -> 128)
    (1, 192, 256, 448) # 
    (1, 96, 128, 960) # concatenated 64 + 128
    (1, 48, 64, 960) # concatenated 64 + 128
    (1, 24, 32, 960) # concatenated 64 + 128
    (1, 12, 16, 960) # concatenated 64 + 128

    feat_net/sub_extractor/cfeat_conv_0/kernel:0 — (3, 3, 3, 64)                                                                                  
    feat_net/sub_extractor/cfeat_conv_0/bias:0 — (64,)                     
                                                                           
    feat_net/sub_extractor/cfeat_conv_1/kernel:0 — (3, 3, 64, 64)
    feat_net/sub_extractor/cfeat_conv_1/bias:0 — (64,)
                       
    feat_net/sub_extractor/cfeat_conv_2/kernel:0 — (3, 3, 64, 128)
    feat_net/sub_extractor/cfeat_conv_2/bias:0 — (128,)                
    
    feat_net/sub_extractor/cfeat_conv_3/kernel:0 — (3, 3, 128, 128)
    feat_net/sub_extractor/cfeat_conv_3/bias:0 — (128,)
    
    feat_net/sub_extractor/cfeat_conv_4/kernel:0 — (3, 3, 128, 256)
    feat_net/sub_extractor/cfeat_conv_4/bias:0 — (256,)
    
    feat_net/sub_extractor/cfeat_conv_5/kernel:0 — (3, 3, 256, 256)
    feat_net/sub_extractor/cfeat_conv_5/bias:0 — (256,)
    
    feat_net/sub_extractor/cfeat_conv_6/kernel:0 — (3, 3, 256, 512)
    feat_net/sub_extractor/cfeat_conv_6/bias:0 — (512,)
    
    feat_net/sub_extractor/cfeat_conv_7/kernel:0 — (3, 3, 512, 512)
    feat_net/sub_extractor/cfeat_conv_7/bias:0 — (512,)

    '''