# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

AVAILABLE_ENCODERS = ['burgess', 'chen_mlp', 'locatello', 'montero_large', 'montero_small', 'lie_montero_small', 'sys_montero_small', 'twoview_masked_encoder']

def select(name):
    if name not in AVAILABLE_ENCODERS:
        raise ValueError(
            'No encoder [{name}] available. Please choose from {AVAILABLE_ENCODERS}.')
    
    if name == 'chen_mlp':
        from .chen_mlp import Encoder
    if name == 'burgess':
        from .burgess import Encoder
    if name == 'locatello':
        from .locatello import Encoder
    if name == 'montero_small':
        from .montero_small import Encoder
    if name == 'montero_large':
        from .montero_large import Encoder
    if name == 'lie_montero_small':
        from .lie_montero_small import Encoder
    if name == 'sys_montero_small':
        from .sys_montero_small import Encoder
    if name == 'twoview_masked_encoder':
        from .twoview_masked_encoder import Encoder
    if name == 'ibasaw_small':
        from .ibasaw_small import Encoder

    return Encoder
