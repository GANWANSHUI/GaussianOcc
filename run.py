# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import torch
import numpy as np
from runner import Runer
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True



if __name__ == "__main__":
    setup_seed(42)
    trainer = Runer(opts)
    trainer.train()
