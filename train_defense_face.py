## train_defense.py
##
## Copyright (C) 2017, Dongyu Meng <zbshfmmm@gmail.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

from setup_face import FACE
from defensive_models import DenoisingAutoEncoder as DAE

poolings = ["average", "max"]

shape = [32, 32, 1]
combination_I = [3, "average", 3]
combination_II = [3]
activation = "sigmoid"
reg_strength = 1e-9
epochs = 10000

data = FACE()

AE_I = DAE(shape, combination_I, v_noise=0.1, activation=activation,
           reg_strength=reg_strength)
AE_I.train(data, "FACEADV_I", num_epochs=epochs)

AE_II = DAE(shape, combination_II, v_noise=0.1, activation=activation,
            reg_strength=reg_strength)
AE_II.train(data, "FACEADV_II", num_epochs=epochs)

