#!/usr/bin/env python

# @author: Giulio Isacchini
import logging
import os
from typing import *

#logging.getLogger("tensorflow").disabled = True
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["KERAS_BACKEND"] = "torch" # use torch backend

import keras
import keras.ops as ko
import numpy as np
from keras.losses import BinaryCrossentropy
from numpy.typing import NDArray

from sonnia.sonia import GENE_FEATURE_OPTIONS, Sonia


class SoNNia(Sonia):
    def __init__(
        self,
        *args: Tuple[Any],
        gene_features: str = "joint_vj",
        include_aminoacids: bool = True,
        deep: bool = True,
        **kwargs: Dict[str, Any],
    ) -> None:
        invalid_gene_features = {"vjl", "none"}
        if gene_features in invalid_gene_features:
            valid_gene_features = f"{GENE_FEATURE_OPTIONS - invalid_gene_features}"[
                1:-1
            ]
            invalid_gene_features = f"{invalid_gene_features}"[1:-1]
            raise ValueError(
                f"gene_features = '{gene_features}' is an invalid option "
                "when using a SoNNia model. Use one of the following "
                f"instead: {valid_gene_features}."
            )

        if not include_aminoacids:
            raise ValueError("include_aminoacids must be True for a SoNNia model.")

        self.deep = deep
        Sonia.__init__(self, *args, gene_features=gene_features, **kwargs)

    def update_model_structure(
        self, output_layer: List = [], input_layer: List = [], initialize: bool = False
    ) -> bool:
        """Defines the model structure and compiles it.

        Parameters
        ----------
        structure : Sequential Model Keras
            structure of the model
        initialize: bool
            if True, it initializes to linear model, otherwise it updates to new structure

        """
        if len(self.features) > 1:
            initial = np.array([s[0][0] for s in self.features])
        else:
            initial = np.array(["c", "c", "c"])
        self.l_length = np.count_nonzero(initial == "l")
        self.a_length = np.count_nonzero(initial == "a")
        self.vj_length = np.count_nonzero((initial == "v") | (initial == "j"))

        length_input = np.max([len(self.features), 1])

        min_clip = self.min_energy_clip
        max_clip = self.max_energy_clip
        l2_reg = self.l2_reg
        l1_reg = self.l1_reg
        max_depth = self.max_depth
        l_length = self.l_length
        vj_length = self.vj_length
        activation='tanh'

        if initialize:
            input_l = keras.layers.Input(shape=(l_length,), dtype="float32")
            input_cdr3 = keras.layers.Input(
                shape=(
                    max_depth * 2,
                    20,
                ),
                dtype="float32",
            )
            input_vj = keras.layers.Input(shape=(vj_length,), dtype="float32")
            input_layer = [input_l, input_cdr3, input_vj]

            if not self.deep:
                # linear model definition
                cdr3_length = input_l
                cdr3 = keras.layers.Flatten()(input_cdr3)
                vj = input_vj
                merge = keras.layers.Concatenate()([cdr3_length, cdr3, vj])
                output_layer = keras.layers.Dense(
                    1,
                    use_bias=False,
                    activation="linear",
                    kernel_regularizer=keras.regularizers.l1_l2(l2=l2_reg, l1=l1_reg),
                    kernel_initializer="zeros",
                )(merge)
            else:
                # cdr3 length reduced to dimension 10
                cdr3_length = keras.layers.Dense(
                    5,
                    activation=activation,
                    kernel_initializer="lecun_normal",
                    kernel_regularizer=keras.regularizers.l2(l2_reg),
                )(input_l)

                # cdr3 processes embeddings, flattens, and reduces to dimension 40
                cdr3 = EmbedViaMatrix(10)(input_cdr3)
                cdr3 = keras.layers.Flatten()(cdr3)
                cdr3 = keras.layers.Dense(
                    40,
                    activation=activation,
                    kernel_initializer="lecun_normal",
                    kernel_regularizer=keras.regularizers.l2(l2_reg),
                )(cdr3)

                # vj reduces to dimension 30
                vj = keras.layers.Dense(
                    25,
                    activation=activation,
                    kernel_initializer="lecun_normal",
                    kernel_regularizer=keras.regularizers.l2(l2_reg),
                )(input_vj)
                # merge
                merge = keras.layers.Concatenate()([cdr3_length, cdr3, vj])

                # 2 residual blocks of size 70
                for _ in range(2):
                    h = keras.layers.Dense(
                        70,
                        activation=activation,
                        kernel_initializer="lecun_normal",
                        kernel_regularizer=keras.regularizers.l2(l2_reg),
                    )(merge)
                    merge=keras.layers.Add()([h, merge])

                h= keras.layers.Concatenate()([cdr3_length, cdr3, vj, merge])


                output_layer = keras.layers.Dense(
                    1,
                    activation="linear",
                    use_bias=True,
                    kernel_initializer="lecun_normal",
                    kernel_regularizer=keras.regularizers.l2(l2_reg),
                )(h)

        # Define model
        clipped_out = keras.layers.Lambda(
            ko.clip,
            arguments={"x_min": min_clip, "x_max": max_clip},
        )(output_layer)

        self.model = keras.models.Model(inputs=input_layer, outputs=clipped_out)
        if self.optimizer_name == "adam":
            self.optimizer = keras.optimizers.Adam()
        elif self.optimizer_name == "rmsprop":
            self.optimizer = keras.optimizers.RMSprop()
        else:
            raise RuntimeError(''f"Optimizer {self.optimizer_name} not recognized.")

        if self.objective == "BCE":
            self.model.compile(
                optimizer=self.optimizer,
                loss=BinaryCrossentropy(from_logits=True),
                metrics=[
                    self._likelihood,
                    BinaryCrossentropy(from_logits=True, name="binary_crossentropy"),
                ],
            )
        else:
            self.model.compile(
                optimizer=self.optimizer,
                loss=self._loss,
                metrics=[
                    self._likelihood,
                    BinaryCrossentropy(from_logits=True, name="binary_crossentropy"),
                ],
            )
        self.model_params = self.model.get_weights()
        return True

    def split_encoding(self, encoding: NDArray[np.int8]) -> Tuple[NDArray[np.int8]]:
        length_encoding = encoding.shape[0]
        enc1 = encoding[:, : self.l_length]
        enc2 = (
            encoding[:, self.l_length : self.l_length + self.a_length]
            .reshape(length_encoding, 20, self.max_depth * 2)
            .swapaxes(1, 2)
        )
        enc3 = encoding[:, self.l_length + self.a_length :]
        return enc1, enc2, enc3

    def _load_features_and_model(
        self, feature_file: str, model_file: str, verbose: bool = True
    ) -> None:
        """Loads features and model.

        This is set as an internal function to allow daughter classes to load
        models from saved feature energies directly.
        """
        features = []
        data_marginals = []
        gen_marginals = []
        model_marginals = []
        initial = []

        with open(feature_file) as features_file:
            column_names = next(features_file)
            sonia_or_sonnia = column_names.split(",")[1]
            if sonia_or_sonnia == "marginal_data":
                k = 0
            else:
                k = 1

            for line in features_file:
                line = line.strip()
                splitted = line.split(",")
                features.append(splitted[0].split(";"))
                initial.append(features[-1][0][0])
                data_marginals.append(float(splitted[1 + k]))
                model_marginals.append(float(splitted[2 + k]))
                gen_marginals.append(float(splitted[3 + k]))

        self.features = np.array(features, dtype=object)
        self.data_marginals = np.array(data_marginals)
        self.model_marginals = np.array(model_marginals)
        self.gen_marginals = np.array(gen_marginals)

        self.feature_dict = {tuple(f): i for i, f in enumerate(self.features)}

        initial = np.array(initial)
        self.l_length = np.count_nonzero(initial == "l")
        self.a_length = np.count_nonzero(initial == "a")
        self.vj_length = np.count_nonzero((initial == "v") | (initial == "j"))

        self.update_model_structure(initialize=True)
        self.model.load_weights(model_file)


    def set_gauge(self):
        """
        Placeholder for gauges.
        """


class EmbedViaMatrix(keras.layers.Layer):
    """
    This layer defines a (learned) matrix M such that given matrix input X the
    output is XM. The number of columns of M is embedding_dim, and the number
    of rows is set so that X and M can be multiplied.
    If the rows of the input give the coordinates of a series of objects, we
    can think of this layer as giving an embedding of each of the encoded
    objects in a embedding_dim-dimensional space.
    Adapted from Vampire package.
    """

    def __init__(self, embedding_dim, **kwargs):
        self.embedding_dim = embedding_dim
        super(EmbedViaMatrix, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[2], self.embedding_dim),
            initializer="uniform",
            trainable=True,
        )
        super(EmbedViaMatrix, self).build(input_shape)

    def get_config(self):
        config = super(EmbedViaMatrix, self).get_config().copy()
        config.update({"embedding_dim": self.embedding_dim})
        return config

    def call(self, x):
        return ko.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.embedding_dim)
