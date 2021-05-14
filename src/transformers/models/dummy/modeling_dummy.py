# coding=utf-8
# Copyright 2021 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Dummy model. """

import warnings
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_outputs import BaseModelOutput, CausalLMOutput, MaskedLMOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_dummy import DummyConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DummyConfig"

import os
DUMMY_PRETRAINED_MODEL_ARCHIVE_LIST = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained')
    # See all Dummy models at https://huggingface.co/models?filter=dummy
]

class DummyPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DummyConfig
    base_model_prefix = "dummy"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.kaiming_normal_(module.weight.data)
        if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
            module.bias.data.zero_()


DUMMY_START_DOCSTRING = r"""
    Dummy was proposed in `dummy 2.0: A Framework for Self-Supervised Learning of Speech Representations
    
    This model inherits from :class:`~transformers.PreTrainedModel`.

    Parameters:
        config (:class:`~transformers.DummyConfig`): Model configuration class.
"""


DUMMY_INPUTS_DOCSTRING = r"""
    Args:
        input_values (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            Float values of input raw speech waveform.
"""


@add_start_docstrings(
    "The bare Dummy Model transformer outputting raw hidden-states without any specific head on top.",
    DUMMY_START_DOCSTRING,
)
class DummyModel(DummyPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.linear = nn.Linear(1, config.hidden_size)

        self.init_weights()

    @add_start_docstrings_to_model_forward(DUMMY_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """

        Returns:

        Example::

            >>> from transformers import DummyModel
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.linear(torch.unsqueeze(input_values, 2))

        if not return_dict:
            return (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=None,
            attentions=None,
        )

@add_start_docstrings(
    """Dummy Model with a head. """,
    DUMMY_START_DOCSTRING,
)
class DummyForCTC(DummyPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.dummy = DummyModel(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        self.init_weights()

    @add_start_docstrings_to_model_forward(DUMMY_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_length)`, `optional`):
            Labels for connectionist temporal classification.

        Returns:

        Example::

            >>> import torch
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.dummy(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
