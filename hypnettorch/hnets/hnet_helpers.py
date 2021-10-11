#!/usr/bin/env python3
# Copyright 2020 Christian Henning
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @title          :hnets/hnet_helpers.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :12/11/2020
# @version        :1.0
# @python_version :3.6.10
"""
Helper functions for hypernetworks
----------------------------------

The module :mod:`hnets.hnet_helpers` contains utilities that should simplify
working with hypernetworks that implement the interface
:class:`hnets.hnet_interface.HyperNetInterface`. Those helper functions are
meant to handle common manipulations (such as embedding initialization) in an
abstract way that hides implementation details to the user.
"""
from warnings import warn
import torch

from hypnettorch.hnets.chunked_deconv_hnet import ChunkedHDeconv
from hypnettorch.hnets.chunked_mlp_hnet import ChunkedHMLP
from hypnettorch.hnets.hnet_container import HContainer
from hypnettorch.hnets.hnet_perturbation_wrapper import HPerturbWrapper
from hypnettorch.hnets.hnet_interface import HyperNetInterface
from hypnettorch.hnets.deconv_hnet import HDeconv
from hypnettorch.hnets.mlp_hnet import HMLP
from hypnettorch.hnets.structured_mlp_hnet import StructuredHMLP

def init_conditional_embeddings(hnet, normal_mean=0., normal_std=1.,
                                init_fct=None):
    """Initialize internally maintained conditional input embeddings.

    This function initializes conditional embeddings if the hypernetwork has
    any and they are internally maintained. For instance, the conditional
    embeddings of an ``HMLP`` instance are those returned by the method
    :meth:`hnets.mlp_hnet.HMLP.get_cond_in_emb`.

    By default, those embedding will follow a normal distribution. However, one
    may pass a custom init function ``init_fct`` that receives the embedding
    and its corresponding conditional ID as input (as is expected to modify the
    embedding in-place):

    .. code-block:: python

        init_fct(cond_emb, cond_id)

    Hypernetworks that don't make use of internally maintained conditional input
    embeddings will not be affected by this function.

    Note:
        Chunk embeddings may also be conditional parameters, but are not
        considered conditional input embeddings here. Conditional chunk
        embeddings can be initialized using function
        :func:`init_chunk_embeddings`.

    Args:
        hnet (hnets.hnet_interface.HyperNetInterface): The hypernetwork whose
            conditional embeddings should be initialized.
        normal_mean (float): The mean of the normal distribution with which
            embeddings should be initialized.
        normal_std (float): The std of the normal distribution with which
            embeddings should be initialized.
        init_fct (func, optional): A function handle that receives a conditional
            embedding and its ID as input and initializes the embedding
            in-place. If provided, arguments ``normal_mean`` and ``normal_std``
            will be ignored.
    """
    assert isinstance(hnet, HyperNetInterface)

    if hnet.conditional_params is None:
        warn('Conditional parameters are not internally maintained by the ' +
             'hypernetwork!')
        return

    if isinstance(hnet, (HMLP, HDeconv, ChunkedHMLP, ChunkedHDeconv,
                         StructuredHMLP)):
        for cond_id in range(hnet.num_known_conds):
            try:
                cond_emb = hnet.get_cond_in_emb(cond_id)
            except:
                # This may occur if the `hnet` has conditional parameters but
                # not conditional input embeddings (e.g., a `ChunkedHMLP` with
                # only conditional chunk embeddings).
                return

            if init_fct is None:
                torch.nn.init.normal_(cond_emb, mean=normal_mean,
                                      std=normal_std)
            else:
                init_fct(cond_emb, cond_id)

    elif isinstance(hnet, HContainer):
        # We simply loop through the provided `hnets`, assuming
        # `cond_param_shapes` can't be conditional input embeddings.
        for int_hnet in hnet.internal_hnets:
            if int_hnet.conditional_params is None:
                continue
            init_conditional_embeddings(int_hnet, normal_mean=normal_mean,
                normal_std=normal_std, init_fct=init_fct)

    elif isinstance(hnet, HPerturbWrapper):
        init_conditional_embeddings(hnet.internal_hnet, normal_mean=normal_mean,
                normal_std=normal_std, init_fct=init_fct)

    else:
        raise NotImplementedError('Function not implemented for hypernetwork ' +
                                  'of type "%s".' % type(hnet))

def init_chunk_embeddings(hnet, normal_mean=0., normal_std=1., init_fct=None):
    """Initialize chunk embeddings.

    This function only applies to hypernetworks that make use of chunking,
    such as :class:`hnets.chunked_mlp_hnet.ChunkedHMLP`. All other hypernetwork
    types will be unaffected by this function.

    This function handles the initialization of embeddings very similar to
    function :func:`init_conditional_embeddings`, except that the function
    handle ``init_fct`` has a slightly different signature. It receives two
    positional arguments, the chunk embedding and the chunk embedding ID as well
    as one optional argument ``cond_id``, the conditional ID (in case of
    conditional chunk embeddings).

    .. code-block:: python

        init_fct = lambda cemb, cid, cond_id=None : nn.init.constant_(cemb, 0)

    Note:
        Class :class:`hnets.structured_mlp_hnet.StructuredHMLP` has multiple
        sets of chunk tensors as specified by attribute
        :attr:`hnets.structured_mlp_hnet.StructuredHMLP.chunk_emb_shapes`. As
        a simplifying design choice, the tensors passed to ``init_fct`` will not
        be single embeddings (i.e., vectors), but tensors of embeddings
        according to the shapes in attribute
        :attr:`hnets.structured_mlp_hnet.StructuredHMLP.chunk_emb_shapes`.

    Args:
        (....): See docstring of function :func:`init_conditional_embeddings`.
    """
    assert isinstance(hnet, HyperNetInterface)

    if isinstance(hnet, (HMLP, HDeconv)):
        return # No chunk embeddings

    elif isinstance(hnet, (ChunkedHMLP, ChunkedHDeconv, StructuredHMLP)):
        num_conds = hnet.num_known_conds if hnet.cond_chunk_embs else 1

        for cid in range(num_conds):
            if hnet.cond_chunk_embs:
                cond_id = cid
            else:
                cond_id = None

            if isinstance(hnet, StructuredHMLP):
                try:
                    cembs = hnet.get_chunk_embs(cond_id=cond_id)
                except:
                    return

                for chunk_id, cemb in enumerate(cembs):
                    # Note, here `cemb` might be a collection of embeddings,
                    # rather than a single one. So `chunk_id` is a bit
                    # misleading.
                    if init_fct is None:
                        torch.nn.init.normal_(cemb, mean=normal_mean,
                                              std=normal_std)
                    else:
                        init_fct(cemb, chunk_id, cond_id=cond_id)
            else:
                for chunk_id in range(hnet.num_chunks):
                    try:
                        cemb = hnet.get_chunk_emb(chunk_id=chunk_id,
                                                  cond_id=cond_id)
                    except:
                        return

                    if init_fct is None:
                        torch.nn.init.normal_(cemb, mean=normal_mean,
                                              std=normal_std)
                    else:
                        init_fct(cemb, chunk_id, cond_id=cond_id)

    elif isinstance(hnet, HContainer):
        # We simply loop through the provided `hnets`, assuming
        # `uncond_param_shapes` and `cond_param_shapes` can't be chunk
        # embeddings.
        for int_hnet in hnet.internal_hnets:
            init_chunk_embeddings(int_hnet, normal_mean=normal_mean,
                normal_std=normal_std, init_fct=init_fct)

    elif isinstance(hnet, HPerturbWrapper):
        init_chunk_embeddings(hnet.internal_hnet, normal_mean=normal_mean,
                normal_std=normal_std, init_fct=init_fct)

    else:
        raise NotImplementedError('Function not implemented for hypernetwork ' +
                                  'of type "%s".' % type(hnet))

def get_conditional_parameters(hnet, cond_id):
    """Get condition specific parameters from the hypernetwork.

    Example:
        Class :class:`hnets.mlp_hnet.HMLP` may only have one embedding (the
        conditional input embedding) per condition as conditional parameter.
        Thus, this function will simply return
        ``[hnet.get_cond_in_emb(cond_id)]``.

    Args:
        hnet (hnets.hnet_interface.HyperNetInterface): The hypernetwork whose
            conditional parameters regarding ``cond_id`` should be extraced.
        cond_id (int): The condition (or its conditional ID) for which
            parameters should be extraced.

    Returns:
        (list): A list of tensors, a subset of attribute
            :attr:`hnets.hnet_interface.HyperNetInterface.conditional_params`,
            that are specific to the condition ``cond_id``. An empty list is
            returned if conditional parameters are not maintained internally.
    """
    assert isinstance(hnet, HyperNetInterface)

    if hnet.conditional_params is None:
        warn('Conditional parameters are not internally maintained by the ' +
             'hypernetwork!')
        return []

    assert cond_id < hnet.num_known_conds

    if isinstance(hnet, (HMLP, HDeconv)):
        return [hnet.get_cond_in_emb(cond_id)]

    elif isinstance(hnet, (ChunkedHMLP, ChunkedHDeconv, StructuredHMLP)):
        ret = [hnet.get_cond_in_emb(cond_id)]

        if hnet.cond_chunk_embs:
            if isinstance(hnet, StructuredHMLP):
                ret += hnet.get_chunk_embs(cond_id=cond_id)
            else:
                ret.append(hnet.get_chunk_emb(cond_id=cond_id))

        return ret

    elif isinstance(hnet, HContainer):
        ret = []

        for int_hnet in hnet.internal_hnets:
            if int_hnet.conditional_params is None:
                continue
            ret += get_conditional_parameters(int_hnet, cond_id)

        # The HContainer can have additional conditional parameters passed
        # via constructor argument `cond_param_shapes`.
        for meta in hnet.param_shapes_meta:
            if 'celement_type' in meta.keys() and \
                    meta['celement_type'] == 'cond':
                if meta['celement_cind'] == cond_id:
                    assert meta['index'] != -1
                    ret.append(meta.internal_params[meta['index']])

    elif isinstance(hnet, HPerturbWrapper):
        return get_conditional_parameters(hnet.internal_hnet, cond_id)

    else:
        raise NotImplementedError('Function not implemented for hypernetwork ' +
                                  'of type "%s".' % type(hnet))

if __name__ == '__main__':
    pass


