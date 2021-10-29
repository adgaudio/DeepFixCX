import torch as T
import torch.fx
import networkx as nx
from collections import deque
from efficientnet_pytorch.utils import Conv2dStaticSamePadding


def get_submodule(model:T.nn.Module, target:str):
    """workaround to remain backwards compatible with
    earlier pytorch versions < 1.9.
    Copies the code from pytorch 1.9 with minor changes.
    """
    if hasattr(model, 'get_submodule'):
        return model.get_submodule(target)
    else:
        if target == "":
            return model
        atoms: list[str] = target.split(".")
        mod: T.nn.Module = model
        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(mod._get_name() + " has no "
                                     "attribute `" + item + "`")
            mod = getattr(mod, item)
            if not isinstance(mod, T.nn.Module):
                raise AttributeError("`" + item + "` is not "
                                     "an nn.Module")
        return mod


def get_graph(model:T.nn.Module) -> nx.DiGraph:
    """
    Construct a directed graph showing how all the modules in the
    network are connected.

    Use the (experimental beta but useful) symbolic trace
    functionality of Pytorch.  Get away from the beta code asap by outputting a
    networkx digraph of modules.

    Future note: using pytorch FX library directly with some transforms
    could be a much better future approach to pruning.  The FX library needs
    more development and documentation though. 2021-07-05
    """
    lookup = dict(model.named_modules())
    tracer = T.fx.symbolic_trace(model.eval())
    graph = nx.DiGraph()
    # add modules
    nodes = []
    for node in tracer.graph.nodes:
        child_name = node.name
        if node.op == 'call_module':
            meta = dict( module=lookup[node.target], module_name=node.target,)
        elif node.op == 'call_function':
            meta = dict( module=node.target, module_name=None,)
        else:
            meta = dict( module=None, module_name=node.target,)
        assert child_name not in graph.nodes, 'sanity check: fx gave duplicate node %s' % child_name
        graph.add_node(child_name,
                       op=node.op, args=node.args, kwargs=node.kwargs, **meta)
        nodes.append((child_name, node))
    for child_name, node in nodes:
        for parent in node.all_input_nodes:
            parent_name = parent.name
            graph.add_edge(parent_name, child_name)
    #  nx.draw_networkx(graph, pos=nx.spectral_layout(graph))
    return graph


def replace_module(model, name, module):
    name_list = name.split(".")
    for name in name_list[:-1]:
        model = getattr(model, name)
    setattr(model, name_list[-1], module)


def traverse(graph:nx.DiGraph, initial_mode:str, node):
    """
    Coroutine to traverse neighbor nodes along a path. Initially looking at
    only descendants or ancestors, but switch directions at "v" nodes (like
    when encounter skip connections).  If the path splits into two, traverse
    both paths.  Don't stop traversing down any branch until a False is sent to
    the coroutine.  Iteratively return nodes along this traversal path until
    the path is stopped.
    When traversing a "V" node, can use the `from_node` and `msg=='v_node_switch'`
    to figure out if traversing up a V and switched directions.  For instance,
    don't want to traverse v nodes for channel-wise concatenation functions
    (T.cat), but do want to traverse them for sum functions.
    #
        :graph: a networkx.DiGraph instance
        :initial_mode: 'descendants' or 'ascendants' denoting whether to traverse down or up.
        :node: a node in your graph

        :returns: None, but the coroutine returns (node, mode) where
            mode is "descendants" or "ascendants" for going down and going up
    #
    The coroutine only accepts .send(True) or .send(False). Don't call next(co) or iter(co).
    Pass a True to continue along the path.  False stops the path (but continues evaluating any other branches)
    #
    >>> G = nx.DiGraph(...)  # with nodes in it
    >>> co = traverse(G.succ, 'myrootnode')
    >>> next_node, from_node, msg, mode, _ = co.send(pruned_channels)  # initialize coroutine
    >>> assert from_node == 'myrootnode'
    >>> co.send(pruned_channels)  # continue along this path, following descendant1 in given direction `mode`
    >>> next_node, from_node, msg, mode, pruned_channels = co.send(False)  # stop this path (but explore other branches previously found)
    >>> co.send(False)  # when no more descendants left
    StopIteration:
    """
    _modes = ['ancestors', 'descendants']  # indexes of [0, 1]
    modes = {1: graph.succ, 0: graph.pred}
    seen = set([ (node, _modes.index(initial_mode)) ])
    start_state = (node, _modes.index(initial_mode))
    branches = deque([(node, None, 'start_node', _modes.index(initial_mode), None)])
    #  branches = set([(node, _modes.index(initial_mode), None)])
    # in code, assume we are descending without loss of generality.
    # it works the same both ways.  it's just easier to code.
    while branches:
        node, from_node, msg, mode, pruned_channels = branches.popleft()
        pruned_channels = (yield (node, from_node, msg, _modes[mode], pruned_channels))
        if not (isinstance(pruned_channels, T.Tensor) or pruned_channels in {True,False}):
            raise UserWarning("This traverse() object is not a coroutine.  Use .send(tensor) or .send(False).  Don't use next(gen).")
        pursue_this_path = pruned_channels is not False
        if pursue_this_path is False:
            continue
        neighbors = [(child, mode) for child in modes[mode][node]]
        #  for child in modes[mode][node]:
        for child, mode in neighbors:
            if child == node:  # and mode == _modes.index(initial_mode):
                raise UserWarning('code bug: self loop in graph')
            if (child, mode) in seen:
                #  raise Exception('code error: seen this node already %s' % ((child, _modes[mode]),) )  # I think will need to remove this for v nodes.
                raise UserWarning('code bug: already seen this node and direction %s' % ((child, mode), ))
                continue  # ignore self-loops (and visited itms)
            seen.add((child, mode))
            branches.append((child, node, 'unchanged_direction', mode, pruned_channels))
        # switch directions on the path at v nodes
        if (node, mode) == start_state:
            continue
        for parent in modes[mode^1][node]:
            if (parent, mode) in seen or (parent, mode^1) in seen:
                continue
            seen.add((parent, mode^1))
            branches.append((parent, node, 'v_node_switch', mode^1, pruned_channels))


def prune_model(model:T.nn.Module):
    """
    Modify model inplace.
    Prune input and output channels of all convolution layers
    and their neighboring layers by a connectivity graph traversal.  A kernel
    is considered for pruning only if its kernel weights are all exactly zeros.
    If all kernels related to an input channel or output channel are zero, then
    that channel is pruned from the convolution.  All neighboring layers are
    also modified so that the network inputs and outputs are compatible.

    This pruning function follows some hardcoded rules.  It stops pruning
    neighbors when it hits another conv or linear layer.  Stopping at other
    layers (like ConvTranspose) is not currently implemented.  It works with
    common layers, like AvgPooling and BatchNorm, and with layers invariant to
    changes in the number of channels input or output.  It does not currently
    support concatenation along channels.  It does not currently work with some
    existing network architectures without code additions.  It does not work
    with variables with channel-specific information (e.g. conv weights) are
    manually passed into pytorch backend functions (e.g. T.conv2d) It requires
    using torch.fx to build a graph.  Torch.fx is experimental and beta.

    The pruned model should have the same output during
    inference as the non-pruned model unless your model has non-deterministic
    outputs or bias terms.  Batchnorm and bias terms hallucinate values even when
    activations are zero, and this hallucination can be important for good
    predictions (basically, it's like memorizing information with bias terms).
    The pruned model should be computationally faster on inference and training
    and it will use less RAM.  It may take more epochs to fine-tune!
    """
    graph = get_graph(model)
    for node, attrs in graph.nodes.items():
        if not isinstance(node, str):
            continue  # ignore the tracer objects that are torch functions
                      # and assume they have nothing to do with pruning
        if attrs['op'] != 'call_module':
            continue  # only modify modules.  don't prune weights for functions, though with FX we could.
        # start pruning at conv layers that have exactly zero weights
        layer = get_submodule(model, attrs['module_name'])
        if isinstance(layer, (T.nn.Conv1d, T.nn.Conv2d, T.nn.Conv3d)):
            newlayer, pruned_inputs, pruned_outputs = prune_conv_layer(layer)
            replace_module(model, attrs['module_name'], newlayer)
        else:
            continue  # don't prune starting from this layer
        #  prune downstream neighbors
        if newlayer.out_channels != layer.out_channels:
            _prune_neighbors('descendants', pruned_outputs, model, graph, node)
        # prune upstream neighbors
        if newlayer.in_channels != layer.in_channels:
            _prune_neighbors('ancestors', pruned_inputs, model, graph, node)


def _find_channels_per_parent(graph, _mode, parent, model, seen):
    """Helper for figuring out how to pass pruning information through the
    T.cat function"""
    co = traverse(graph, _mode, parent)
    send_ = None
    while True:
        node, _, _, _mode, _ = co.send(send_)
        send_ = True
        # if we fail due to stopiteration from co.send(...),
        # then we shouldn't be calling this function in the first place!
        stopping_modules = ( T.nn.Conv1d, T.nn.Conv2d, T.nn.Conv3d, 
            # T.nn.ConvTranspose1d, T.nn.ConvTranspose2d, T.nn.ConvTranspose3d
            T.nn.Linear
            )
        if graph.nodes[node]['module_name'] is None:
            continue

        if node in {x[0] for x in seen}:
            layer = graph.nodes[node]['module']
        else:
            layer = get_submodule(model, graph.nodes[node]['module_name'])
        #
        if isinstance(layer, stopping_modules):
            if _mode == 'descendants':
                return layer.weight.shape[1]  # input channels
            else:
                assert _mode == 'ancestors'
                return layer.weight.shape[0]  # output channels

def _prune_neighbors(initial_mode:str, pruned_channels:T.Tensor,
                     model:T.nn.Module,
                     graph:nx.DiGraph, node:str, seen=None):
    """Modify layers of the given model so that they have the correct number of
    input and/or output channels.  See prune_model(...) docstring for details.
    This step changes in and out channels of different
    layer types by utilizing a graph traversal mechanism.
    """
    assert initial_mode in {'descendants', 'ancestors'}
    # traverse child branches until all are modified
    co = traverse(graph, initial_mode, node)
    descendant_name, _, _mode, _, _ = co.send(None)
    _traverse_this_branch = pruned_channels
    if seen is None:
        seen = {(descendant_name, _mode)}
    while True:
        try:
            descendant_name, from_node, msg, _mode, pruned_channels = co.send(_traverse_this_branch)
        except StopIteration:
            break
        # do not traverse up v-nodes if the function is T.cat.
        if msg == 'v_node_switch' and graph.nodes[from_node]['module'] in {T.cat, }:
            _traverse_this_branch = False
            continue
        if (descendant_name, _mode) in seen:
            raise UserWarning("code bug sanity check - I think this shouldn't happen")
        seen.add((descendant_name, _mode))
        # figure out if we should reduce inputs or outputs.
        def should_i_prune_inputs_or_outputs(_mode):
            """
            in 'descendants' mode (going down the graph to leaves), we should
                reduce input channels
            in 'ancestors' mode (going up the graph to root), we should
                reduce output channels.
            """
            return 'prune the inputs' if _mode == 'descendants' else 'prune the outputs'
        mode = should_i_prune_inputs_or_outputs(_mode)
        #  print('debugging', node, descendant_name, from_node, msg, _mode, pruned_channels.shape)
        descendant_layer = graph.nodes[descendant_name]['module_name']
        if descendant_layer is not None:
            descendant_layer = get_submodule(
                model, graph.nodes[descendant_name]['module_name'])
        # Prune Linear Layers
        if isinstance(descendant_layer, T.nn.Linear):
            _traverse_this_branch = False
            newlayer = T.nn.Linear(
                in_features=(~pruned_channels).sum() if mode == 'prune the inputs' else descendant_layer.in_features,
                out_features=(~pruned_channels).sum() if mode == 'prune the outputs' else descendant_layer.out_features,
                bias=descendant_layer.bias is not None)
            state = descendant_layer.state_dict()
            if mode == 'prune the inputs':  # reduce input channels
                state['weight'] = state['weight'][:, ~pruned_channels]
            elif mode == 'prune the outputs':  # reduce output channels
                state['weight'] = state['weight'][~pruned_channels, :]
                if 'bias' in state:
                    state['bias'] = state['bias'][~pruned_channels]
            newlayer.load_state_dict(state)
            newlayer.to(descendant_layer.weight.device, non_blocking=True)
            replace_module(model, graph.nodes[descendant_name]['module_name'], newlayer)
        # Prune Conv Layers
        elif isinstance(descendant_layer, (T.nn.Conv1d, T.nn.Conv2d, T.nn.Conv3d)):
            _traverse_this_branch = False
            if mode == 'prune the inputs':
                in_, out_ = pruned_channels, T.zeros(
                    descendant_layer.out_channels,
                    dtype=T.bool, device=pruned_channels.device)
            elif mode == 'prune the outputs':
                in_, out_ = (
                    T.zeros(descendant_layer.in_channels,
                            dtype=T.bool, device=pruned_channels.device),
                    pruned_channels)
            newlayer = get_replacement_conv_layer(descendant_layer, in_, out_)
            replace_module(model, graph.nodes[descendant_name]['module_name'], newlayer)
        # Prune ConvTranspose layers
        elif isinstance(descendant_layer, (T.nn.ConvTranspose1d,
                                           T.nn.ConvTranspose2d,
                                           T.nn.ConvTranspose3d)):
            raise NotImplementedError(
                'code comment: I think it can be treated same as conv[123]d above,'
                " but I didn't test it.")
            #  _traverse_this_branch = False
            #  newlayer = ...
            #  replace_module(model, graph.nodes[descendant_name]['module_name'], newlayer)
        # Prune BatchNorm layers
        elif isinstance(descendant_layer, (T.nn.BatchNorm1d, T.nn.BatchNorm2d, T.nn.BatchNorm3d)):
            # continue through batchnorm layers
            _traverse_this_branch = pruned_channels
            newlayer = descendant_layer.__class__(
                num_features=(~pruned_channels).sum().item(),
                **{k: getattr(descendant_layer, k)
                   for k in descendant_layer.__constants__
                   if k != 'num_features'})
            newlayer.load_state_dict(
                {k: v[~pruned_channels] if k!='num_batches_tracked' else v
                 for k, v in descendant_layer.state_dict().items()})
            newlayer.to(descendant_layer.bias.device, non_blocking=True)
            replace_module(model, graph.nodes[descendant_name]['module_name'], newlayer)
        # Try to work with other modules, unless it has learned params.
        elif graph.nodes[descendant_name]['op'] == 'call_module':
            # continue through arbitrary layers.  Assume they don't modify channels
            # This is brittle but works for typical activation and pooling layers
            # and simple modules.  Raise an error if find modules with learned parameters.
            _traverse_this_branch = pruned_channels
            if descendant_layer.state_dict() or list(descendant_layer.buffers()):
                print("Warning: pruning may fail or cause silent errors"
                      " because don't know how to process this layer: %s"
                      % descendant_name)
        # Try to work with pytorch functions
        elif graph.nodes[descendant_name]['op'] == 'call_function':
            _traverse_this_branch = pruned_channels

            #  if descendant_name == 'cat_5':
                #  import IPython ; IPython.embed() ; import sys ; sys.exit()
            meta = graph.nodes[descendant_name]
            fn = meta['module']

            # Channel-wise concatenation using T.cat(..., dim=1)
            if fn == T.cat:
                arg_dim = meta['kwargs'].get('dim', meta['args'][1])
                if arg_dim != 1: continue
                # --> get incoming connections to this concat node
                parent_nodes = [_node.name for _node in meta['kwargs'].get(
                    'tensors', meta['args'][0])]
                if len(parent_nodes) < 2: continue
                num_channels_per_parent:T.Tensor = T.empty(len(parent_nodes), dtype=T.int)  # tensor  TODO by graph traversal until closest conv, batchnorm, linear.
                for n, parent in enumerate(parent_nodes):
                    assert _mode in {'ascendants', 'descendants'}, 'code bug'
                    num_channels_per_parent[n] = _find_channels_per_parent(
                        graph, 'ancestors', parent, model, seen)
                if mode == 'prune the outputs':
                    # --> pass up through the cat operator, splitting the
                    # pruned_channels appropriately for the outputs of each
                    # parent node.
                    raise NotImplementedError('channel concatenation, prune ancestor nodes')
                elif mode == 'prune the inputs':
                    # --> pass down through the cat operator, making
                    # pruned_channels larger with padded zeros to match the
                    # output channels of the operator
                    primary_parent = list({x[0] for x in seen}.intersection(parent_nodes))
                    assert len(primary_parent) == 1
                    _parent_idx = parent_nodes.index(primary_parent[0])
                    if _parent_idx == 0:
                        _start_channels = 0
                    else:
                        _start_channels = num_channels_per_parent.cumsum(0)[_parent_idx-1]
                    new_pruned_channels = T.zeros(
                        (num_channels_per_parent.sum(), ),
                        dtype=pruned_channels.dtype, device=pruned_channels.device)
                    new_pruned_channels[_start_channels:_start_channels+len(pruned_channels)] = pruned_channels
                    _traverse_this_branch = new_pruned_channels
                else:
                    raise UserWarning('code error: unrecognized mode')

            # anothe problem: models that call pytorch functions directly with
            # parameters not registered in modules is not supported.  This
            # could be supported if we traversed up to all parent
            # nodes of op='get_attr'.  This may raise an error in these cases.
        else:
            raise Exception()


def get_replacement_conv_layer(layer, prune_these_inputs, prune_these_outputs):
    is_depthwise_separable = (
        layer.groups == layer.in_channels == layer.out_channels)
    # --> optimization: allow a specific kind of depthwise separable to just
    # use a normal convolution layer.  Otherwise, grouped convs need to apply
    # multiple convs in parallel (one per block of the block diagonal matrix in
    # prune_conv_layer)
    if layer.groups > 1 and not is_depthwise_separable:
        replacement_layer = ...
        raise NotImplementedError(
            'Not implemented yet! Please consider submitting a GH issue and'
            ' then PR for this if you like.  Implementation is a set of'
            ' parallel convolutions, one per block of the'
            ' block diagonal matrix')
    else:
        kws = dict(
            in_channels=(~prune_these_inputs).sum(),
            out_channels=(~prune_these_outputs).sum(),
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=(
                (~prune_these_inputs).sum().item()
                if is_depthwise_separable else layer.groups),
            bias=layer.bias is not None,
            padding_mode=layer.padding_mode,
        )
        if isinstance(layer, Conv2dStaticSamePadding):
            # special handling for efficientnet_pytorch, the annoying bad code design.
            kws.update({'image_size': -1})  # ignore it, and replace static_padding later.
        replacement_layer = layer.__class__(
            **kws)
        if isinstance(layer, Conv2dStaticSamePadding):
            # special handling for efficientnet_pytorch, the annoying bad code design.
            replacement_layer.static_padding = layer.static_padding
    # update the bias
    if layer.bias is not None:
        replacement_layer.bias.data[:] = layer.bias.data[~prune_these_outputs]
        replacement_layer.bias.requires_grad_(layer.bias.requires_grad)
    # update the spatial weights
    tmp = layer.weight.data[~prune_these_outputs]
    if not is_depthwise_separable:
        tmp = tmp[:, ~prune_these_inputs]
    replacement_layer.weight.data[:] = tmp
    replacement_layer.weight.requires_grad_(layer.weight.requires_grad)
    replacement_layer = replacement_layer.to(layer.weight.device, non_blocking=True)
    return replacement_layer


def prune_conv_layer(layer:T.nn.Module, return_a_pruned_layer:bool=True
                     ) -> (T.nn.Module, T.Tensor, T.Tensor):
    """Prune a pytorch Conv layer under assumption that some of the layer's 
    spatial weights are exactly zero.  Optionally return a replacement layer.

    This function exists because Pytorch does not support sparse convolutions.
    Attempt to prune input and output channels, updating in_channels,
    out_channels, weight and bias accordingly.  This function even works with
    grouped convolutions by giving a single layer that evaluates multiple conv
    layers in "parallel".  Using this function to replace any single conv layer
    of a network will require changes to connected nodes in the network, hence
    this function also specifies which input and output channels were pruned.

    TODO NOTE: have not implemented the parallelized grouped convolutions
    since I don't need them yet.

    Input channels can be pruned when they are not used for any outputs (ALL
    corresponding spatial weights are zero).
    Output channels can be pruned when all corresponding spatial input weights
    are zero.

    :layer: a pytorch Conv1d, Conv2d or Conv3d layer.  The spatial
        weights to consider pruning should be already set to zero.
    :return_a_pruned_layer: iff True, return a pruned layer.
    :returns: (replacement, pruned_input_channels, pruned_output_channels)
        - `replacement` is None if return_a_pruned_layer is not True, and is a
            T.nn.Module otherwise
        - `pruned_[input|output]_channels` are boolean vectors specifying which
            input and output channels were removed.

    Implementation details:
    We apply the following steps:
        define:
            O: out channels
            I: in channels
            G: groups
        assume:
            The way pytorch assigns output channels to input channels when
            using groups=G is to take first G input channels as group 1, and
            the second G channels as group 2, ...
            Given the assumption, we can create a block diagonal matrix, where
            each block has the outputs and inputs that are applied to each other.

            This assumption is not currently advertised in the pytorch api
            documentation, so this assumption is not guaranteed, to my
            knowledge. (alex 2021-06-27).  I added tests called
            test_pytorch_assumption_prune_output_channel and
            test_pytorch_assumption_prune_output_channel to verify this
            assumption holds.
        implement:
        1. Reduce the spatial filter weights to a binary matrix (O, I/G) that
            is true if the spatial filter is nonzero.  Note that PyTorch conv
            layer weights are of shape (O,I/G,...).
        2. Re-format this matrix into a block diagonal matrix, B.  Each block is
            of shape (O/G, I/G), and the resulting matrix has shape (O,I).
            This assigns each row to an output channel,
            and each column to an input channel.
        3. In this block diagonal matrix:
            Input channels that can be pruned have a column of all zeros.
            Output channels to prune have a row of all zeros.
        4. Generate a new layer based on this information.
           Grouped convolution layers that are not depthwise separable can be
           replaced with a set of non-grouped convolutions run in parallel (but
           I didn't implement this).
    """
    # reduce the conv layer to a binary matrix of shape (out_channels,
    # in_channels) to identify which spatial filters were zeroed out.
    spatial_dims = [-1*x for x in range(1, layer.weight.ndim-2+1)]
    mags = layer.weight.data.abs().sum(spatial_dims)
    nonzero = (mags != 0)
    assert nonzero.shape == (layer.out_channels, layer.in_channels / layer.groups), 'sanity check'
    # get the output channels we can remove.  can remove an output if all
    # inputs are zero.  (this is only possible for grouped convolutions)
    prune_these_outputs = (nonzero.sum(1) == 0)
    # get the input channels we can remove (this works even with grouped convs)
    tensors = T.block_diag(*list(nonzero.reshape(
        layer.groups,
        layer.out_channels // layer.groups,
        layer.in_channels // layer.groups)))
    nonzero2 = T.block_diag(tensors)
    prune_these_inputs = (nonzero2.sum(0) == 0)
    assert (prune_these_outputs == (nonzero2.sum(1) == 0)).all()
    # now create a replacement conv layer
    if return_a_pruned_layer:
        replacement_layer = get_replacement_conv_layer(
            layer, prune_these_inputs, prune_these_outputs)
    else:
        replacement_layer = None
    return replacement_layer, prune_these_inputs, prune_these_outputs


class PrunedConvWrapper(T.nn.Module):
    """For testing.  Prune input and output channels of a convolution layer
    yet still work as a drop-in replacement.  Makes the model SLOWER!

    Normally, pruning a layer means the neighboring layers need to be adjusted
    too.  For instance, pruning the input channels typically means the output
    of the previous layer should be adjusted to match this.  Clearly, modifying
    the surrounding network should give better efficiency gains, but this can be
    complicated to implement.

    This module allows the pruned convolution to serve as a drop-in replacement
    by ignoring certain (pruned) input channels, and inserting zeros to fake
    the pruned output channels.  A drop-in replacement is not an ideal solution.
    Adding zeros to output channels adds some overhead, and full gradients are
    still computed.  As a result, the pruned model is SLOWER using this method!
    """
    def __init__(self,
                 pruned_conv=None, pruned_inputs=None, pruned_outputs=None,
                 *, conv=None,
                 ):
        """receive as input either conv or the output of prune_conv_layer(conv)
        side note: too bad python doesn't have multiple dispatch...
        conv is a T.nn.Conv1d, T.nn.Conv2d or T.nn.Conv3d
        """
        super().__init__()
        if conv is not None:
            pruned_conv, pruned_inputs, pruned_outputs = prune_conv_layer(conv)
        self.pruned_conv, self.unpruned_inputs, self.unpruned_outputs = (
            pruned_conv, ~pruned_inputs, ~pruned_outputs)

    def forward(self, x):
        # ignore the pruned input channels
        x = x[:,self.unpruned_inputs]
        # compute the convolution
        x = self.pruned_conv(x)
        # zero pad the missing output channels
        B,_C = x.shape[:2]
        C = len(self.unpruned_outputs)
        tmp = T.zeros(x.shape[0],C,*x.shape[2:], dtype=x.dtype, device=x.device)
        tmp[:,self.unpruned_outputs] = x
        #  tmp[:, self.pruned_outputs] = 0
        return tmp


def test_pytorch_assumption_prune_output_channel():
    """test assumption that pytorch grouped convolution groups inputs as
    consecutive blocks by pruning an output channel"""
    z = T.nn.Conv2d(4, 2, 1, groups=2, bias=False)
    x = T.rand(1,z.in_channels,1,1) + 1
    expected = z(x)
    assert (expected == 0).sum() == 0, 'sanity check'
    z.weight.data[0] = 0  # remove the output channel weights
    pred = z(x)
    assert pred[0,0] == 0, 'zeroed output channel should be zero'
    assert pred[0,1] == expected[0,1], 'non-zeroed output channel should be unmodified'


def test_pytorch_assumption_prune_input_channel():
    """test assumption that pytorch grouped convolution groups inputs as
    consecutive blocks by pruning an input channel"""
    z = T.nn.Conv2d(4, 2, 1, groups=2, bias=False)
    x = T.rand(1,z.in_channels,1,1) + 1
    x_zeroed_input = x.clone()
    x_zeroed_input[:,1] = 0
    expected = z(x_zeroed_input)
    assert (expected == 0).sum() == 0, 'sanity check'
    z.weight.data[0,1] = 0  # remove the corresponding input channel weights
    pred = z(x)
    assert (pred == expected).all()


def _test_prune_conv_layer__pruning_output_channel():
    convs = [
        (lambda: T.nn.Conv2d(4, 6, 1, groups=1, bias=False),
         T.tensor([10,26,42,58,74]).reshape(1,5,1,1)),
        (lambda: T.nn.Conv2d(6, 4, 1, groups=1, bias=False),
         T.tensor([21,57,93]).reshape(1,3,1,1)),
        (lambda: T.nn.Conv2d(3, 3, 1, groups=3, bias=False),
         T.tensor([1, 2]).reshape(1,2,1,1)),
        # not implemented grouped convolution tests...
        #  (lambda: T.nn.Conv2d(2, 4, 1, groups=2, bias=False),
         #  T.tensor([1, 2]).reshape(1,2,1,1)),
        #  (lambda: T.nn.Conv2d(4, 2, 1, groups=2, bias=False),
         #  T.tensor([1, 2]).reshape(1,2,1,1)),
        #  (lambda: T.nn.Conv2d(4, 6, 1, groups=2, bias=False),
         #  T.tensor([1, 2]).reshape(1,2,1,1)),
        #  (lambda: T.nn.Conv2d(6, 4, 1, groups=2, bias=False),
         #  T.tensor([1, 2]).reshape(1,2,1,1)),
    ]
    # test removing an output channel
    for _convfn, y in convs:
        # prep data
        z = _convfn()
        x = T.ones(z.in_channels).float().reshape(1,-1,1,1)
        z.weight.data[:] = T.tensor(list(range(1, 1+z.weight.data.numel()))).float().reshape(z.weight.shape)
        z.weight.data[-1] = 0
        # prune
        new_layer, pruned_in_channels, pruned_out_channels = prune_conv_layer(z)
        # evaluate
        if (z.groups == z.in_channels == z.out_channels):
            assert (pruned_in_channels[:-1] == False).all()
            assert (pruned_in_channels[-1] == True).all()
        else:
            assert (pruned_in_channels == False).all()
        assert (pruned_out_channels[:-1] == False).all()
        assert (pruned_out_channels[-1] == True).all()
        with T.no_grad():
            yhat = new_layer(x[:,~pruned_in_channels])
        #  print(z, '\n', new_layer, '\n', yhat)
        assert (y == yhat).all()


def _test_prune_conv_layer__pruning_input_channel():
    convs = [
        (lambda: T.nn.Conv2d(4, 6, 1, groups=1, bias=False),
         T.tensor([6,
                   5+6+7,
                   9+10+11,
                   13+14+15,
                   17+18+19,
                   21+22+23,
                   ]).reshape(1,6,1,1)),
        (lambda: T.nn.Conv2d(6, 4, 1, groups=1, bias=False),
         T.tensor([
             sum(range(1,6)),sum(range(7,12)),sum(range(13,18)),
             sum(range(19,24)) ]).reshape(1,4,1,1)),
        (lambda: T.nn.Conv2d(3, 3, 1, groups=3, bias=False),
         T.tensor([1, 2,0]).reshape(1,3,1,1)),
        # not implemented grouped convolution tests...
        #  (lambda: T.nn.Conv2d(2, 4, 1, groups=2, bias=False),
         #  T.tensor([1, 2]).reshape(1,2,1,1)),
        #  (lambda: T.nn.Conv2d(4, 2, 1, groups=2, bias=False),
         #  T.tensor([1, 2]).reshape(1,2,1,1)),
        #  (lambda: T.nn.Conv2d(4, 6, 1, groups=2, bias=False),
         #  T.tensor([1, 2]).reshape(1,2,1,1)),
        #  (lambda: T.nn.Conv2d(6, 4, 1, groups=2, bias=False),
         #  T.tensor([1, 2]).reshape(1,2,1,1)),
    ]
    for _convfn, y in convs:
        # prep data
        z = _convfn()
        z.weight.data[:] = T.tensor(list(range(1, 1+z.weight.data.numel()))).float().reshape(z.weight.shape)
        x = T.ones(z.in_channels).float().reshape(1,-1,1,1)
        x[:,-1] = 0
        # prune layer
        new_layer, pruned_in_channels, pruned_out_channels = prune_conv_layer(z)
        # evaluate
        assert (z(x) == new_layer(x)).all()
        assert (z(x) == y).all()
        #  print(z,'\n',new_layer, pruned_in_channels, pruned_out_channels)
        #  if not (z.groups == z.in_channels == z.out_channels):
            #  assert (pruned_in_channels == False).all()
        #  assert (pruned_out_channels[:-1] == False).all()
        #  assert pruned_out_channels[-1] == True
        #  with T.no_grad():
            #  yhat = new_layer(x[:,~pruned_in_channels])
        #  print(z, '\n', new_layer, '\n', yhat)
        #  assert (y == yhat).all()


def test_prune_conv_layer():
    _test_prune_conv_layer__pruning_output_channel()
    _test_prune_conv_layer__pruning_input_channel()


def _test_prune_model(model):
        a = (sum([x.numel() for x in model.parameters()]))
        test_inpt = T.rand(1,3,64,64)
        res1 = model(test_inpt)  # sanity check
        prune_model(model)
        b = (sum([x.numel() for x in model.parameters()]))
        try:
            res2 = model(test_inpt)  # sanity check
        except Exception as err:
            print("TEST FAILED")
            raise
        print('diff in params after pruning', a-b, 'orig', a, 'pruned', b)
        print('are outputs same?  (probably not, due to pruning batchnorm)', T.allclose(res1, res2))


if __name__ == "__main__":
    from torchvision.models import resnet18, resnet50, densenet121, DenseNet
    def test_prune_resnet():
        model = resnet18()
        model.layer1[0].conv2.weight.data[:,:2] = 0
        model.layer1[0].conv2.weight.data[:10,:] = 0
        model.layer1[0].conv1.weight.data[:,-3:] = 0
        model.layer1[0].conv1.weight.data[-8:,:] = 0
        _test_prune_model(model)
    def test_prune_densenet():
        #  model = densenet121()
        model = DenseNet(10, (1,1,2,2), 32, 6, 0, 1)
        model.eval()
        # test all variations of traversals along the dense blocks
        model.features.denseblock3.denselayer1.conv2.weight.data[-2:, :] = 0
        model.features.denseblock3.denselayer1.conv1.weight.data[:, :2] = 0
        model.features.denseblock3.denselayer1.conv1.weight.data[:2, :] = 0
        model.features.denseblock3.denselayer1.conv2.weight.data[:, -2:] = 0
        _test_prune_model(model)
    #  print('run tests')
    test_pytorch_assumption_prune_output_channel()
    test_pytorch_assumption_prune_input_channel()
    test_prune_conv_layer()
    test_prune_densenet()
    test_prune_resnet()


    model = DenseNet(10, (1,1,2,2), 32, 6, 0, 1).features.denseblock1
    model.eval()
    graph = get_graph(model)

    from matplotlib import pyplot as plt
    # pretty print the graph.
    # for small graphs.  Doesn't work well on medium to large models (too hard to see)
    #  graph = get_graph(model)
    #  # for small models?
    with open('output.png', 'wb') as fout: fout.write(nx.drawing.nx_pydot.to_pydot(graph).create_png())

    #  # for medium or larger models
    def labels(graph):
        out = {}
        for k in graph.nodes:
            v = graph.nodes[k]['module']
            if isinstance(v, T.nn.Module):
                v = f'{v.__class__.__name__}:{k}'
            elif v is None:
                v = f':{k}'
            else:
                v = f'{v.__name__}():{k}'
            out[k] = v
        return out
    fig, ax = plt.subplots(figsize=(120,120), clear=True)
    nx.draw_networkx(graph, labels=labels(graph),
                     pos=nx.nx_pydot.pydot_layout(graph), ax=ax)
    fig.savefig('output2.png', bbox_inches='tight')
    plt.close(fig)

