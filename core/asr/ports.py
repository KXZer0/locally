"""Find each graph input by what it IS, not by what one exporter called it.

FluidInference's runtime binds by name (`melspectogram`, `targets`, `h_in`,
`encoder_outputs`, ...) and those names are preferred here too, but every
role is also recognisable from rank and element type: the audio/mel input is
the float tensor, the length is the integer vector, the predictor's states
are the float rank-3 inputs. A future re-export that renames a port then
still loads, and one that genuinely changes the contract fails with a
sentence that lists what the graph actually has.
"""

import numpy as np


def describe(ports):
    return ", ".join(f"{p.get_any_name()}{p.get_partial_shape()}"
                     f":{p.get_element_type().to_dtype().name}" for p in ports)


def _named(ports, *fragments):
    for frag in fragments:
        for p in ports:
            if any(frag in n.lower() for n in p.get_names()):
                return p
    return None


def is_int(port):
    return port.get_element_type().is_integral()


def rank(port):
    return port.get_partial_shape().rank.get_length()


def signal_and_length(ports, prefer, what, graph):
    """(float signal port, integer length port or None) of a graph's inputs
    or outputs. `prefer` lists name fragments for the signal."""
    floats = [p for p in ports if not is_int(p)]
    ints = [p for p in ports if is_int(p)]
    sig = _named(floats, *prefer) or (floats[0] if floats else None)
    length = _named(ints, "length", "len") or (ints[0] if ints else None)
    if sig is None:
        raise RuntimeError(f"Could not find the {what} of {graph} "
                           f"(it has: {describe(ports)}).")
    return sig, length


def decoder_roles(inputs, outputs):
    """Predictor graph: token input, optional target_length, LSTM states."""
    ints = [p for p in inputs if is_int(p)]
    target_len = _named(ints, "target_length", "length")
    token = next((p for p in ints if p is not target_len), None)
    states = [p for p in inputs if not is_int(p)]
    # h before c, which is the order the outputs come back in for both the
    # FluidInference export (h_in/c_in) and NeMo's (input_states_1/_2).
    states.sort(key=lambda p: (0 if "h" in p.get_any_name().lower().split("_")[0]
                               else 1, p.get_any_name()))
    if token is None or not states:
        raise RuntimeError("The Parakeet decoder graph does not look like an LSTM "
                           f"predictor (inputs: {describe(inputs)}).")
    float_outs = [p for p in outputs if not is_int(p)]
    if len(float_outs) < 1 + len(states):
        raise RuntimeError("The Parakeet decoder returns fewer outputs than it "
                           f"takes states (outputs: {describe(outputs)}).")
    return token, target_len, states, float_outs[0], float_outs[1:1 + len(states)]


def joint_roles(inputs):
    if len(inputs) != 2:
        raise RuntimeError("The Parakeet joint graph should take the encoder and "
                           f"predictor outputs (it takes: {describe(inputs)}).")
    enc = _named(inputs, "enc") or inputs[0]
    dec = _named([p for p in inputs if p is not enc], "dec", "pred") or \
        next(p for p in inputs if p is not enc)
    return enc, dec


def template(port):
    """Port shape with dynamic dimensions as None."""
    return [d.get_length() if d.is_static else None for d in port.get_partial_shape()]


def fit(vec, tmpl, dtype=np.float32):
    """Reshape a flat vector into a port's shape: dynamic dims become 1,
    except the last one, which absorbs whatever is left."""
    vec = np.asarray(vec, dtype=dtype)
    if all(d is not None for d in tmpl):
        return vec.reshape(tmpl)
    dims = [1 if d is None else d for d in tmpl]
    last = max(i for i, d in enumerate(tmpl) if d is None)
    dims[last] = -1
    return vec.reshape(dims)


def zeros(port):
    return np.zeros([1 if d is None else d for d in template(port)],
                    dtype=port.get_element_type().to_dtype())


def static_shape(port, **fixed):
    """A fully static shape for NPU compilation: dims given by index in
    `fixed` (as d0=, d1=, ...), the rest 1 where dynamic."""
    shape = [1 if d is None else d for d in template(port)]
    for k, v in fixed.items():
        shape[int(k[1:])] = v
    return shape
