"""Ports and static shapes of a Nemotron-family export (core/slots/nemotron.py).

Port names are eddy-audio's (src/models/nemotron/nemotron_openvino.cpp); the
static shapes for the NPU come from metadata.json (core/asr/geometry.py),
because the NPU compiles static shapes only and a cache-aware encoder's
cache tensors have no other source for their size.
"""

import openvino as ov
import openvino.opset1 as op1

from core.asr import ports as P


def port(ports, name):
    return next((p for p in ports if name in p.get_names()), None)


def npu_safe(model):
    """BitwiseNot(bool) -> LogicalNot, as eddy does before every compile.

    The NPU plugin compiles BitwiseNot on a boolean as an integer complement
    (~0 = -1, ~1 = -2, both "true"), so the FastConformer attention mask goes
    all-true and every transcript comes back empty. LogicalNot is the same
    operation on booleans and compiles correctly; on CPU/GPU it is a no-op
    change. Returns how many nodes were rewritten."""
    n = 0
    for node in model.get_ordered_ops():
        if (node.get_type_name() == "BitwiseNot"
                and node.get_input_element_type(0) == ov.Type.boolean):
            repl = op1.logical_not(node.input_value(0))
            repl.set_friendly_name(node.get_friendly_name())
            node.output(0).replace(repl.output(0))
            n += 1
    if n:
        model.validate_nodes_and_infer_types()
    return n


def static_encoder_shapes(enc, geo):
    width = geo.window_mel if geo.mode == "window" else geo.total_mel
    shapes = {}
    for p in enc.inputs:
        names = p.get_names()
        if "cache_channel" in names:
            shapes[p.get_any_name()] = geo.cache_channel
        elif "cache_time" in names:
            shapes[p.get_any_name()] = geo.cache_time
        elif not P.is_int(p) and P.rank(p) == 3:
            shapes[p.get_any_name()] = [1, geo.n_mels, width]
        else:
            shapes[p.get_any_name()] = P.static_shape(p)
    return shapes

def static_small_shapes(model, geo):
    state = [geo.decoder_layers, 1, geo.decoder_hidden]
    return {p.get_any_name(): (state if p.get_names() & {"h_in", "c_in"}
                               else P.static_shape(p)) for p in model.inputs}

def bind(enc, dec, joint, geo):
    """Every port the slot feeds or reads, by eddy's names first and by role
    where a re-export renamed one; a contract that genuinely differs fails
    with the ports the graph actually has."""
    e_in, e_out = enc.inputs, enc.outputs
    io = {}
    io["mel"] = port(e_in, "mel")
    io["mel_len"] = port(e_in, "mel_length")
    if io["mel"] is None:
        plain = [p for p in e_in if not p.get_names() & {
            "cache_channel", "cache_time", "cache_len", "prompt_id"}]
        io["mel"], io["mel_len"] = P.signal_and_length(
            plain, ("mel", "audio_signal"), "mel input", "the encoder")
    io["encoded"] = port(e_out, "encoded") or next(
        p for p in e_out if not P.is_int(p) and "cache" not in p.get_any_name())
    if geo.mode == "cache":
        for k in ("cache_channel", "cache_time", "cache_len"):
            io[k], io[k + "_out"] = port(e_in, k), port(e_out, k + "_out")
            if io[k] is None or io[k + "_out"] is None:
                raise RuntimeError(f"The cache-aware encoder has no {k} / "
                                   f"{k}_out port (it has: {P.describe(e_in)}"
                                   f" -> {P.describe(e_out)}).")
    io["prompt"] = port(e_in, "prompt_id")
    d_in, d_out = dec.inputs, dec.outputs
    named = [port(d_in, n) for n in ("token", "token_length", "h_in", "c_in")]
    outs = [port(d_out, n) for n in ("decoder_out", "h_out", "c_out")]
    if all(p is not None for p in named[0:1] + named[2:] + outs):
        io["tok"], io["tok_len"] = named[0], named[1]
        io["states"], io["dec_out"], io["state_outs"] = named[2:], outs[0], outs[1:]
    else:
        (io["tok"], io["tok_len"], io["states"], io["dec_out"],
         io["state_outs"]) = P.decoder_roles(d_in, d_out)
    io["j_enc"], io["j_dec"] = P.joint_roles(joint.inputs)
    io["j_enc_t"], io["j_dec_t"] = P.template(io["j_enc"]), P.template(io["j_dec"])
    io["logits"] = port(joint.outputs, "logits") or joint.outputs[0]
    return io
