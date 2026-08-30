"""locally's internals, split by what each part is FOR.

locally.py stays the entry point and the Flask app; everything a person
might want to read on its own lives here.

    models/    what a model directory is, and what is inside it
    hardware/  what the machine has, and what a device can be given
    genai/     reading openvino-genai results and translating its errors
    tools/     rendering tool schemas, and parsing calls back out
    sandbox/   running one calculation in a killed child process
    media/     images in, tensors out
    web/       fetching a page and reducing it to Markdown
"""
