"""Speech-to-text engines that are not a genai pipeline.

    layout  which ASR engine a directory holds, and where its files are
    tdt     Token-and-Duration Transducer greedy decoding (pure Python)
    vocab   SentencePiece vocabulary and detokenisation
    mel     NeMo log-mel features in numpy, for exports without a preprocessor

Whisper keeps its genai pipeline (core/slots/whisper.py). Parakeet has no genai
pipeline at all, so its slot (core/slots/parakeet.py) drives the plain runtime
and the decoding loop lives here, where it can be tested without a model.
"""
