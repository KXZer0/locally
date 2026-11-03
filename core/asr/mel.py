"""NeMo's log-mel front end in numpy.

Only used when an export ships no preprocessor graph: the FluidInference
OpenVINO exports include one (`parakeet_melspectogram.xml`, upstream's
spelling) and the slot prefers it, because a graph exported from the model is
the model's own definition of its input. This exists so a bare
encoder/decoder/joint export is still loadable.

Settings are NeMo's `AudioToMelSpectrogramPreprocessor` defaults for Parakeet,
and the order follows onnx-asr's NumPy port, which was validated against
NeMo: 0.97 pre-emphasis, centre padding with zeros, a 400-sample symmetric
Hann window inside a 512-point FFT, power spectrum, Slaney mel filters,
log(x + 2^-24), then per-feature normalisation over the valid frames with
the unbiased variance and +1e-5 on the standard deviation.
"""

import numpy as np

SR = 16000
N_FFT = 512
WIN = 400        # 25 ms
HOP = 160        # 10 ms
PREEMPH = 0.97
LOG_GUARD = 2.0 ** -24


def _hz_to_mel(f):
    """Slaney's mel scale: linear below 1 kHz, logarithmic above."""
    f = np.asarray(f, dtype=np.float64)
    lin = f / (200.0 / 3)
    log_t = 1000.0 / (200.0 / 3)
    return np.where(f >= 1000.0, log_t + np.log(np.maximum(f, 1e-10) / 1000.0)
                    / (np.log(6.4) / 27.0), lin)


def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    log_t = 1000.0 / (200.0 / 3)
    return np.where(m >= log_t, 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - log_t)),
                    m * (200.0 / 3))


def mel_filters(n_mels, sr=SR, n_fft=N_FFT, fmin=0.0, fmax=None):
    """librosa.filters.mel(htk=False, norm='slaney'), shape (n_fft//2+1, n_mels)."""
    fmax = sr / 2 if fmax is None else fmax
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mel_pts = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_pts)
    ramps = mel_pts[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_pts[2:n_mels + 2] - mel_pts[:n_mels]))[:, None]
    return weights.T.astype(np.float32)


_FILTER_CACHE = {}


def log_mel(audio, n_mels=128):
    """float32 16 kHz mono -> (features [n_mels, frames], valid_frames)."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    n = len(audio)
    if n == 0:
        raise ValueError("No audio samples to transcribe.")
    if n_mels not in _FILTER_CACHE:
        _FILTER_CACHE[n_mels] = mel_filters(n_mels)
    fb = _FILTER_CACHE[n_mels]

    x = audio - PREEMPH * np.concatenate([[0.0], audio[:-1]]).astype(np.float32)
    x = np.pad(x, (N_FFT // 2, N_FFT // 2))
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP]
    side = (N_FFT - WIN) // 2
    window = np.pad(np.hanning(WIN), (side, side)).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames * window, N_FFT)).astype(np.float32) ** 2
    feats = np.log(spec @ fb + LOG_GUARD)                    # [frames, n_mels]

    valid = max(1, n // HOP)
    v = feats[:valid]
    mean = v.mean(axis=0, keepdims=True)
    var = ((v - mean) ** 2).sum(axis=0, keepdims=True) / max(valid - 1, 1)
    feats = (feats - mean) / (np.sqrt(var) + 1e-5)
    feats[valid:] = 0.0
    return feats.T.astype(np.float32), valid
