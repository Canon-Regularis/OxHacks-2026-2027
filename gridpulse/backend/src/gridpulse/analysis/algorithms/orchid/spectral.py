"""
Spectral Analysis & DSP -- a self-contained, zero-dependency toolkit
====================================================================

**Everything a signal has to say, it says in the frequency domain.**  Vibration
monitoring, audio, communications, radar, EEG/ECG, power systems, seismology --
they all live and die by the spectrum.  This module is a from-scratch digital
signal-processing and spectral-analysis stack built on the Python standard
library alone: the FFT (radix-2 *and* Bluestein for any length), the whole window
zoo, non-parametric PSD estimation (periodogram / Welch / multitaper with genuine
Slepian tapers), parametric spectra (Yule-Walker / Burg AR, MUSIC, Capon),
time-frequency analysis (STFT / spectrogram / iSTFT), FIR & IIR filter design and
zero-phase filtering, the analytic signal (Hilbert), wavelets (DWT / CWT),
correlation, a battery of spectral features and a signal generator.  No NumPy, no
SciPy: the FFT, the Bessel functions, the tridiagonal eigensolver and the bilinear
transform are all built by hand.  Copy ``spectral_template.py`` into a competition
/ hackathon / interview / embedded sandbox and go.

Layers
------
1.  **Transforms** -- ``fft`` / ``ifft`` (radix-2 + Bluestein, any N), ``rfft`` /
    ``irfft``, ``dft``, ``czt`` (chirp-z), ``fftconvolve``, frequency axes, shifts.
2.  **Windows** -- Hann/Hamming/Blackman(-Harris)/Bartlett/Kaiser/Tukey/Gaussian/
    flat-top/Nuttall + ``get_window`` and window metrics (ENBW, coherent gain).
3.  **Non-parametric PSD** -- ``periodogram``, ``welch``, ``csd``, ``coherence``.
4.  **Time-frequency** -- ``stft`` / ``istft`` / ``spectrogram``.
5.  **Parametric spectra** -- ``levinson_durbin``, AR by Yule-Walker & ``burg``,
    ``ar_psd``, ``music`` and ``capon`` (minimum-variance) line-spectrum estimators.
6.  **Multitaper** -- genuine ``dpss`` Slepian sequences (tridiagonal eigensolver)
    and the ``multitaper`` PSD.
7.  **Filtering** -- ``firwin`` (windowed-sinc, all four bands), ``butter`` (bilinear
    biquads), ``lfilter`` / ``filtfilt`` (zero-phase), ``freqz``, ``group_delay``,
    ``hilbert`` (analytic signal) and the envelope.
8.  **Correlation** -- ``convolve`` / ``correlate`` / ``autocorrelate`` (FFT-fast).
9.  **Wavelets** -- Haar / Daubechies ``dwt`` / ``wavedec`` and Morlet ``cwt``.
10. **Features** -- spectral centroid / bandwidth / flatness / rolloff / entropy /
    flux / crest, ``dominant_frequency``, ``thd``, ``snr``, ``find_peaks``.
11. **Utilities & generators** -- detrend / resample / unwrap / dB, and sine / chirp
    / square / sawtooth / impulse / white & pink noise.

Conventions
-----------
* A real signal is a ``list`` of floats; a spectrum is a ``list`` of ``complex``.
* Sampling frequency ``fs`` is in Hz (default 1.0 -> normalised to cycles/sample).
* PSD is one-sided for real input by default, scaled as a density (units²/Hz).
* Everything is deterministic; noise generators and randomised helpers take a ``seed``.
"""

from __future__ import annotations

import cmath
import math
import random
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple)

__version__ = "1.0.0"

Signal = List[float]
Spectrum = List[complex]

_EPS = 1e-12
_TWO_PI = 2.0 * math.pi

__all__ = [
    # transforms
    "fft", "ifft", "rfft", "irfft", "dft", "idft", "czt", "fftconvolve",
    "fftfreq", "rfftfreq", "fftshift", "ifftshift", "next_pow2", "next_fast_len",
    # windows
    "get_window", "boxcar", "hann", "hamming", "blackman", "blackmanharris",
    "bartlett", "kaiser", "tukey", "gaussian", "flattop", "nuttall",
    "window_enbw", "coherent_gain",
    # non-parametric psd
    "periodogram", "welch", "csd", "coherence",
    # time-frequency
    "stft", "istft", "spectrogram",
    # parametric
    "levinson_durbin", "ar_yule_walker", "burg", "ar_psd", "music", "capon",
    # multitaper
    "dpss", "multitaper",
    # filtering
    "firwin", "butter", "lfilter", "filtfilt", "freqz", "group_delay",
    "hilbert", "envelope", "sosfilt",
    # correlation
    "convolve", "correlate", "autocorrelate",
    # wavelets
    "dwt", "idwt", "wavedec", "waverec", "cwt", "morlet",
    # features
    "spectral_centroid", "spectral_bandwidth", "spectral_flatness",
    "spectral_rolloff", "spectral_entropy", "spectral_flux", "spectral_crest",
    "dominant_frequency", "thd", "snr", "find_peaks",
    # utilities & generators
    "detrend", "resample", "unwrap", "mag2db", "db2mag", "rms", "zero_pad",
    "sine_wave", "chirp", "square_wave", "sawtooth", "impulse", "white_noise",
    "pink_noise",
]


# ===========================================================================
# 0. VALIDATION HELPERS
# ===========================================================================
class SpectralError(ValueError):
    """Invalid input to a spectral routine."""


def _as_floats(x: Sequence[Any], name: str = "x") -> Signal:
    try:
        out = [float(v) for v in x]
    except (TypeError, ValueError):
        raise SpectralError("%s must be a sequence of real numbers." % name)
    if any(math.isnan(v) or math.isinf(v) for v in out):
        raise SpectralError("%s contains non-finite values." % name)
    return out


def _as_complex(x: Sequence[Any]) -> Spectrum:
    return [complex(v) for v in x]


def _check_len(x: Sequence[Any], min_len: int = 1, name: str = "x") -> None:
    if len(x) < min_len:
        raise SpectralError("%s needs at least %d samples (got %d)."
                            % (name, min_len, len(x)))


# ===========================================================================
# 1. TRANSFORMS -- FFT (radix-2 + Bluestein), rFFT, DFT, chirp-z
# ===========================================================================
def _fft_pow2(a: Spectrum, inverse: bool = False) -> Spectrum:
    """Iterative in-place radix-2 Cooley-Tukey FFT (``len(a)`` a power of two)."""
    n = len(a)
    a = list(a)
    j = 0
    for i in range(1, n):                       # bit-reversal permutation
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    sign = 1.0 if inverse else -1.0
    length = 2
    while length <= n:
        wlen = cmath.exp(sign * 2j * math.pi / length)
        half = length >> 1
        for i in range(0, n, length):
            w = 1.0 + 0.0j
            for k in range(half):
                u = a[i + k]
                v = a[i + k + half] * w
                a[i + k] = u + v
                a[i + k + half] = u - v
                w *= wlen
        length <<= 1
    if inverse:
        a = [x / n for x in a]
    return a


def _bluestein(a: Spectrum, inverse: bool = False) -> Spectrum:
    """Bluestein's chirp-z FFT for arbitrary length (O(N log N))."""
    n = len(a)
    sign = 1.0 if inverse else -1.0
    # chirp c_k = exp(sign * i * pi * k^2 / n); use k^2 mod 2n for float precision
    c = [cmath.exp(sign * 1j * math.pi * ((k * k) % (2 * n)) / n) for k in range(n)]
    m = 1
    while m < 2 * n - 1:
        m <<= 1
    aa = [a[k] * c[k] for k in range(n)] + [0j] * (m - n)
    b = [0j] * m
    b[0] = c[0].conjugate()
    for k in range(1, n):
        b[k] = c[k].conjugate()
        b[m - k] = c[k].conjugate()
    fa = _fft_pow2(aa)
    fb = _fft_pow2(b)
    conv = _fft_pow2([fa[i] * fb[i] for i in range(m)], inverse=True)
    out = [c[k] * conv[k] for k in range(n)]
    if inverse:
        out = [v / n for v in out]
    return out


def fft(x: Sequence[Any]) -> Spectrum:
    """Discrete Fourier transform (any length; unnormalised, NumPy convention)."""
    a = _as_complex(x)
    n = len(a)
    if n == 0:
        return []
    if n & (n - 1) == 0:
        return _fft_pow2(a)
    return _bluestein(a)


def ifft(x: Sequence[Any]) -> Spectrum:
    """Inverse DFT (any length; normalised by 1/N)."""
    a = _as_complex(x)
    n = len(a)
    if n == 0:
        return []
    if n & (n - 1) == 0:
        return _fft_pow2(a, inverse=True)
    return _bluestein(a, inverse=True)


def rfft(x: Sequence[float]) -> Spectrum:
    """FFT of a real signal, returning the ``N//2 + 1`` non-redundant bins."""
    n = len(x)
    full = fft(_as_floats(x))
    return full[:n // 2 + 1]


def irfft(x: Sequence[Any], n: Optional[int] = None) -> Signal:
    """Inverse of :func:`rfft`; reconstruct the length-``n`` real signal."""
    half = _as_complex(x)
    if n is None:
        n = 2 * (len(half) - 1)
    full = [0j] * n
    for k in range(len(half)):
        if k < n:
            full[k] = half[k]
    for k in range(1, (n + 1) // 2):
        if k < len(half):
            full[n - k] = half[k].conjugate()
    return [v.real for v in ifft(full)]


def dft(x: Sequence[Any]) -> Spectrum:
    """Direct O(N^2) DFT (reference implementation / any length)."""
    a = _as_complex(x)
    n = len(a)
    return [sum(a[t] * cmath.exp(-2j * math.pi * k * t / n) for t in range(n))
            for k in range(n)]


def idft(x: Sequence[Any]) -> Spectrum:
    """Direct O(N^2) inverse DFT."""
    a = _as_complex(x)
    n = len(a)
    return [sum(a[k] * cmath.exp(2j * math.pi * k * t / n) for k in range(n)) / n
            for t in range(n)]


def czt(x: Sequence[Any], m: Optional[int] = None, w: Optional[complex] = None,
        a: complex = 1.0 + 0j) -> Spectrum:
    """Chirp-z transform: sample the z-plane along ``a * w^{-k}`` for ``k=0..m-1``
    (a spiral contour).  With the defaults it is the DFT on a finer/zoomed grid."""
    xs = _as_complex(x)
    n = len(xs)
    if m is None:
        m = n
    if w is None:
        w = cmath.exp(-2j * math.pi / m)
    # Bluestein-style: y_k = sum_n x_n a^{-n} w^{nk}
    g = [xs[k] * (a ** (-k)) * (w ** ((k * k) / 2.0)) for k in range(n)]
    L = 1
    while L < n + m - 1:
        L <<= 1
    gg = g + [0j] * (L - n)
    h = [0j] * L
    for k in range(m):
        h[k] = w ** (-(k * k) / 2.0)
    for k in range(1, n):
        h[L - k] = w ** (-(k * k) / 2.0)
    Fg = _fft_pow2(gg)                                    # transform each ONCE, not per-bin
    Fh = _fft_pow2(h)
    conv = _fft_pow2([Fg[i] * Fh[i] for i in range(L)], inverse=True)
    return [(w ** ((k * k) / 2.0)) * conv[k] for k in range(m)]


def fftconvolve(a: Sequence[float], b: Sequence[float], mode: str = "full") -> Signal:
    """Linear convolution of two real signals via the FFT (O(N log N))."""
    x = _as_floats(a, "a")
    y = _as_floats(b, "b")
    if not x or not y:
        return []
    n = len(x) + len(y) - 1
    L = next_pow2(n)
    fx = fft(x + [0.0] * (L - len(x)))
    fy = fft(y + [0.0] * (L - len(y)))
    full = [v.real for v in ifft([fx[i] * fy[i] for i in range(L)])][:n]
    if mode == "full":
        return full
    if mode == "same":
        start = (len(y) - 1) // 2
        return full[start:start + len(x)]
    if mode == "valid":
        lo = min(len(x), len(y)) - 1
        hi = max(len(x), len(y))
        return full[lo:hi]
    raise SpectralError("mode must be 'full', 'same' or 'valid'.")


def fftfreq(n: int, fs: float = 1.0) -> Signal:
    """Two-sided FFT bin frequencies (Hz), matching :func:`fft`'s bin order."""
    df = fs / n
    return [(k if k <= (n - 1) // 2 else k - n) * df for k in range(n)]


def rfftfreq(n: int, fs: float = 1.0) -> Signal:
    """One-sided (non-negative) FFT bin frequencies for :func:`rfft`."""
    df = fs / n
    return [k * df for k in range(n // 2 + 1)]


def fftshift(x: Sequence[Any]) -> List[Any]:
    """Shift the zero-frequency component to the centre of the spectrum."""
    n = len(x)
    p = (n + 1) // 2
    return list(x[p:]) + list(x[:p])


def ifftshift(x: Sequence[Any]) -> List[Any]:
    """Inverse of :func:`fftshift`."""
    n = len(x)
    p = n // 2
    return list(x[p:]) + list(x[:p])


def next_pow2(n: int) -> int:
    """Smallest power of two >= ``n``."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def next_fast_len(n: int) -> int:
    """Smallest 5-smooth (2,3,5) length >= ``n`` for efficient FFT sizing."""
    if n <= 1:
        return 1
    best = next_pow2(n)
    a = 1
    while a < best:
        b = a
        while b < best:
            c = b
            while c < best:
                if c >= n:
                    best = min(best, c)
                c *= 5
            b *= 3
        a *= 2
    return best


# ===========================================================================
# 2. WINDOWS
# ===========================================================================
def _bessel_i0(x: float) -> float:
    """Modified Bessel function I0 (series), for the Kaiser window."""
    s = 1.0
    term = 1.0
    k = 1
    half = x / 2.0
    while True:
        term *= (half / k) ** 2
        s += term
        if term < 1e-14 * s:
            break
        k += 1
        if k > 1000:
            break
    return s


def boxcar(n: int) -> Signal:
    return [1.0] * n


def _sym_index(n: int, sym: bool) -> Tuple[int, int]:
    # window generators use M = n-1 for symmetric (default) windows
    return n, (n - 1 if n > 1 else 1)


def hann(n: int) -> Signal:
    if n == 1:
        return [1.0]
    M = n - 1
    return [0.5 - 0.5 * math.cos(_TWO_PI * i / M) for i in range(n)]


def hamming(n: int) -> Signal:
    if n == 1:
        return [1.0]
    M = n - 1
    return [0.54 - 0.46 * math.cos(_TWO_PI * i / M) for i in range(n)]


def blackman(n: int) -> Signal:
    if n == 1:
        return [1.0]
    M = n - 1
    return [0.42 - 0.5 * math.cos(_TWO_PI * i / M)
            + 0.08 * math.cos(2 * _TWO_PI * i / M) for i in range(n)]


def blackmanharris(n: int) -> Signal:
    if n == 1:
        return [1.0]
    M = n - 1
    a = [0.35875, 0.48829, 0.14128, 0.01168]
    return [a[0] - a[1] * math.cos(_TWO_PI * i / M)
            + a[2] * math.cos(2 * _TWO_PI * i / M)
            - a[3] * math.cos(3 * _TWO_PI * i / M) for i in range(n)]


def nuttall(n: int) -> Signal:
    if n == 1:
        return [1.0]
    M = n - 1
    a = [0.3635819, 0.4891775, 0.1365995, 0.0106411]
    return [a[0] - a[1] * math.cos(_TWO_PI * i / M)
            + a[2] * math.cos(2 * _TWO_PI * i / M)
            - a[3] * math.cos(3 * _TWO_PI * i / M) for i in range(n)]


def flattop(n: int) -> Signal:
    if n == 1:
        return [1.0]
    M = n - 1
    a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
    return [a[0] - a[1] * math.cos(_TWO_PI * i / M)
            + a[2] * math.cos(2 * _TWO_PI * i / M)
            - a[3] * math.cos(3 * _TWO_PI * i / M)
            + a[4] * math.cos(4 * _TWO_PI * i / M) for i in range(n)]


def bartlett(n: int) -> Signal:
    if n == 1:
        return [1.0]
    M = n - 1
    return [1.0 - abs((i - M / 2.0) / (M / 2.0)) for i in range(n)]


def kaiser(n: int, beta: float = 8.6) -> Signal:
    if n == 1:
        return [1.0]
    M = n - 1
    denom = _bessel_i0(beta)
    return [_bessel_i0(beta * math.sqrt(1.0 - ((2.0 * i / M - 1.0) ** 2))) / denom
            for i in range(n)]


def gaussian(n: int, std: float = 0.4) -> Signal:
    if n == 1:
        return [1.0]
    M = n - 1
    sig = std * M / 2.0 if std < 1.0 else std
    center = M / 2.0
    return [math.exp(-0.5 * ((i - center) / sig) ** 2) for i in range(n)]


def tukey(n: int, alpha: float = 0.5) -> Signal:
    if n == 1:
        return [1.0]
    if alpha <= 0:
        return boxcar(n)
    if alpha >= 1:
        return hann(n)
    M = n - 1
    w = []
    edge = alpha * M / 2.0
    for i in range(n):
        if i < edge:
            w.append(0.5 * (1 + math.cos(math.pi * (2 * i / (alpha * M) - 1))))
        elif i <= M - edge:
            w.append(1.0)
        else:
            w.append(0.5 * (1 + math.cos(math.pi * (2 * i / (alpha * M) - 2 / alpha + 1))))
    return w


_WINDOWS: Dict[str, Callable[..., Signal]] = {
    "boxcar": boxcar, "rectangular": boxcar, "rect": boxcar, "ones": boxcar,
    "hann": hann, "hanning": hann, "hamming": hamming, "blackman": blackman,
    "blackmanharris": blackmanharris, "nuttall": nuttall, "flattop": flattop,
    "bartlett": bartlett, "triangular": bartlett, "kaiser": kaiser,
    "gaussian": gaussian, "tukey": tukey,
}


def get_window(window: Any, n: int) -> Signal:
    """Resolve ``window`` (name string, ``(name, param)`` tuple, or explicit list)
    into a length-``n`` window."""
    if isinstance(window, (list, tuple)) and window and not isinstance(window[0], str):
        w = _as_floats(window, "window")
        if len(w) != n:
            raise SpectralError("explicit window length %d != %d." % (len(w), n))
        return w
    if isinstance(window, tuple):
        name, *params = window
        return _WINDOWS[str(name).lower()](n, *params)
    return _WINDOWS[str(window).lower()](n)


def window_enbw(w: Sequence[float]) -> float:
    """Equivalent noise bandwidth (in bins) of a window: ``N * sum(w^2)/sum(w)^2``."""
    w = _as_floats(w, "w")
    s1 = math.fsum(w)
    s2 = math.fsum(v * v for v in w)
    return len(w) * s2 / (s1 * s1) if abs(s1) > _EPS else float("inf")


def coherent_gain(w: Sequence[float]) -> float:
    """Coherent gain (DC amplitude response) of a window: ``mean(w)``."""
    w = _as_floats(w, "w")
    return math.fsum(w) / len(w)


def _spectral_window(window: Any, n: int) -> Signal:
    """Periodic (DFT-even) window for spectral ESTIMATION -- the correct form for the
    periodogram / Welch / STFT (matches SciPy's ``fftbins=True``).  It is the
    symmetric window of length ``n+1`` with the duplicated endpoint dropped, which
    removes the small bias the symmetric window's repeated endpoint would introduce
    into a PSD.  (Filter design -- :func:`firwin` -- keeps the symmetric window.)
    An explicit numeric window is used verbatim."""
    if isinstance(window, (list, tuple)) and window and not isinstance(window[0], str):
        return get_window(window, n)
    return get_window(window, n + 1)[:n]


# ===========================================================================
# 3. NON-PARAMETRIC PSD
# ===========================================================================
def _median_bias(n: int) -> float:
    """Bias factor correcting the median of ``n`` exponential-distributed periodograms
    into an unbiased estimator of their mean PSD (matches SciPy's ``_median_bias``;
    -> 1/ln2 as n -> inf).  Without it, median-averaged Welch PSD is biased low."""
    if n <= 1:
        return 1.0
    return 1.0 + math.fsum(1.0 / (2.0 * i + 1.0) - 1.0 / (2.0 * i)
                           for i in range(1, (n - 1) // 2 + 1))


def _psd_scale(w: Signal, fs: float, scaling: str) -> float:
    if scaling == "density":
        return 1.0 / (fs * math.fsum(v * v for v in w))
    if scaling == "spectrum":
        s1 = math.fsum(w)
        return 1.0 / (s1 * s1) if abs(s1) > _EPS else 0.0
    raise SpectralError("scaling must be 'density' or 'spectrum'.")


def periodogram(x: Sequence[float], fs: float = 1.0, window: Any = "boxcar",
                detrend_type: str = "constant", scaling: str = "density",
                onesided: bool = True) -> Tuple[Signal, Signal]:
    """Periodogram PSD estimate.  Returns ``(freqs, Pxx)``."""
    xs = detrend(_as_floats(x), detrend_type)
    n = len(xs)
    _check_len(xs, 1)
    w = _spectral_window(window, n)
    xw = [xs[i] * w[i] for i in range(n)]
    X = fft(xw)
    scale = _psd_scale(w, fs, scaling)
    Pxx = [(X[k].real ** 2 + X[k].imag ** 2) * scale for k in range(n)]
    if onesided:
        m = n // 2 + 1
        P = list(Pxx[:m])
        for k in range(1, m - (1 if n % 2 == 0 else 0)):
            P[k] *= 2.0                          # fold negative-frequency power
        return rfftfreq(n, fs), P
    return fftfreq(n, fs), Pxx


def welch(x: Sequence[float], fs: float = 1.0, window: Any = "hann",
          nperseg: int = 256, noverlap: Optional[int] = None,
          detrend_type: str = "constant", scaling: str = "density",
          onesided: bool = True, average: str = "mean"
          ) -> Tuple[Signal, Signal]:
    """Welch's averaged-periodogram PSD.  Returns ``(freqs, Pxx)``."""
    xs = _as_floats(x)
    n = len(xs)
    if nperseg > n:
        nperseg = n
    if noverlap is None:
        noverlap = nperseg // 2
    step = nperseg - noverlap
    if step <= 0:
        raise SpectralError("noverlap must be < nperseg.")
    w = _spectral_window(window, nperseg)
    scale = _psd_scale(w, fs, scaling)
    segs = []
    start = 0
    while start + nperseg <= n:
        seg = detrend(xs[start:start + nperseg], detrend_type)
        xw = [seg[i] * w[i] for i in range(nperseg)]
        X = fft(xw)
        segs.append([(X[k].real ** 2 + X[k].imag ** 2) * scale for k in range(nperseg)])
        start += step
    if not segs:                                 # signal shorter than one segment
        return periodogram(xs, fs, window, detrend_type, scaling, onesided)
    if average == "median":
        bias = _median_bias(len(segs))                   # unbias the median (as SciPy does)
        Pxx = [_median([s[k] for s in segs]) / bias for k in range(nperseg)]
    elif average == "mean":
        Pxx = [math.fsum(s[k] for s in segs) / len(segs) for k in range(nperseg)]
    else:
        raise SpectralError("average must be 'mean' or 'median'.")
    if onesided:
        m = nperseg // 2 + 1
        P = list(Pxx[:m])
        for k in range(1, m - (1 if nperseg % 2 == 0 else 0)):
            P[k] *= 2.0
        return rfftfreq(nperseg, fs), P
    return fftfreq(nperseg, fs), Pxx


def csd(x: Sequence[float], y: Sequence[float], fs: float = 1.0,
        window: Any = "hann", nperseg: int = 256, noverlap: Optional[int] = None
        ) -> Tuple[Signal, Spectrum]:
    """Cross power spectral density ``Pxy`` (Welch-averaged, one-sided)."""
    xs, ys = _as_floats(x, "x"), _as_floats(y, "y")
    n = min(len(xs), len(ys))
    xs, ys = xs[:n], ys[:n]
    if nperseg > n:
        nperseg = n
    if noverlap is None:
        noverlap = nperseg // 2
    step = nperseg - noverlap
    w = _spectral_window(window, nperseg)
    scale = 1.0 / (fs * math.fsum(v * v for v in w))
    m = nperseg // 2 + 1
    acc = [0j] * m
    cnt = 0
    start = 0
    while start + nperseg <= n:
        sx = detrend(xs[start:start + nperseg], "constant")
        sy = detrend(ys[start:start + nperseg], "constant")
        X = fft([sx[i] * w[i] for i in range(nperseg)])
        Y = fft([sy[i] * w[i] for i in range(nperseg)])
        for k in range(m):
            acc[k] += X[k].conjugate() * Y[k] * scale
        cnt += 1
        start += step
    if cnt == 0:
        cnt = 1
    Pxy = [acc[k] / cnt for k in range(m)]
    for k in range(1, m - (1 if nperseg % 2 == 0 else 0)):
        Pxy[k] *= 2.0
    return rfftfreq(nperseg, fs), Pxy


def coherence(x: Sequence[float], y: Sequence[float], fs: float = 1.0,
              window: Any = "hann", nperseg: int = 256,
              noverlap: Optional[int] = None) -> Tuple[Signal, Signal]:
    """Magnitude-squared coherence ``|Pxy|^2 / (Pxx Pyy)`` in ``[0, 1]``."""
    f, Pxy = csd(x, y, fs, window, nperseg, noverlap)
    _, Pxx = welch(x, fs, window, nperseg, noverlap)
    _, Pyy = welch(y, fs, window, nperseg, noverlap)
    # coherence is a normalised (scale-invariant) quantity, so the "empty bin" cutoff
    # must be RELATIVE to the spectral level -- an absolute floor wrongly zeroed the
    # coherence of small-amplitude (e.g. microvolt) signals.
    ref = max((Pxx[k] * Pyy[k] for k in range(len(f))), default=0.0)
    thresh = _EPS * ref
    Cxy = []
    for k in range(len(f)):
        denom = Pxx[k] * Pyy[k]
        num = Pxy[k].real ** 2 + Pxy[k].imag ** 2
        Cxy.append(min(1.0, num / denom) if denom > thresh else 0.0)
    return f, Cxy


# ===========================================================================
# 4. TIME-FREQUENCY (STFT / spectrogram)
# ===========================================================================
def stft(x: Sequence[float], fs: float = 1.0, window: Any = "hann",
         nperseg: int = 256, noverlap: Optional[int] = None,
         detrend_type: str = "none") -> Tuple[Signal, Signal, List[Spectrum]]:
    """Short-time Fourier transform.  Returns ``(freqs, times, Zxx)`` where
    ``Zxx[frame]`` is the one-sided spectrum of that frame.  ``detrend_type`` defaults
    to ``'none'`` so the transform stays invertible by :func:`istft`; the
    :func:`spectrogram` uses ``'constant'`` to match the Welch PSD."""
    xs = _as_floats(x)
    n = len(xs)
    if nperseg > n:
        nperseg = n
    if noverlap is None:
        noverlap = nperseg // 2
    step = nperseg - noverlap
    if step <= 0:
        raise SpectralError("noverlap must be < nperseg.")
    w = _spectral_window(window, nperseg)
    m = nperseg // 2 + 1
    frames, times = [], []
    start = 0
    while start + nperseg <= n:
        seg = xs[start:start + nperseg]
        if detrend_type not in ("none", None):
            seg = detrend(seg, detrend_type)
        X = rfft([seg[i] * w[i] for i in range(nperseg)])
        frames.append(X)
        times.append((start + nperseg / 2.0) / fs)
        start += step
    return rfftfreq(nperseg, fs), times, frames


def istft(Zxx: List[Spectrum], fs: float = 1.0, window: Any = "hann",
          nperseg: int = 256, noverlap: Optional[int] = None,
          nfft_time: Optional[int] = None) -> Signal:
    """Inverse STFT via weighted overlap-add (reconstructs the signal)."""
    if not Zxx:
        return []
    if noverlap is None:
        noverlap = nperseg // 2
    step = nperseg - noverlap
    w = _spectral_window(window, nperseg)
    total = (len(Zxx) - 1) * step + nperseg
    out = [0.0] * total
    wsum = [0.0] * total
    for f, spec in enumerate(Zxx):
        seg = irfft(spec, nperseg)
        base = f * step
        for i in range(nperseg):
            out[base + i] += seg[i] * w[i]
            wsum[base + i] += w[i] * w[i]
    rec = [out[i] / wsum[i] if wsum[i] > _EPS else 0.0 for i in range(total)]
    return rec[:nfft_time] if nfft_time else rec


def spectrogram(x: Sequence[float], fs: float = 1.0, window: Any = "hann",
                nperseg: int = 256, noverlap: Optional[int] = None,
                scaling: str = "density") -> Tuple[Signal, Signal, List[Signal]]:
    """Power spectrogram.  Returns ``(freqs, times, Sxx)`` with ``Sxx[frame]`` the
    one-sided power spectrum of that frame."""
    f, t, Z = stft(x, fs, window, nperseg, noverlap, detrend_type="constant")
    npseg = nperseg if nperseg <= len(x) else len(x)
    w = _spectral_window(window, npseg)
    scale = _psd_scale(w, fs, scaling)
    Sxx = []
    for spec in Z:
        P = [(c.real ** 2 + c.imag ** 2) * scale for c in spec]
        for k in range(1, len(P) - (1 if npseg % 2 == 0 else 0)):   # fold by TRUE parity:
            P[k] *= 2.0                                             # odd nperseg has no Nyquist
        Sxx.append(P)
    return f, t, Sxx


# ===========================================================================
# 5. PARAMETRIC SPECTRA (AR: Yule-Walker / Burg; MUSIC; Capon)
# ===========================================================================
def _autocorr(x: Signal, maxlag: int) -> Signal:
    n = len(x)
    return [math.fsum(x[t] * x[t - k] for t in range(k, n)) / n
            for k in range(maxlag + 1)]


def levinson_durbin(r: Sequence[float], order: int
                    ) -> Tuple[Signal, float, Signal]:
    """Solve the Toeplitz Yule-Walker system from autocovariances ``r`` -> AR
    coefficients ``a`` (length ``order``), noise variance ``sigma2`` and the
    reflection (PARCOR) coefficients."""
    r = _as_floats(r, "r")
    a = [0.0] * (order + 1)
    r0 = r[0]
    E = r0
    refl = []
    if E <= 0.0:                                 # zero power (silence) -> no model
        return [0.0] * order, 0.0, []
    for m in range(1, order + 1):
        acc = r[m] - math.fsum(a[j] * r[m - j] for j in range(1, m))
        k = acc / E if E > 1e-300 else 0.0       # reflection coeff (acc/E is scale-invariant)
        new = list(a)
        new[m] = k
        for j in range(1, m):
            new[j] = a[j] - k * a[m - j]
        a = new
        E *= (1.0 - k * k)
        refl.append(k)
        if E <= 0:
            E = r0 * 1e-20                        # tiny positive, PROPORTIONAL to power
            break
    return a[1:order + 1], E, refl


def ar_yule_walker(x: Sequence[float], order: int) -> Tuple[Signal, float]:
    """AR(p) parameters by the Yule-Walker equations.  Returns ``(a, sigma2)`` for
    the model ``x[t] = sum a_i x[t-i] + e[t]``."""
    xs = detrend(_as_floats(x), "constant")
    _check_len(xs, order + 1)
    r = _autocorr(xs, order)
    a, sigma2, _ = levinson_durbin(r, order)
    return a, sigma2


def burg(x: Sequence[float], order: int) -> Tuple[Signal, float]:
    """AR(p) parameters by Burg's method (maximum-entropy; low-bias, high-resolution).
    Returns ``(a, sigma2)``."""
    xs = detrend(_as_floats(x), "constant")
    n = len(xs)
    _check_len(xs, order + 1)
    f = list(xs)
    b = list(xs)
    a = [0.0] * (order + 1)
    a[0] = 1.0
    E = math.fsum(v * v for v in xs) / n
    for m in range(1, order + 1):
        num = 2.0 * math.fsum(f[t] * b[t - 1] for t in range(m, n))
        den = math.fsum(f[t] * f[t] + b[t - 1] * b[t - 1] for t in range(m, n))
        k = num / den if den > 1e-300 else 0.0   # num/den is scale-invariant
        new = list(a)
        for j in range(1, m + 1):
            new[j] = a[j] - k * a[m - j]
        a = new
        fn = list(f)
        for t in range(n - 1, m - 1, -1):
            fn[t] = f[t] - k * b[t - 1]
            b[t] = b[t - 1] - k * f[t]
        f = fn
        E *= (1.0 - k * k)
    return [-a[i] for i in range(1, order + 1)], E


def ar_psd(a: Sequence[float], sigma2: float, nfft: int = 512, fs: float = 1.0,
           onesided: bool = True) -> Tuple[Signal, Signal]:
    """Power spectral density of an AR model with coefficients ``a`` (``x[t] =
    sum a_i x[t-i] + e``) and innovation variance ``sigma2``."""
    p = len(a)
    freqs = rfftfreq(nfft, fs) if onesided else fftfreq(nfft, fs)
    out = []
    for fq in freqs:
        z = cmath.exp(-2j * math.pi * fq / fs)
        denom = 1.0 - sum(a[i] * z ** (i + 1) for i in range(p))
        d2 = max(denom.real ** 2 + denom.imag ** 2, 1e-300)     # floor: huge-but-FINITE
        out.append(sigma2 / (fs * d2))                          # at a near-unit-circle pole
    if onesided:                                 # fold negative-freq power (as periodogram
        for k in range(1, len(out) - (1 if nfft % 2 == 0 else 0)):   # /welch/multitaper do)
            out[k] *= 2.0                        # -> integrates to the full signal variance
    return freqs, out


def _corr_matrix(x: Signal, p: int) -> List[List[complex]]:
    """Forward-backward (modified-covariance) estimate of the ``p x p`` correlation
    matrix from a single record.  Averaging each length-``p`` snapshot with its
    reversed (backward) copy decorrelates the sources and effectively doubles the
    snapshot count -- the standard, well-conditioned estimate for the subspace
    (MUSIC) and minimum-variance (Capon) line-spectrum estimators, far sharper than
    the plain autocorrelation-Toeplitz form."""
    n = len(x)
    m = min(p, n)
    L = n - m + 1
    R = [[0.0] * m for _ in range(m)]
    for t in range(L):
        seg = x[t:t + m]
        for i in range(m):
            fi = seg[i]
            bi = seg[m - 1 - i]
            Ri = R[i]
            for j in range(m):
                Ri[j] += fi * seg[j] + bi * seg[m - 1 - j]
    scale = 1.0 / (2 * L) if L > 0 else 1.0
    return [[complex(R[i][j] * scale) for j in range(m)] for i in range(m)]


def music(x: Sequence[float], n_signals: int, nfft: int = 512, fs: float = 1.0,
          p: Optional[int] = None) -> Tuple[Signal, Signal]:
    """MUSIC pseudo-spectrum for resolving ``n_signals`` complex exponentials in
    noise.  Returns ``(freqs, P)`` with sharp peaks at the line frequencies.

    ``n_signals`` is the dimension of the signal subspace = the number of complex
    exponentials.  A *real* sinusoid ``cos(w t)`` is the sum of two exponentials at
    ``+/-w``, so to resolve ``r`` real tones pass ``n_signals = 2 * r``.  ``p`` is the
    correlation-matrix order (subspace size); it must exceed ``n_signals`` and larger
    ``p`` gives sharper resolution at the cost of more variance."""
    xs = detrend(_as_floats(x), "constant")
    if p is None:
        p = min(len(xs) // 2, max(2 * n_signals + 2, len(xs) // 8))
    if p <= n_signals:
        p = n_signals + 1
    R = _corr_matrix(xs, p)
    evals, evecs = _hermitian_eig(R)
    order = sorted(range(p), key=lambda i: evals[i])          # ascending
    noise = [order[i] for i in range(p - n_signals)]          # smallest eigenvalues
    freqs = rfftfreq(nfft, fs)
    P = []
    for fq in freqs:
        sv = [cmath.exp(2j * math.pi * fq / fs * k) for k in range(p)]
        s = 0.0
        for idx in noise:
            proj = sum(evecs[k][idx].conjugate() * sv[k] for k in range(p))
            s += proj.real ** 2 + proj.imag ** 2
        P.append(1.0 / max(s, 1e-300))           # huge-but-FINITE at a line frequency
    return freqs, P


def capon(x: Sequence[float], nfft: int = 512, fs: float = 1.0,
          p: Optional[int] = None) -> Tuple[Signal, Signal]:
    """Capon (minimum-variance / MVDR) spectral estimate -- higher resolution than
    the periodogram.  Returns ``(freqs, P)``."""
    xs = detrend(_as_floats(x), "constant")
    if p is None:
        p = min(len(xs) // 2, 16)
    R = _corr_matrix(xs, p)
    Rinv = _herm_inv(R)
    freqs = rfftfreq(nfft, fs)
    P = []
    for fq in freqs:
        sv = [cmath.exp(2j * math.pi * fq / fs * k) for k in range(p)]
        # denom = sv^H Rinv sv
        Rs = [sum(Rinv[i][j] * sv[j] for j in range(p)) for i in range(p)]
        denom = sum(sv[i].conjugate() * Rs[i] for i in range(p))
        d = denom.real
        P.append(p / (fs * max(d, 1e-300)))      # huge-but-FINITE for a near-singular R
    return freqs, P


# -- small Hermitian linear algebra for MUSIC/Capon --
def _hermitian_eig(A: List[List[complex]], iters: int = 100
                   ) -> Tuple[Signal, List[List[complex]]]:
    """Eigen-decomposition of a Hermitian matrix by cyclic complex-Jacobi rotations.
    Returns ``(eigenvalues, eigenvectors)`` with ``eigenvectors[i][j]`` the i-th
    component of the j-th eigenvector.

    Each rotation annihilates an off-diagonal element ``a_pq = r e^{i alpha}`` with the
    unitary ``U`` (identity except ``U_pp=U_qq=c``, ``U_pq=-s e^{i alpha}``,
    ``U_qp=s e^{-i alpha}``), where ``theta = 0.5 * atan2(2r, a_pp - a_qq)`` and
    ``c,s = cos,sin(theta)``; ``A <- U^H A U`` drives it to diagonal form.  (For a real
    symmetric matrix ``alpha`` is 0 or pi and this is the ordinary real Jacobi sweep.)"""
    n = len(A)
    # normalise by the matrix magnitude first: eigenVECTORS are unchanged by scaling and
    # eigenVALUES just scale back, but it keeps |a|^2 from overflowing for an extreme-
    # amplitude covariance (standard numerical hygiene).
    mag = max((abs(A[i][j]) for i in range(n) for j in range(n)), default=0.0)
    norm = mag if mag > 0.0 else 1.0
    a = [[complex(A[i][j]) / norm for j in range(n)] for i in range(n)]
    v = [[complex(1.0 if i == j else 0.0) for j in range(n)] for i in range(n)]
    scale = math.fsum(abs(a[i][j]) ** 2 for i in range(n) for j in range(n)) + 1e-300
    for _ in range(iters):
        off = math.fsum(abs(a[i][j]) ** 2 for i in range(n) for j in range(i + 1, n))
        if off < 1e-30 * scale:
            break
        for p in range(n):
            for q in range(p + 1, n):
                apq = a[p][q]
                if abs(apq) < 1e-300:
                    continue
                app, aqq = a[p][p].real, a[q][q].real
                alpha = cmath.phase(apq)
                theta = 0.5 * math.atan2(2.0 * abs(apq), app - aqq)
                c = math.cos(theta)
                s = math.sin(theta)
                ei = cmath.exp(1j * alpha)
                eic = ei.conjugate()
                for k in range(n):                       # columns: A <- A U
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = akp * c + akq * s * eic
                    a[k][q] = -akp * s * ei + akq * c
                for k in range(n):                       # rows: A <- U^H A
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk + s * ei * aqk
                    a[q][k] = -s * eic * apk + c * aqk
                for k in range(n):                       # eigenvectors: V <- V U
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = vkp * c + vkq * s * eic
                    v[k][q] = -vkp * s * ei + vkq * c
    evals = [a[i][i].real * norm for i in range(n)]      # scale eigenvalues back
    return evals, v


def _herm_inv(A: List[List[complex]]) -> List[List[complex]]:
    """Inverse of a Hermitian (positive-definite) matrix via Gauss-Jordan with a
    small diagonal loading for numerical stability.  Both the loading and the
    singularity test are SCALE-RELATIVE (to the matrix magnitude), so a covariance
    from a small-amplitude signal is not mistaken for singular."""
    n = len(A)
    scale = max((abs(A[i][j]) for i in range(n) for j in range(n)), default=0.0)
    if scale <= 0.0:
        raise SpectralError("singular correlation matrix.")
    mean_diag = sum(A[i][i].real for i in range(n)) / n
    load = 1e-9 * (mean_diag if mean_diag > 0.0 else scale)
    M = [[complex(A[i][j]) + (load if i == j else 0.0) for j in range(n)]
         + [complex(1.0 if i == j else 0.0) for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-14 * scale:             # relative singularity floor
            raise SpectralError("singular correlation matrix.")
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [v / pv for v in M[col]]
        for r in range(n):
            if r != col and abs(M[r][col]) > 0:
                fac = M[r][col]
                M[r] = [M[r][j] - fac * M[col][j] for j in range(2 * n)]
    return [[M[i][n + j] for j in range(n)] for i in range(n)]


# ===========================================================================
# 6. MULTITAPER (Slepian / DPSS)
# ===========================================================================
def _solve_tridiag(diag: Signal, sub: Signal, b: Signal) -> Signal:
    """Thomas algorithm for a symmetric tridiagonal system (``sub`` = the shared
    sub/super-diagonal, length n-1)."""
    n = len(diag)
    tiny = 1e-300
    cp = list(sub)
    dp = list(diag)
    bp = list(b)
    for i in range(1, n):
        piv = dp[i - 1] if abs(dp[i - 1]) > tiny else tiny
        m = cp[i - 1] / piv
        dp[i] -= m * cp[i - 1]
        bp[i] -= m * bp[i - 1]
    x = [0.0] * n
    piv = dp[n - 1] if abs(dp[n - 1]) > tiny else tiny
    x[n - 1] = bp[n - 1] / piv
    for i in range(n - 2, -1, -1):
        piv = dp[i] if abs(dp[i]) > tiny else tiny
        x[i] = (bp[i] - cp[i] * x[i + 1]) / piv
    return x


def _sturm_count(diag: Signal, sub2: Signal, x: float) -> int:
    """Number of eigenvalues of the symmetric tridiagonal matrix that are ``< x``
    (Sturm sequence / LDL sign count), for eigenvalue bisection."""
    q = diag[0] - x
    count = 1 if q < 0 else 0
    for i in range(1, len(diag)):
        if abs(q) < 1e-300:
            q = 1e-300 if q >= 0 else -1e-300
        q = (diag[i] - x) - sub2[i - 1] / q
        if q < 0:
            count += 1
    return count


def dpss(n: int, nw: float = 4.0, k: Optional[int] = None) -> List[Signal]:
    """Discrete prolate spheroidal (Slepian) sequences -- the ``k`` most
    band-concentrated tapers for time-bandwidth product ``nw``.  Returns a list of
    ``k`` length-``n`` unit-energy tapers, computed the efficient way (only the top-K
    eigenpairs of the defining tridiagonal matrix, via Sturm bisection + inverse
    iteration -- O(k*n), not a full O(n^3) eigendecomposition)."""
    if k is None:
        k = int(2 * nw) - 1
    k = max(1, min(k, n))
    if n == 1:
        return [[1.0]] * k
    cw = math.cos(_TWO_PI * nw / n)
    diag = [((n - 1 - 2 * i) / 2.0) ** 2 * cw for i in range(n)]
    sub = [0.5 * i * (n - i) for i in range(1, n)]
    sub2 = [s * s for s in sub]
    # Gershgorin bounds on the spectrum
    lo = min(diag[i] - (abs(sub[i - 1]) if i > 0 else 0.0)
             - (abs(sub[i]) if i < n - 1 else 0.0) for i in range(n))
    hi = max(diag[i] + (abs(sub[i - 1]) if i > 0 else 0.0)
             + (abs(sub[i]) if i < n - 1 else 0.0) for i in range(n))
    span = hi - lo + 1.0
    lo -= 0.01 * span
    hi += 0.01 * span
    tapers: List[Signal] = []
    for j in range(k):
        target = n - 1 - j                       # ascending index of the j-th largest eigenvalue
        a, b = lo, hi
        for _ in range(90):                      # bisection on the Sturm count
            mid = 0.5 * (a + b)
            if _sturm_count(diag, sub2, mid) <= target:
                a = mid
            else:
                b = mid
        lam = 0.5 * (a + b)
        v = [math.sin(math.pi * (i + 1) / (n + 1)) for i in range(n)]   # non-degenerate start
        shift = lam + 1e-11 * (abs(lam) + 1.0)    # perturb JUST off the eigenvalue: a
        shifted = [d - shift for d in diag]       # coarse shift (~span) loses accuracy for
        for _ in range(5):                        # high-order tapers when n is large
            v = _solve_tridiag(shifted, sub, v)   # inverse iteration...
            for _gs in range(2):                  # ...+ twice-is-enough re-orthogonalisation
                for u in tapers:
                    dot = math.fsum(v[i] * u[i] for i in range(n))
                    v = [v[i] - dot * u[i] for i in range(n)]
            nrm = math.sqrt(math.fsum(t * t for t in v)) or 1.0
            v = [t / nrm for t in v]
        # sign convention: even orders sum positive; odd orders have positive slope
        if j % 2 == 0:
            if math.fsum(v) < 0:
                v = [-t for t in v]
        else:
            if math.fsum((i - (n - 1) / 2.0) * v[i] for i in range(n)) < 0:
                v = [-t for t in v]
        tapers.append(v)
    return tapers


def multitaper(x: Sequence[float], fs: float = 1.0, nw: float = 4.0,
               k: Optional[int] = None, onesided: bool = True
               ) -> Tuple[Signal, Signal]:
    """Thomson multitaper PSD -- averages periodograms from ``k`` orthogonal Slepian
    tapers for a low-variance, low-leakage estimate.  Returns ``(freqs, Pxx)``."""
    xs = detrend(_as_floats(x), "constant")
    n = len(xs)
    _check_len(xs, 2)
    tapers = dpss(n, nw, k)
    K = len(tapers)
    if not onesided:                                     # full two-sided PSD (no folding)
        acc = [0.0] * n
        for taper in tapers:
            X = fft([xs[i] * taper[i] for i in range(n)])
            for kk in range(n):
                acc[kk] += X[kk].real ** 2 + X[kk].imag ** 2
        return fftfreq(n, fs), [acc[kk] / (K * fs) for kk in range(n)]
    m = n // 2 + 1
    acc = [0.0] * m
    for taper in tapers:
        X = rfft([xs[i] * taper[i] for i in range(n)])
        for kk in range(m):
            acc[kk] += (X[kk].real ** 2 + X[kk].imag ** 2)
    Pxx = [acc[kk] / (K * fs) for kk in range(m)]
    for kk in range(1, m - (1 if n % 2 == 0 else 0)):    # fold negative-freq power
        Pxx[kk] *= 2.0
    return rfftfreq(n, fs), Pxx


# ===========================================================================
# 7. FILTERING (FIR / IIR design + application + Hilbert)
# ===========================================================================
def firwin(numtaps: int, cutoff: Any, fs: float = 1.0, window: Any = "hamming",
           pass_zero: bool = True) -> Signal:
    """Windowed-sinc FIR filter design.  ``cutoff`` is a single edge (low/high-pass)
    or a ``[lo, hi]`` pair (band-pass/-stop); ``pass_zero`` selects which side
    passes DC.  Returns the (normalised) taps."""
    if isinstance(cutoff, (int, float)):
        cutoff = [float(cutoff)]
    cutoff = [c / (fs / 2.0) for c in cutoff]                # normalise to Nyquist=1
    # a Type II FIR (even numtaps) is forced to zero response at Nyquist, so it cannot
    # realise a filter whose highest band passes Nyquist (high-pass / band-stop).
    pass_nyquist = bool(pass_zero) ^ (len(cutoff) % 2 == 1)
    if numtaps % 2 == 0 and pass_nyquist:
        raise SpectralError("a high-pass / band-stop FIR needs an ODD numtaps "
                            "(an even-length filter has zero gain at Nyquist).")
    M = numtaps - 1
    center = M / 2.0
    w = get_window(window, numtaps)

    def sinc(fc: float, i: int) -> float:
        x = i - center
        return fc if abs(x) < _EPS else math.sin(math.pi * fc * x) / (math.pi * x)

    h = [0.0] * numtaps
    if len(cutoff) == 1:
        fc = cutoff[0]
        for i in range(numtaps):
            h[i] = sinc(fc, i) * w[i]
        if not pass_zero:                        # high-pass = spectral inversion
            h = [-v for v in h]
            h[int(center)] += 1.0 if numtaps % 2 == 1 else 0.0
    else:
        f1, f2 = cutoff
        for i in range(numtaps):
            h[i] = (sinc(f2, i) - sinc(f1, i)) * w[i]         # band-pass
        if pass_zero:                            # band-stop = spectral inversion
            h = [-v for v in h]
            if numtaps % 2 == 1:
                h[int(center)] += 1.0
    # normalise to unity gain at a reference frequency INSIDE the passband: DC for
    # low-pass / band-stop, Nyquist for high-pass, and the passband centre for a
    # band-pass (normalising a band-pass at DC or Nyquist -- where its gain is ~0 --
    # would divide by nearly zero and blow the taps up).
    if pass_zero:
        scale_freq = 0.0                                  # DC
    elif len(cutoff) == 1:
        scale_freq = 1.0                                  # Nyquist (high-pass)
    else:
        scale_freq = 0.5 * (cutoff[0] + cutoff[1])        # band centre (band-pass)
    s = math.fsum(h[i] * math.cos(math.pi * scale_freq * (i - center))
                  for i in range(numtaps))
    if abs(s) > _EPS:
        h = [v / s for v in h]
    return h


def _bilinear_biquad(b: Signal, a: Signal) -> Tuple[Signal, Signal]:
    return b, a


def butter(order: int, cutoff: Any, fs: float = 1.0, btype: str = "low"
           ) -> List[Signal]:
    """Butterworth IIR filter as a cascade of second-order sections (SOS).  Each row
    is ``[b0, b1, b2, a0, a1, a2]``.  ``btype`` in {'low','high'} (band via two
    passes).  Designed by the bilinear transform of the analog prototype."""
    if btype not in ("low", "high"):
        raise SpectralError("butter btype must be 'low' or 'high' (band-pass: chain two).")
    wc = 2.0 * math.pi * (float(cutoff) / fs)                # digital cutoff (rad/sample)
    warp = 2.0 * fs * math.tan(wc / 2.0)                     # pre-warp to analog
    sos = []
    n_biquads = order // 2
    for i in range(n_biquads):
        theta = math.pi * (2.0 * i + 1) / (2.0 * order)
        # analog conjugate pole pair at warp * exp(j*(pi/2 + theta)) ... use s-plane
        pole_re = -warp * math.sin(theta)
        pole_im = warp * math.cos(theta)
        # analog lowpass biquad: H(s) = wc^2 / (s^2 - 2 Re(p) s + |p|^2)
        p2 = pole_re * pole_re + pole_im * pole_im
        if btype == "low":
            b_s = [0.0, 0.0, warp * warp]
        else:
            b_s = [1.0, 0.0, 0.0]                            # highpass numerator s^2
        a_s = [1.0, -2.0 * pole_re, p2]
        sos.append(_bilinear_sos(b_s, a_s, fs, btype, warp))
    if order % 2 == 1:                                       # first-order section
        if btype == "low":
            b_s = [0.0, warp]
        else:
            b_s = [1.0, 0.0]
        a_s = [1.0, warp]
        sos.append(_bilinear_first(b_s, a_s, fs, btype, warp))
    return sos


def _bilinear_sos(b_s: Signal, a_s: Signal, fs: float, btype: str, warp: float
                  ) -> Signal:
    # bilinear: s -> 2 fs (1 - z^-1)/(1 + z^-1); substitute into the biquad
    K = 2.0 * fs
    b2, b1, b0 = b_s
    a2, a1, a0 = a_s
    # after substitution and clearing (1+z^-1)^2 denominators:
    bb0 = b2 * K * K + b1 * K + b0
    bb1 = -2.0 * b2 * K * K + 2.0 * b0
    bb2 = b2 * K * K - b1 * K + b0
    aa0 = a2 * K * K + a1 * K + a0
    aa1 = -2.0 * a2 * K * K + 2.0 * a0
    aa2 = a2 * K * K - a1 * K + a0
    return [bb0 / aa0, bb1 / aa0, bb2 / aa0, 1.0, aa1 / aa0, aa2 / aa0]


def _bilinear_first(b_s: Signal, a_s: Signal, fs: float, btype: str, warp: float
                    ) -> Signal:
    K = 2.0 * fs
    b1, b0 = b_s
    a1, a0 = a_s
    bb0 = b1 * K + b0
    bb1 = -b1 * K + b0
    aa0 = a1 * K + a0
    aa1 = -a1 * K + a0
    return [bb0 / aa0, bb1 / aa0, 0.0, 1.0, aa1 / aa0, 0.0]


def sosfilt(sos: List[Signal], x: Sequence[float]) -> Signal:
    """Apply a cascade of second-order sections to a signal (direct form II)."""
    y = _as_floats(x)
    for s in sos:
        b0, b1, b2, a0, a1, a2 = s
        w1 = w2 = 0.0
        out = []
        for v in y:
            w0 = v - a1 * w1 - a2 * w2
            out.append(b0 * w0 + b1 * w1 + b2 * w2)
            w2, w1 = w1, w0
        y = out
    return y


def lfilter(b: Sequence[float], a: Sequence[float], x: Sequence[float],
            zi: Optional[Sequence[float]] = None) -> Signal:
    """Apply an IIR/FIR filter ``a[0] y[n] = sum b_i x[n-i] - sum a_j y[n-j]`` using the
    transposed direct-form II structure (as SciPy does).  ``zi`` is an optional initial
    delay-line state of length ``max(len(a), len(b)) - 1`` (default: zeros)."""
    b = _as_floats(b, "b")
    a = _as_floats(a, "a")
    x = _as_floats(x)
    if abs(a[0]) < _EPS:
        raise SpectralError("a[0] must be non-zero.")
    order = max(len(a), len(b))
    a0 = a[0]
    bb = [v / a0 for v in b] + [0.0] * (order - len(b))
    aa = [v / a0 for v in a] + [0.0] * (order - len(a))
    z = [0.0] * (order - 1) if zi is None else list(zi)
    y = [0.0] * len(x)
    for m in range(len(x)):
        xm = x[m]
        ym = bb[0] * xm + (z[0] if order > 1 else 0.0)
        y[m] = ym
        for i in range(order - 2):                       # propagate the transposed state
            z[i] = bb[i + 1] * xm + z[i + 1] - aa[i + 1] * ym
        if order > 1:
            z[order - 2] = bb[order - 1] * xm - aa[order - 1] * ym
    return y


def _solve_linear(A: List[List[float]], b: Signal) -> Signal:
    """Solve the small dense system ``A x = b`` by Gaussian elimination with partial
    pivoting (used for the steady-state filter initial condition)."""
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-300:
            continue
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [v / pv for v in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                f = M[r][col]
                M[r] = [M[r][j] - f * M[col][j] for j in range(n + 1)]
    return [M[i][n] for i in range(n)]


def _lfilter_zi(b: Sequence[float], a: Sequence[float]) -> Signal:
    """Steady-state delay-line state of the filter for a unit-step input (SciPy's
    ``lfilter_zi``): ``zi = (I - A^T)^{-1} (b[1:] - a[1:] b[0])`` with ``A`` the
    companion matrix of ``a``.  Scaled by the first sample, it starts :func:`filtfilt`
    in steady state so the endpoints are transient-free."""
    b = _as_floats(b, "b")
    a = _as_floats(a, "a")
    order = max(len(a), len(b))
    a0 = a[0]
    aa = [v / a0 for v in a] + [0.0] * (order - len(a))
    bb = [v / a0 for v in b] + [0.0] * (order - len(b))
    m = order - 1
    if m == 0:
        return []
    # I - companion(a).T : companion.T has first column -a[1:], superdiagonal 1
    IminusA = [[(1.0 if i == j else 0.0) for j in range(m)] for i in range(m)]
    for i in range(m):
        IminusA[i][0] -= -aa[i + 1]
    for j in range(m - 1):
        IminusA[j][j + 1] -= 1.0
    B = [bb[i + 1] - aa[i + 1] * bb[0] for i in range(m)]
    return _solve_linear(IminusA, B)


def filtfilt(b: Sequence[float], a: Sequence[float], x: Sequence[float]) -> Signal:
    """Zero-phase filtering: filter forwards then backwards (doubles the order,
    cancels phase).  Uses odd reflection at the ends to reduce transients."""
    x = _as_floats(x)
    n = len(x)
    a = _as_floats(a, "a")
    ntaps = max(len(a), len(b))
    edge = min(3 * ntaps, n - 1) if n > 1 else 0
    if edge > 0:
        # odd reflection about each endpoint (excluding the endpoint itself, so the
        # leading and trailing extensions are symmetric -- the trailing one previously
        # duplicated x[-1] and was shifted by one, weakening the end transient control)
        pre = [2 * x[0] - x[i] for i in range(edge, 0, -1)]
        post = [2 * x[-1] - x[-i - 2] for i in range(edge)]
        ext = pre + x + post
    else:
        ext = list(x)
    zi = _lfilter_zi(b, a)                                # seed each pass in steady state
    y = lfilter(b, a, ext, zi=[z * ext[0] for z in zi])
    y = list(reversed(y))
    y = lfilter(b, a, y, zi=[z * y[0] for z in zi])
    y = list(reversed(y))
    return y[edge:edge + n]


def freqz(b: Sequence[float], a: Sequence[float] = (1.0,), worN: int = 512,
          fs: float = 1.0) -> Tuple[Signal, Spectrum]:
    """Frequency response ``H(e^{jw})`` of a filter.  Returns ``(freqs, H)``."""
    b = _as_floats(b, "b")
    a = _as_floats(a, "a")
    freqs = [k * (fs / 2.0) / (worN - 1) for k in range(worN)]
    H = []
    for f in freqs:
        w = math.pi * f / (fs / 2.0)
        z = cmath.exp(-1j * w)
        num = sum(b[i] * z ** i for i in range(len(b)))
        den = sum(a[j] * z ** j for j in range(len(a)))
        H.append(num / den if abs(den) > _EPS else complex(float("inf")))
    return freqs, H


def group_delay(b: Sequence[float], a: Sequence[float] = (1.0,), worN: int = 512,
                fs: float = 1.0) -> Tuple[Signal, Signal]:
    """Group delay ``-d(phase)/dw`` (in samples) of a filter."""
    f, H = freqz(b, a, worN, fs)
    phase = unwrap([cmath.phase(h) for h in H])
    dw = math.pi / (worN - 1)
    gd = [0.0] * worN
    for k in range(worN):
        if k == 0:
            gd[k] = -(phase[1] - phase[0]) / dw
        elif k == worN - 1:
            gd[k] = -(phase[-1] - phase[-2]) / dw
        else:
            gd[k] = -(phase[k + 1] - phase[k - 1]) / (2 * dw)
    return f, gd


def hilbert(x: Sequence[float]) -> Spectrum:
    """Analytic signal via the Hilbert transform: ``x + j*H{x}`` (the FFT method)."""
    xs = _as_floats(x)
    n = len(xs)
    X = fft(xs)
    h = [0.0] * n
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        for i in range(1, n // 2):
            h[i] = 2.0
    else:
        h[0] = 1.0
        for i in range(1, (n + 1) // 2):
            h[i] = 2.0
    return ifft([X[i] * h[i] for i in range(n)])


def envelope(x: Sequence[float]) -> Signal:
    """Amplitude envelope = ``|analytic signal|``."""
    return [abs(v) for v in hilbert(x)]


# ===========================================================================
# 8. CORRELATION / CONVOLUTION
# ===========================================================================
def convolve(a: Sequence[float], b: Sequence[float], mode: str = "full") -> Signal:
    """Convolution of two signals (FFT-fast for large inputs)."""
    x, y = _as_floats(a, "a"), _as_floats(b, "b")
    if len(x) * len(y) <= 4096:
        n = len(x) + len(y) - 1
        full = [0.0] * n
        for i in range(len(x)):
            for j in range(len(y)):
                full[i + j] += x[i] * y[j]
    else:
        return fftconvolve(x, y, mode)
    if mode == "full":
        return full
    if mode == "same":
        start = (len(y) - 1) // 2
        return full[start:start + len(x)]
    if mode == "valid":
        lo = min(len(x), len(y)) - 1
        return full[lo:max(len(x), len(y))]
    raise SpectralError("mode must be 'full', 'same' or 'valid'.")


def correlate(a: Sequence[float], b: Sequence[float], mode: str = "full") -> Signal:
    """Cross-correlation ``(a * conj(reversed b))``."""
    y = _as_floats(b, "b")
    return convolve(a, list(reversed(y)), mode)


def autocorrelate(x: Sequence[float], maxlag: Optional[int] = None,
                  normalize: bool = True) -> Signal:
    """Autocorrelation for lags ``0..maxlag`` (biased estimator, FFT-fast)."""
    xs = _as_floats(x)
    n = len(xs)
    if maxlag is None:
        maxlag = n - 1
    L = next_pow2(2 * n)
    X = fft(xs + [0.0] * (L - n))
    S = [X[i] * X[i].conjugate() for i in range(L)]
    r = [v.real for v in ifft(S)][:maxlag + 1]
    if normalize and abs(r[0]) > 1e-300:         # normalise for ANY non-zero energy
        r = [v / r[0] for v in r]                # (absolute _EPS wrongly skipped tiny signals)
    return r


# ===========================================================================
# 9. WAVELETS
# ===========================================================================
_WAVELETS: Dict[str, Signal] = {
    "haar": [0.7071067811865476, 0.7071067811865476],
    "db1": [0.7071067811865476, 0.7071067811865476],
    "db2": [-0.12940952255092145, 0.22414386804185735,
            0.836516303737469, 0.48296291314469025],
    "db4": [-0.010597401784997278, 0.032883011666982945, 0.030841381835986965,
            -0.18703481171888114, -0.02798376941698385, 0.6308807679295904,
            0.7148465705525415, 0.23037781330885523],
}


def _qmf(h: Signal) -> Signal:
    # quadrature mirror (high-pass) filter from the low-pass scaling filter
    return [(-1) ** i * h[len(h) - 1 - i] for i in range(len(h))]


def dwt(x: Sequence[float], wavelet: str = "db2") -> Tuple[Signal, Signal]:
    """Single-level discrete wavelet transform.  Returns ``(cA, cD)`` -- the
    approximation and detail coefficients (periodic boundary)."""
    xs = _as_floats(x)
    h = _WAVELETS[wavelet.lower()]
    g = _qmf(h)
    lf = len(h)
    n = len(xs)
    if n % 2:
        xs = xs + [xs[-1]]
        n += 1
    cA, cD = [], []
    for k in range(n // 2):
        sa = sd = 0.0
        for i in range(lf):
            idx = (2 * k + i) % n
            sa += h[i] * xs[idx]
            sd += g[i] * xs[idx]
        cA.append(sa)
        cD.append(sd)
    return cA, cD


def idwt(cA: Sequence[float], cD: Sequence[float], wavelet: str = "db2") -> Signal:
    """Inverse single-level DWT (reconstructs the signal)."""
    ca, cd = _as_floats(cA, "cA"), _as_floats(cD, "cD")
    h = _WAVELETS[wavelet.lower()]
    g = _qmf(h)
    lf = len(h)
    n = len(ca)
    out = [0.0] * (2 * n)
    for k in range(n):
        for i in range(lf):
            idx = (2 * k + i) % (2 * n)
            out[idx] += h[i] * ca[k] + g[i] * cd[k]
    return out


def wavedec(x: Sequence[float], wavelet: str = "db2", level: int = 1
            ) -> List[Signal]:
    """Multi-level DWT.  Returns ``[cA_L, cD_L, cD_{L-1}, ..., cD_1]``."""
    xs = _as_floats(x)
    coeffs = []
    a = xs
    for _ in range(level):
        a, d = dwt(a, wavelet)
        coeffs.append(d)
    coeffs.append(a)
    return [coeffs[-1]] + list(reversed(coeffs[:-1]))


def waverec(coeffs: List[Signal], wavelet: str = "db2") -> Signal:
    """Inverse multi-level DWT from :func:`wavedec` output.  Reconstructs exactly for
    ANY signal length: each level's approximation is cropped back to the length of the
    next finer detail band (an odd-length level pads by one sample on the way down, so
    the reconstruction is trimmed on the way up).  For an odd ORIGINAL length the final
    result may carry one trailing sample; crop to the known length if needed."""
    a = list(coeffs[0])
    for i in range(1, len(coeffs)):
        d = list(coeffs[i])
        if len(a) > len(d):                              # keep approx/detail aligned
            a = a[:len(d)]
        elif len(a) < len(d):
            a = a + [a[-1] if a else 0.0] * (len(d) - len(a))
        a = idwt(a, d, wavelet)
        if i + 1 < len(coeffs):                          # trim to the finer level's size
            a = a[:len(coeffs[i + 1])]
    return a


def morlet(points: int, w: float = 5.0, s: float = 1.0) -> Spectrum:
    """Complex Morlet wavelet of ``points`` samples (central frequency ``w``)."""
    out = []
    for i in range(points):
        t = (i - (points - 1) / 2.0) / s
        out.append(cmath.exp(1j * w * t) * math.exp(-0.5 * t * t))
    return out


def cwt(x: Sequence[float], scales: Sequence[float], w: float = 5.0
        ) -> List[Spectrum]:
    """Continuous wavelet transform with the complex Morlet wavelet.  Returns one
    complex row per scale."""
    xs = _as_floats(x)
    out = []
    for sc in scales:
        pts = min(len(xs), int(10 * sc) + 1)
        wav = [v.conjugate() for v in reversed(morlet(pts, w, sc))]
        conv = convolve(xs, [v.real for v in wav], "same")
        convi = convolve(xs, [v.imag for v in wav], "same")
        out.append([complex(conv[i], convi[i]) / math.sqrt(sc)
                    for i in range(len(xs))])
    return out


# ===========================================================================
# 10. SPECTRAL FEATURES
# ===========================================================================
def _mag_spectrum(x: Sequence[float]) -> Tuple[Signal, Signal]:
    xs = _as_floats(x)
    X = rfft(xs)
    mag = [abs(v) for v in X]
    freqs = rfftfreq(len(xs), 1.0)
    return freqs, mag


def spectral_centroid(x: Sequence[float], fs: float = 1.0) -> float:
    """Centre of mass of the magnitude spectrum (Hz) -- perceived 'brightness'."""
    f, mag = _mag_spectrum(x)
    f = [v * fs for v in f]
    num = math.fsum(f[k] * mag[k] for k in range(len(f)))
    den = math.fsum(mag)
    return num / den if den > 0.0 else 0.0       # scale-invariant: only silence -> 0


def spectral_bandwidth(x: Sequence[float], fs: float = 1.0, p: int = 2) -> float:
    """Spectral bandwidth: the p-th-order spread about the centroid."""
    f, mag = _mag_spectrum(x)
    f = [v * fs for v in f]
    c = spectral_centroid(x, fs)
    den = math.fsum(mag)
    if den <= 0.0:
        return 0.0
    return (math.fsum(mag[k] * abs(f[k] - c) ** p for k in range(len(f))) / den) ** (1.0 / p)


def spectral_flatness(x: Sequence[float]) -> float:
    """Spectral flatness (Wiener entropy) in ``[0, 1]``: geometric/arithmetic mean of
    the power spectrum.  Near 1 = noise-like, near 0 = tonal."""
    _, mag = _mag_spectrum(x)
    power = [m * m for m in mag]
    mean_p = math.fsum(power) / len(power)
    if mean_p <= 0.0:
        return 0.0
    floor = _EPS * mean_p                        # RELATIVE floor (keeps flatness scale-invariant;
    p = [v + floor for v in power]               # an absolute +_EPS dominated tiny-amplitude spectra)
    log_gm = math.fsum(math.log(v) for v in p) / len(p)
    am = math.fsum(p) / len(p)
    return math.exp(log_gm) / am


def spectral_rolloff(x: Sequence[float], fs: float = 1.0, roll: float = 0.85
                     ) -> float:
    """Roll-off frequency below which ``roll`` of the spectral energy lies."""
    f, mag = _mag_spectrum(x)
    f = [v * fs for v in f]
    energy = [m * m for m in mag]
    total = math.fsum(energy)
    if total <= 0.0:
        return 0.0
    cum = 0.0
    for k in range(len(energy)):
        cum += energy[k]
        if cum >= roll * total:
            return f[k]
    return f[-1]


def spectral_entropy(x: Sequence[float], normalize: bool = True) -> float:
    """Shannon entropy of the normalised power spectrum (spectral disorder)."""
    _, mag = _mag_spectrum(x)
    p = [m * m for m in mag]
    total = math.fsum(p)
    if total <= 0.0:
        return 0.0
    p = [v / total for v in p]                   # normalised (scale-invariant) distribution
    h = -math.fsum(v * math.log(v) for v in p if v > _EPS)
    return h / math.log(len(p)) if normalize and len(p) > 1 else h


def spectral_flux(x: Sequence[float], y: Sequence[float]) -> float:
    """Spectral flux between two frames: L2 norm of the magnitude-spectrum change."""
    _, mx = _mag_spectrum(x)
    _, my = _mag_spectrum(y)
    n = min(len(mx), len(my))
    return math.sqrt(math.fsum((mx[k] - my[k]) ** 2 for k in range(n)))


def spectral_crest(x: Sequence[float]) -> float:
    """Spectral crest factor: peak / mean of the power spectrum (tonality)."""
    _, mag = _mag_spectrum(x)
    p = [m * m for m in mag]
    mean = math.fsum(p) / len(p)
    return max(p) / mean if mean > 0.0 else 0.0  # scale-invariant peak/mean ratio


def dominant_frequency(x: Sequence[float], fs: float = 1.0) -> float:
    """Frequency (Hz) of the largest spectral peak, with parabolic interpolation."""
    f, mag = _mag_spectrum(x)
    if len(mag) < 2:
        return 0.0
    k = max(range(1, len(mag)), key=lambda i: mag[i]) if len(mag) > 1 else 0
    if 1 <= k < len(mag) - 1:                    # parabolic peak refinement
        a, b, c = mag[k - 1], mag[k], mag[k + 1]
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if abs(denom) > 0.0 else 0.0   # scale-invariant
        return (k + delta) * fs / len(x) if False else (k + delta) * (fs / (2 * (len(mag) - 1)))
    return f[k] * fs


def thd(x: Sequence[float], fs: float = 1.0, n_harmonics: int = 5) -> float:
    """Total harmonic distortion: RMS of harmonics 2..N relative to the fundamental."""
    f, mag = _mag_spectrum(x)
    if len(mag) < 2:
        return 0.0
    fund = max(range(1, len(mag)), key=lambda i: mag[i])
    if mag[fund] <= 0.0:                          # scale-invariant ratio; only silence -> 0
        return 0.0
    harm_power = 0.0
    for h in range(2, n_harmonics + 1):
        idx = fund * h
        if idx < len(mag):
            harm_power += mag[idx] ** 2
    return math.sqrt(harm_power) / mag[fund]


def snr(signal: Sequence[float], noise: Sequence[float]) -> float:
    """Signal-to-noise ratio in dB from separate signal and noise records."""
    ps = math.fsum(v * v for v in _as_floats(signal, "signal"))
    pn = math.fsum(v * v for v in _as_floats(noise, "noise"))
    return 10.0 * math.log10(ps / pn) if pn > 0.0 else float("inf")   # only zero noise -> inf


def find_peaks(x: Sequence[float], height: Optional[float] = None,
               distance: int = 1) -> List[int]:
    """Indices of local maxima, optionally filtered by minimum ``height`` and
    minimum ``distance`` between peaks."""
    xs = _as_floats(x)
    n = len(xs)
    peaks = [i for i in range(1, n - 1) if xs[i] > xs[i - 1] and xs[i] >= xs[i + 1]]
    if height is not None:
        peaks = [i for i in peaks if xs[i] >= height]
    if distance > 1 and peaks:
        peaks.sort(key=lambda i: xs[i], reverse=True)
        kept = []
        for pk in peaks:
            if all(abs(pk - q) >= distance for q in kept):
                kept.append(pk)
        peaks = sorted(kept)
    return peaks


# ===========================================================================
# 11. UTILITIES & GENERATORS
# ===========================================================================
def _median(v: Sequence[float]) -> float:
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def detrend(x: Sequence[float], type: str = "linear") -> Signal:
    """Remove a constant ('constant'/'mean') or linear ('linear') trend."""
    xs = _as_floats(x)
    n = len(xs)
    if n == 0 or type in ("none", None):
        return xs
    if type in ("constant", "mean"):
        mu = math.fsum(xs) / n
        return [v - mu for v in xs]
    if type == "linear":
        if n < 2:
            return [0.0] * n
        tbar = (n - 1) / 2.0
        ybar = math.fsum(xs) / n
        sxx = math.fsum((t - tbar) ** 2 for t in range(n))
        sxy = math.fsum((t - tbar) * (xs[t] - ybar) for t in range(n))
        slope = sxy / sxx if sxx > _EPS else 0.0
        icpt = ybar - slope * tbar
        return [xs[t] - (icpt + slope * t) for t in range(n)]
    raise SpectralError("detrend type must be 'constant', 'linear' or 'none'.")


def resample(x: Sequence[float], num: int) -> Signal:
    """Resample a signal to ``num`` samples by Fourier-domain interpolation."""
    xs = _as_floats(x)
    n = len(xs)
    if num <= 0:
        return []
    if num == n:
        return list(xs)
    X = fft(xs)
    Y = [0j] * num
    half = min(n, num) // 2
    for k in range(half + 1):
        Y[k] = X[k]
    for k in range(1, half + 1):
        Y[num - k] = X[n - k]
    if num < n and num % 2 == 0:                 # Nyquist handling on downsample
        Y[num // 2] = X[num // 2]
    scale = num / n
    return [v.real * scale for v in ifft(Y)]


def unwrap(phase: Sequence[float], period: float = _TWO_PI) -> Signal:
    """Unwrap phase by removing jumps greater than half a period."""
    p = _as_floats(phase, "phase")
    if not p:
        return []
    out = [p[0]]
    offset = 0.0
    for i in range(1, len(p)):
        d = p[i] - p[i - 1]
        while d > period / 2:
            offset -= period
            d -= period
        while d < -period / 2:
            offset += period
            d += period
        out.append(p[i] + offset)
    return out


def mag2db(x: Any) -> Any:
    """Amplitude -> decibels (``20 log10``)."""
    # only EXACT zero is -inf dB; a tiny-but-nonzero amplitude has a large finite dB
    # (an absolute _EPS cutoff punched artificial -inf holes into a dB spectrum)
    if isinstance(x, (list, tuple)):
        return [20.0 * math.log10(abs(v)) if abs(v) > 0.0 else -float("inf") for v in x]
    return 20.0 * math.log10(abs(x)) if abs(x) > 0.0 else -float("inf")


def db2mag(x: Any) -> Any:
    """Decibels -> amplitude."""
    if isinstance(x, (list, tuple)):
        return [10.0 ** (v / 20.0) for v in x]
    return 10.0 ** (x / 20.0)


def rms(x: Sequence[float]) -> float:
    """Root-mean-square level of a signal."""
    xs = _as_floats(x)
    return math.sqrt(math.fsum(v * v for v in xs) / len(xs)) if xs else 0.0


def zero_pad(x: Sequence[float], n: int) -> Signal:
    """Zero-pad (or truncate) a signal to length ``n``."""
    xs = _as_floats(x)
    if n <= len(xs):
        return xs[:n]
    return xs + [0.0] * (n - len(xs))


def sine_wave(freq: float, n: int, fs: float = 1.0, amp: float = 1.0,
              phase: float = 0.0) -> Signal:
    """Sampled sine wave."""
    return [amp * math.sin(_TWO_PI * freq * t / fs + phase) for t in range(n)]


def chirp(f0: float, f1: float, n: int, fs: float = 1.0, method: str = "linear"
          ) -> Signal:
    """Frequency sweep from ``f0`` to ``f1`` over ``n`` samples."""
    if method not in ("linear", "logarithmic"):
        raise SpectralError("method must be 'linear' or 'logarithmic'.")
    T = n / fs
    kk = logkk = 0.0
    if method == "logarithmic":
        if f0 <= 0:
            raise SpectralError("logarithmic chirp needs f0 > 0.")
        kk = (f1 / f0) ** (1.0 / T) if T > 0 else 1.0    # loop-invariant: compute ONCE
        logkk = math.log(kk) if kk > 0 else 0.0
    out = []
    for i in range(n):
        t = i / fs
        if method == "linear":
            ph = _TWO_PI * (f0 * t + 0.5 * (f1 - f0) / T * t * t)
        elif abs(logkk) > 1e-300:
            ph = _TWO_PI * f0 * (kk ** t - 1.0) / logkk
        else:                                            # f1 == f0: constant-frequency limit
            ph = _TWO_PI * f0 * t
        out.append(math.sin(ph))
    return out


def square_wave(freq: float, n: int, fs: float = 1.0, duty: float = 0.5) -> Signal:
    """Square wave with the given duty cycle."""
    return [1.0 if (freq * t / fs) % 1.0 < duty else -1.0 for t in range(n)]


def sawtooth(freq: float, n: int, fs: float = 1.0, width: float = 1.0) -> Signal:
    """Sawtooth / triangle wave (``width`` in [0,1]; 0.5 = triangle)."""
    out = []
    for t in range(n):
        ph = (freq * t / fs) % 1.0
        if ph < width:
            out.append(2.0 * ph / width - 1.0 if width > 0 else -1.0)
        else:
            out.append(1.0 - 2.0 * (ph - width) / (1.0 - width) if width < 1 else 1.0)
    return out


def impulse(n: int, position: int = 0) -> Signal:
    """Unit impulse of length ``n`` at ``position``."""
    x = [0.0] * n
    if 0 <= position < n:
        x[position] = 1.0
    return x


def white_noise(n: int, sigma: float = 1.0, seed: int = 0) -> Signal:
    """Gaussian white noise."""
    rng = random.Random(seed)
    return [rng.gauss(0.0, sigma) for _ in range(n)]


def pink_noise(n: int, seed: int = 0) -> Signal:
    """1/f (pink) noise via the Voss-McCartney algorithm."""
    rng = random.Random(seed)
    n_rows = 16
    rows = [0.0] * n_rows
    out = []
    for i in range(n):
        # update a subset of rows based on trailing zero bits
        k = i
        for r in range(n_rows):
            if k & 1:
                rows[r] = rng.gauss(0.0, 1.0)
                break
            k >>= 1
        else:
            rows[0] = rng.gauss(0.0, 1.0)
        out.append(math.fsum(rows) / n_rows)
    # normalise to unit variance
    mu = math.fsum(out) / n
    var = math.fsum((v - mu) ** 2 for v in out) / n
    sd = math.sqrt(var) or 1.0
    return [(v - mu) / sd for v in out]


# ===========================================================================
# 12. DEMO
# ===========================================================================
def _demo() -> None:
    print("=" * 70)
    print("  SPECTRAL ANALYSIS & DSP (pure Python) - demo")
    print("=" * 70)

    fs = 1000.0
    n = 1024

    print("\n-- FFT round-trip --")
    x = sine_wave(50.0, n, fs)
    rec = [v.real for v in ifft(fft(x))]
    print("   max |ifft(fft(x)) - x| = %.2e" % max(abs(rec[i] - x[i]) for i in range(n)))

    print("-- periodogram / Welch peak location (two tones at 50 & 120 Hz) --")
    sig = [sine_wave(50.0, n, fs)[i] + 0.5 * sine_wave(120.0, n, fs)[i]
           for i in range(n)]
    sig = [sig[i] + 0.1 * white_noise(n, seed=1)[i] for i in range(n)]
    f, P = welch(sig, fs, "hann", nperseg=256)
    peaks = sorted(find_peaks(P, height=max(P) * 0.1), key=lambda i: P[i], reverse=True)[:2]
    print("   Welch peaks near: %s Hz" % sorted(round(f[p]) for p in peaks))

    print("-- multitaper (Slepian) PSD peak --")
    fm, Pm = multitaper(sig, fs, nw=4.0)
    print("   multitaper dominant freq = %.0f Hz" % fm[max(range(len(Pm)), key=lambda i: Pm[i])])

    print("-- AR / Burg spectrum resolves the tones --")
    a, s2 = burg(sig, 24)
    fa, Pa = ar_psd(a, s2, nfft=512, fs=fs)
    apk = sorted(find_peaks(Pa, height=max(Pa) * 0.1), key=lambda i: Pa[i], reverse=True)[:2]
    print("   Burg-AR peaks near: %s Hz" % sorted(round(fa[p]) for p in apk))

    print("-- MUSIC line-spectrum (2 real tones -> 4 complex exponentials) --")
    two = [sine_wave(50.0, 256, fs)[i] + sine_wave(120.0, 256, fs)[i] for i in range(256)]
    fmu, Pmu = music(two, n_signals=4, nfft=512, fs=fs)
    mpk = sorted(find_peaks(Pmu), key=lambda i: Pmu[i], reverse=True)[:2]
    print("   MUSIC peaks near: %s Hz" % sorted(round(fmu[p]) for p in mpk))

    print("-- FIR low-pass + zero-phase filtfilt --")
    taps = firwin(51, 80.0, fs, "hamming")
    filt = filtfilt(taps, [1.0], sig)
    print("   filter taps sum (DC gain) = %.4f, output RMS = %.3f"
          % (math.fsum(taps), rms(filt)))

    print("-- Butterworth SOS low-pass --")
    sos = butter(4, 100.0, fs, "low")
    yb = sosfilt(sos, sig)
    print("   %d biquad sections, output RMS = %.3f" % (len(sos), rms(yb)))

    print("-- Hilbert envelope of an AM signal --")
    am = [(1.0 + 0.5 * math.sin(_TWO_PI * 5 * t / fs)) * math.sin(_TWO_PI * 100 * t / fs)
          for t in range(n)]
    env = envelope(am)
    print("   envelope mean = %.3f (carrier amplitude ~1)" % (math.fsum(env) / n))

    print("-- STFT / iSTFT reconstruction --")
    f2, t2, Z = stft(sig, fs, "hann", nperseg=256, noverlap=192)
    back = istft(Z, fs, "hann", nperseg=256, noverlap=192)
    m = min(len(back), n)
    print("   %d frames; iSTFT max err (interior) = %.2e"
          % (len(Z), max(abs(back[i] - sig[i]) for i in range(300, 700))))

    print("-- wavelet decompose / reconstruct (db4) --")
    coeffs = wavedec(x, "db4", level=3)
    xr = waverec(coeffs, "db4")
    print("   %d levels, reconstruction err = %.2e"
          % (len(coeffs) - 1, max(abs(xr[i] - x[i]) for i in range(n))))

    print("-- spectral features --")
    print("   centroid=%.1f Hz  flatness=%.3f  entropy=%.3f  dominant=%.0f Hz"
          % (spectral_centroid(sig, fs), spectral_flatness(sig),
             spectral_entropy(sig), dominant_frequency(sig, fs)))

    print("\n" + "=" * 70)
    print("  demo OK")
    print("=" * 70)


if __name__ == "__main__":
    _demo()
