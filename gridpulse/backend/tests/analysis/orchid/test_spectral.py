"""
Oracle-based stress tests for ``spectral_template``.

The philosophy is *no hand-typed magic number where a closed form, an exact
identity or a metamorphic invariant exists*:

* **Closed forms** -- the FFT reproduces the direct DFT to machine precision; a
  cosine at an exact bin lands with power ``A^2/2`` in the one-sided periodogram;
  a linear-phase FIR has constant group delay ``(numtaps-1)/2``; a Butterworth
  filter is exactly ``-3 dB`` at its cutoff; ``_bessel_i0`` matches tabulated
  values; the Haar DWT reproduces a hand-computed pair; ``snr`` matches its formula.
* **Exact identities** -- ``ifft(fft(x)) == x`` and ``rfft``/``irfft``,
  ``idwt(dwt(x)) == x``, ``fftshift``/``ifftshift`` and Box-cox-free round-trips;
  Parseval for the FFT, the periodogram and the (orthonormal) DWT; ``lfilter(b,[1],x)``
  equals FIR convolution; ``fftconvolve`` equals the direct convolution.
* **Metamorphic / spectral** -- Welch, Burg, MUSIC, Capon and multitaper all put
  their peaks at the injected tones; multitaper has lower variance than the raw
  periodogram; the spectral flatness/entropy of an impulse is 1 and of a pure tone
  is ~0; coherence of a signal with itself is 1; a low tone passes ``filtfilt`` with
  zero phase lag; the analytic signal's envelope of a pure tone is constant.
* **Differential** (only if NumPy / SciPy are importable) -- ``fft`` vs ``numpy.fft``,
  windows / ``firwin`` / ``dpss`` / ``welch`` vs SciPy, to machine / estimator
  precision.  These SKIP cleanly when the libraries are absent; the module and the
  rest of the suite remain zero-dependency.

Run:  ``pytest _stress_test.py -q``
"""

import cmath
import math
import random

import pytest

from gridpulse.analysis.algorithms.orchid import spectral as sp
from gridpulse.analysis.algorithms.orchid.spectral import (
    SpectralError,
    # transforms
    fft, ifft, rfft, irfft, dft, idft, czt, fftconvolve,
    fftfreq, rfftfreq, fftshift, ifftshift, next_pow2, next_fast_len,
    # windows
    get_window, boxcar, hann, hamming, blackman, blackmanharris, bartlett,
    kaiser, tukey, gaussian, flattop, nuttall, window_enbw, coherent_gain,
    # psd
    periodogram, welch, csd, coherence,
    # time-frequency
    stft, istft, spectrogram,
    # parametric
    levinson_durbin, ar_yule_walker, burg, ar_psd, music, capon,
    # multitaper
    dpss, multitaper,
    # filtering
    firwin, butter, sosfilt, lfilter, filtfilt, freqz, group_delay,
    hilbert, envelope,
    # correlation
    convolve, correlate, autocorrelate,
    # wavelets
    dwt, idwt, wavedec, waverec, cwt, morlet,
    # features
    spectral_centroid, spectral_bandwidth, spectral_flatness, spectral_rolloff,
    spectral_entropy, spectral_flux, spectral_crest, dominant_frequency, thd,
    snr, find_peaks,
    # utilities & generators
    detrend, resample, unwrap, mag2db, db2mag, rms, zero_pad,
    sine_wave, chirp, square_wave, sawtooth, impulse, white_noise, pink_noise,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _maxabs(a, b):
    return max(abs(a[i] - b[i]) for i in range(len(a)))


def _cmaxabs(a, b):
    return max(abs(a[i] - b[i]) for i in range(len(a)))


def cosine(freq, n, fs=1.0, amp=1.0):
    return sine_wave(freq, n, fs, amp, phase=math.pi / 2.0)   # amp*cos(2pi f t/fs)


# ===========================================================================
# 1. TRANSFORMS -- FFT / DFT / rFFT / CZT
# ===========================================================================
def test_fft_matches_direct_dft_power_of_two():
    x = [1.0, 2.0, -3.0, 0.5, 4.0, -1.0, 2.5, 0.0]           # n = 8 (radix-2 path)
    assert _cmaxabs(fft(x), dft(x)) < 1e-10


def test_fft_matches_direct_dft_non_power_of_two():
    # n = 6 and n = 7 exercise the Bluestein chirp-z path; it MUST equal the DFT.
    for x in ([1.0, 2.0, -3.0, 0.5, 4.0, -1.0], [3.0, -1.0, 0.0, 2.0, 5.0, -2.0, 1.0]):
        assert _cmaxabs(fft(x), dft(x)) < 1e-9


def test_fft_ifft_roundtrip_all_lengths():
    rng = random.Random(7)
    for n in (1, 2, 3, 5, 8, 13, 16, 17, 31, 64):
        x = [rng.gauss(0, 1) for _ in range(n)]
        rec = [v.real for v in ifft(fft(x))]
        assert _maxabs(rec, x) < 1e-9


def test_fft_is_linear():
    rng = random.Random(1)
    x = [rng.gauss(0, 1) for _ in range(12)]
    y = [rng.gauss(0, 1) for _ in range(12)]
    X, Y = fft(x), fft(y)
    Z = fft([3.0 * x[i] - 2.0 * y[i] for i in range(12)])
    ref = [3.0 * X[i] - 2.0 * Y[i] for i in range(12)]
    assert _cmaxabs(Z, ref) < 1e-10


def test_fft_of_impulse_is_flat_and_constant_is_delta():
    d = impulse(16, 0)
    X = fft(d)
    assert max(abs(v - 1.0) for v in X) < 1e-12         # delta -> all ones
    c = [2.5] * 16
    C = fft(c)
    assert abs(C[0] - 16 * 2.5) < 1e-10 and max(abs(C[k]) for k in range(1, 16)) < 1e-9


def test_parseval_theorem_fft():
    rng = random.Random(3)
    x = [rng.gauss(0, 1) for _ in range(32)]
    X = fft(x)
    lhs = math.fsum(v * v for v in x)
    rhs = math.fsum(abs(v) ** 2 for v in X) / len(x)
    assert abs(lhs - rhs) < 1e-9


def test_fft_shift_theorem():
    # circularly shifting a signal multiplies its spectrum by a linear phase ramp
    n = 48
    rng = random.Random(48)
    x = [rng.gauss(0, 1) for _ in range(n)]
    s = 7
    xs = [x[(i - s) % n] for i in range(n)]
    X, Xs = fft(x), fft(xs)
    pred = [X[k] * cmath.exp(-2j * math.pi * k * s / n) for k in range(n)]
    assert max(abs(Xs[k] - pred[k]) for k in range(n)) < 1e-11


def test_fft_convolution_and_modulation_theorems():
    n = 64
    rng = random.Random(64)
    a = [rng.gauss(0, 1) for _ in range(n)]
    b = [rng.gauss(0, 1) for _ in range(n)]
    Fa, Fb = fft(a), fft(b)
    circ = [math.fsum(a[j] * b[(i - j) % n] for j in range(n)) for i in range(n)]
    via = [v.real for v in ifft([Fa[k] * Fb[k] for k in range(n)])]      # convolution theorem
    assert max(abs(circ[i] - via[i]) for i in range(n)) < 1e-10
    k0 = 10                                                              # modulation theorem
    M = fft([a[i] * math.cos(2 * math.pi * k0 * i / n) for i in range(n)])
    pred = [0.5 * (Fa[(k - k0) % n] + Fa[(k + k0) % n]) for k in range(n)]
    assert max(abs(M[k] - pred[k]) for k in range(n)) < 1e-10


def test_rfft_matches_fft_slice_and_irfft_roundtrip():
    rng = random.Random(5)
    for n in (16, 17):
        x = [rng.gauss(0, 1) for _ in range(n)]
        assert _cmaxabs(rfft(x), fft(x)[: n // 2 + 1]) < 1e-10
        assert _maxabs(irfft(rfft(x), n), x) < 1e-9


def test_idft_inverts_dft():
    rng = random.Random(9)
    x = [rng.gauss(0, 1) for _ in range(7)]
    rec = [v.real for v in idft(dft(x))]
    assert _maxabs(rec, x) < 1e-9


def test_czt_default_equals_dft():
    x = [1.0, -2.0, 3.0, 0.5, 2.0]
    assert _cmaxabs(czt(x), dft(x)) < 1e-8


def test_czt_zoom_resolves_offbin_tone():
    # Zoom the CZT onto a narrow band around a non-integer-bin tone and confirm the
    # magnitude peak sits at that frequency (finer than the DFT grid).
    fs, n = 100.0, 200
    f0 = 12.7
    x = cosine(f0, n, fs)
    f_lo, f_hi, m = 10.0, 15.0, 256
    a0 = cmath.exp(2j * math.pi * f_lo / fs)
    w0 = cmath.exp(-2j * math.pi * (f_hi - f_lo) / fs / m)
    Z = czt(x, m=m, w=w0, a=a0)
    kmax = max(range(m), key=lambda k: abs(Z[k]))
    f_at = f_lo + (f_hi - f_lo) * kmax / m
    assert abs(f_at - f0) < 0.2


def test_czt_large_n_is_fast_and_correct():
    # regression: czt used to recompute BOTH FFTs inside a length-L comprehension
    # (O(L^2 log L)) and hung for a few hundred points; it must be O(L log L) now.
    import time
    rng = random.Random(19)
    n = 384                                              # non-power-of-two
    x = [rng.gauss(0, 1) for _ in range(n)]
    t0 = time.perf_counter()
    Z = czt(x)
    assert time.perf_counter() - t0 < 3.0                # the old path took far longer
    ref = dft(x)
    assert max(abs(Z[k] - ref[k]) for k in range(n)) < 1e-6


def test_fftconvolve_matches_direct_convolution():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [0.5, -1.0, 2.0]
    ref = convolve(a, b, "full")
    assert _maxabs(fftconvolve(a, b, "full"), ref) < 1e-9


def test_fftfreq_and_rfftfreq_values():
    assert fftfreq(4, 8.0) == [0.0, 2.0, -4.0, -2.0]
    assert fftfreq(5, 5.0) == pytest.approx([0.0, 1.0, 2.0, -2.0, -1.0])
    assert rfftfreq(8, 16.0) == [0.0, 2.0, 4.0, 6.0, 8.0]


def test_fftshift_roundtrip_even_and_odd():
    for x in ([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4]):
        assert ifftshift(fftshift(x)) == list(x)
    # DC (index 0) moves to the centre
    assert fftshift([10, 1, 2, 3])[2] == 10


def test_next_pow2_and_next_fast_len():
    assert [next_pow2(k) for k in (1, 2, 3, 5, 17, 1000)] == [1, 2, 4, 8, 32, 1024]
    for n in (1, 2, 7, 100, 1000):
        m = next_fast_len(n)
        assert m >= n
        r = m
        for f in (2, 3, 5):
            while r % f == 0:
                r //= f
        assert r == 1                                    # 5-smooth


# ===========================================================================
# 2. WINDOWS
# ===========================================================================
def test_symmetric_windows_endpoints_and_peak():
    for n in (9, 33):
        for wf, edge in ((hann, 0.0), (hamming, 0.08), (blackman, 0.0)):
            w = wf(n)
            assert abs(w[0] - edge) < 1e-9 and abs(w[-1] - edge) < 1e-9
            assert abs(w[n // 2] - 1.0) < 1e-9           # centre = 1 for odd n
            assert _maxabs(w, list(reversed(w))) < 1e-12  # symmetric


def test_boxcar_and_bartlett():
    assert boxcar(5) == [1.0] * 5
    b = bartlett(5)                                       # 0,0.5,1,0.5,0
    assert b == pytest.approx([0.0, 0.5, 1.0, 0.5, 0.0])


def test_kaiser_beta_zero_is_boxcar():
    w = kaiser(16, beta=0.0)
    assert _maxabs(w, [1.0] * 16) < 1e-12


def test_bessel_i0_matches_tabulated():
    assert abs(sp._bessel_i0(0.0) - 1.0) < 1e-15
    assert abs(sp._bessel_i0(1.0) - 1.2660658778) < 1e-8
    assert abs(sp._bessel_i0(2.0) - 2.2795853023) < 1e-8


def test_tukey_limits_reduce_to_boxcar_and_hann():
    assert _maxabs(tukey(32, alpha=0.0), boxcar(32)) < 1e-12
    assert _maxabs(tukey(32, alpha=1.0), hann(32)) < 1e-12


def test_window_enbw_and_coherent_gain_boxcar():
    assert abs(window_enbw(boxcar(64)) - 1.0) < 1e-12
    assert abs(coherent_gain(boxcar(64)) - 1.0) < 1e-12
    # Hann: ENBW -> 1.5, coherent gain -> ~0.5 (N/(N-1)*0.5 for the symmetric form)
    assert abs(window_enbw(hann(4096)) - 1.5) < 1e-2
    assert abs(coherent_gain(hann(4096)) - 0.5) < 1e-3


def test_window_enbw_matches_literature():
    # equivalent noise bandwidth (bins) vs Harris (1978) / SciPy for the standard
    # windows -- pins the window coefficients, not just their shape.  (nuttall here is
    # the minimum-4-term variant SciPy ships, ENBW ~1.976, not the 2.02 C1 variant.)
    N = 8192
    known = {"hann": 1.5, "hamming": 1.3628, "blackman": 1.7268,
             "blackmanharris": 2.0044, "nuttall": 1.9761, "flattop": 3.7702,
             "boxcar": 1.0}
    for name, ref in known.items():
        w = get_window(name, N + 1)[:N]                  # periodic (spectral) form
        assert abs(window_enbw(w) - ref) < 5e-3


def test_get_window_dispatch_forms():
    assert _maxabs(get_window("hann", 8), hann(8)) < 1e-12
    assert _maxabs(get_window(("kaiser", 5.0), 8), kaiser(8, 5.0)) < 1e-12
    assert get_window([1.0, 2.0, 3.0], 3) == [1.0, 2.0, 3.0]
    with pytest.raises(SpectralError):
        get_window([1.0, 2.0], 3)                        # length mismatch


# ===========================================================================
# 3. NON-PARAMETRIC PSD
# ===========================================================================
def test_periodogram_parseval_energy():
    # integral of the one-sided density PSD == mean power of the detrended signal
    rng = random.Random(2)
    n, fs = 128, 50.0
    x = [rng.gauss(0, 1) for _ in range(n)]
    f, P = periodogram(x, fs=fs, window="boxcar", scaling="density")
    df = fs / n
    xd = detrend(x, "constant")
    assert abs(math.fsum(P) * df - math.fsum(v * v for v in xd) / n) < 1e-9


def test_periodogram_tone_power_is_amp_squared_over_two():
    # a full-amplitude cosine at an EXACT bin: one-sided 'spectrum' peak == A^2/2
    n, fs, A = 64, 64.0, 3.0
    k0 = 8
    x = cosine(k0 * fs / n, n, fs, amp=A)
    f, P = periodogram(x, fs=fs, window="boxcar", scaling="spectrum")
    kpk = max(range(len(P)), key=lambda i: P[i])
    assert kpk == k0
    assert abs(P[kpk] - A * A / 2.0) < 1e-6


def test_welch_and_features_are_sampling_rate_consistent():
    # a tone at (f, fs) and (2f, 2fs) share the same normalised frequency -> same peak
    # bin; the density PSD scales as 1/fs; flatness is dimensionless (fs-independent).
    n = 1024
    x1 = sine_wave(50.0, n, 1000.0)
    x2 = sine_wave(100.0, n, 2000.0)
    f1, P1 = welch(x1, 1000.0, "hann", 256)
    f2, P2 = welch(x2, 2000.0, "hann", 256)
    k1 = max(range(len(P1)), key=lambda i: P1[i])
    k2 = max(range(len(P2)), key=lambda i: P2[i])
    assert k1 == k2
    assert abs(P1[k1] / P2[k2] - 2.0) < 0.01                # density ~ 1/fs
    assert abs(spectral_flatness(x1) - spectral_flatness(x2)) < 1e-9


def test_welch_locates_two_tones():
    fs, n = 1000.0, 4096
    x = [sine_wave(70.0, n, fs)[i] + sine_wave(200.0, n, fs)[i] for i in range(n)]
    f, P = welch(x, fs, "hann", nperseg=512)
    pk = sorted(find_peaks(P, height=max(P) * 0.1), key=lambda i: P[i], reverse=True)[:2]
    got = sorted(f[i] for i in pk)
    assert abs(got[0] - 70.0) <= fs / 512 and abs(got[1] - 200.0) <= fs / 512


def test_welch_reduces_variance_versus_periodogram():
    # on white noise, segment-averaging must lower the PSD estimator variance
    x = white_noise(4096, 1.0, seed=11)
    _, Pp = periodogram(x, fs=1.0, window="boxcar")
    _, Pw = welch(x, fs=1.0, window="boxcar", nperseg=256, noverlap=0)

    def cv(P):
        m = math.fsum(P) / len(P)
        v = math.fsum((p - m) ** 2 for p in P) / len(P)
        return math.sqrt(v) / m
    assert cv(Pw) < cv(Pp)


def test_welch_short_signal_falls_back_to_periodogram():
    x = [1.0, 2.0, 3.0, 2.0, 1.0]
    f, P = welch(x, nperseg=256)                          # nperseg clamped to n
    assert len(f) == len(P) and len(P) == len(x) // 2 + 1


def test_welch_median_averaging_is_bias_corrected():
    # the median of exponential-distributed periodograms is biased LOW; the bias
    # correction (as in SciPy) restores the mean level.  Without it the median-averaged
    # total power was ~0.7x the mean-averaged one.
    x = white_noise(8192, 1.0, seed=21)
    _, Pmean = welch(x, 1.0, "boxcar", nperseg=256, noverlap=0, average="mean")
    _, Pmed = welch(x, 1.0, "boxcar", nperseg=256, noverlap=0, average="median")
    ratio = math.fsum(Pmed) / math.fsum(Pmean)
    assert 0.9 < ratio < 1.1                              # was ~0.71 before the correction


def test_welch_rejects_unknown_average():
    with pytest.raises(SpectralError):
        welch([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], nperseg=4, average="geometric")


def test_welch_single_segment_equals_periodogram():
    # welch over one full-length segment with no overlap must reduce EXACTLY to the
    # periodogram -- a cross-function consistency check independent of any reference.
    rng = random.Random(512)
    x = [rng.gauss(0, 1) for _ in range(512)]
    for scaling in ("density", "spectrum"):
        _, Pw = welch(x, 3.0, "boxcar", nperseg=512, noverlap=0,
                      detrend_type="constant", scaling=scaling)
        _, Pp = periodogram(x, 3.0, "boxcar", detrend_type="constant", scaling=scaling)
        assert max(abs(Pw[k] - Pp[k]) for k in range(len(Pw))) < 1e-12


def test_coherence_of_signal_with_itself_is_one():
    x = white_noise(2048, 1.0, seed=4)
    f, C = coherence(x, x, fs=1.0, nperseg=256)
    assert all(abs(c - 1.0) < 1e-9 for c in C)


def test_coherence_bounded_and_low_for_independent():
    x = white_noise(4096, 1.0, seed=1)
    y = white_noise(4096, 1.0, seed=2)
    f, C = coherence(x, y, nperseg=256)
    assert all(-1e-12 <= c <= 1.0 + 1e-9 for c in C)
    assert math.fsum(C) / len(C) < 0.5                   # uncorrelated -> low


def test_csd_of_signal_with_itself_is_its_psd():
    x = white_noise(2048, 1.0, seed=8)
    f, Pxy = csd(x, x, fs=1.0, nperseg=256)
    _, Pxx = welch(x, fs=1.0, window="hann", nperseg=256)
    assert max(abs(Pxy[k].real - Pxx[k]) for k in range(len(f))) < 1e-9
    assert max(abs(Pxy[k].imag) for k in range(len(f))) < 1e-9


def test_csd_phase_recovers_a_pure_delay():
    # for y[n] = x[n-D] the cross-spectrum phase is -2*pi*f*D/fs (linear in f); recovering
    # D pins the conjugate-convention SIGN, which magnitude / coherence tests cannot.
    fs, n, D = 1000.0, 8192, 5
    x = white_noise(n, 1.0, seed=1)
    y = [x[i - D] if i >= D else 0.0 for i in range(n)]
    f, Pxy = csd(x, y, fs, "hann", 512)
    ph = unwrap([cmath.phase(v) for v in Pxy])
    ks = list(range(3, len(f) // 2))
    w = [2 * math.pi * f[k] / fs for k in ks]
    p = [ph[k] for k in ks]
    mw, mp = sum(w) / len(w), sum(p) / len(p)
    slope = (sum((w[i] - mw) * (p[i] - mp) for i in range(len(w)))
             / sum((w[i] - mw) ** 2 for i in range(len(w))))
    assert abs(-slope - D) < 0.2                          # cross-spectrum group delay = D


def test_coherence_is_scale_invariant():
    # coherence is a normalised [0,1] quantity, so scaling both signals must not change
    # it -- an absolute cutoff wrongly zeroed the coherence of small-amplitude signals.
    fs, n = 1000.0, 2048
    x = [sine_wave(50.0, n, fs)[i] + 0.3 * white_noise(n, 1.0, seed=1)[i] for i in range(n)]
    nz = white_noise(n, 1.0, seed=2)
    y = [0.7 * x[i] + 0.3 * nz[i] for i in range(n)]
    _, C0 = coherence(x, y, fs, "hann", 256)
    for c in (1e-8, 1e8):
        _, Cc = coherence([c * v for v in x], [c * v for v in y], fs, "hann", 256)
        assert max(abs(C0[k] - Cc[k]) for k in range(len(C0))) < 1e-9


def test_csd_recovers_transfer_function_and_coherence_is_one():
    # for a linear system y = FIR(b) * x, the estimated transfer function Pxy / Pxx
    # must recover the filter response freqz(b), and the coherence must be ~1.  This is
    # the strong check that csd uses the right conjugate convention.
    b = [0.2, 0.3, 0.25, 0.1]
    x = white_noise(16384, 1.0, seed=31)
    y = lfilter(b, [1.0], x)
    f, Pxy = csd(x, y, 1.0, "hann", 512)
    _, Pxx = welch(x, 1.0, "hann", 512)
    _, Htrue = freqz(b, [1.0], worN=len(f), fs=1.0)
    Hest = [Pxy[k] / Pxx[k] for k in range(len(f))]
    assert max(abs(Hest[k] - Htrue[k]) for k in range(5, len(f) - 5)) < 0.05
    _, C = coherence(x, y, 1.0, "hann", 512)
    assert min(C[5:-5]) > 0.99                            # deterministic linear relation


# ===========================================================================
# 4. TIME-FREQUENCY (STFT / iSTFT / spectrogram)
# ===========================================================================
def test_stft_istft_reconstructs_interior():
    fs, n = 1000.0, 2048
    x = [sine_wave(60.0, n, fs)[i] + 0.5 * sine_wave(180.0, n, fs)[i] for i in range(n)]
    f, t, Z = stft(x, fs, "hann", nperseg=256, noverlap=192)
    back = istft(Z, fs, "hann", nperseg=256, noverlap=192)
    assert max(abs(back[i] - x[i]) for i in range(300, 1700)) < 1e-9


def test_stft_frame_count_and_shapes():
    n, nperseg, nov = 1000, 256, 128
    x = white_noise(n, 1.0, seed=6)
    f, t, Z = stft(x, 1.0, "hann", nperseg=nperseg, noverlap=nov)
    step = nperseg - nov
    expected = (n - nperseg) // step + 1
    assert len(Z) == expected == len(t)
    assert all(len(frame) == nperseg // 2 + 1 == len(f) for frame in Z)


def test_stft_is_linear():
    # the STFT is a linear operator: STFT(a*x + b*y) == a*STFT(x) + b*STFT(y)
    fs = 1000.0
    x = white_noise(2048, 1.0, seed=8)
    y = white_noise(2048, 1.0, seed=9)
    al, be = 1.3, -0.7
    _, _, Zc = stft([al * x[i] + be * y[i] for i in range(2048)], fs, "hann", 256, 128)
    _, _, Zx = stft(x, fs, "hann", 256, 128)
    _, _, Zy = stft(y, fs, "hann", 256, 128)
    d = max(abs(Zc[fr][k] - (al * Zx[fr][k] + be * Zy[fr][k]))
            for fr in range(len(Zc)) for k in range(len(Zc[0])))
    assert d < 1e-9


def test_spectrogram_power_peaks_at_tone():
    fs, n = 500.0, 2048
    x = sine_wave(50.0, n, fs)
    f, t, S = spectrogram(x, fs, "hann", nperseg=256)
    avg = [math.fsum(S[fr][k] for fr in range(len(S))) / len(S) for k in range(len(f))]
    assert abs(f[max(range(len(avg)), key=lambda k: avg[k])] - 50.0) <= fs / 256


def test_spectrogram_average_equals_welch():
    # a power spectrogram averaged over time IS the Welch PSD: both detrend each
    # segment (spectrogram previously skipped detrending, biasing the low bins) and
    # both must fold by the TRUE parity of nperseg (an ODD nperseg has no Nyquist bin,
    # so its last bin is doubled too -- spectrogram used to miss this).
    x = white_noise(4096, 1.0, seed=22)
    for nperseg in (256, 255):                           # even AND odd
        f, t, Sxx = spectrogram(x, 1.0, "hann", nperseg=nperseg, noverlap=nperseg // 2)
        avg = [math.fsum(Sxx[fr][k] for fr in range(len(Sxx))) / len(Sxx)
               for k in range(len(f))]
        fw, Pw = welch(x, 1.0, "hann", nperseg=nperseg, noverlap=nperseg // 2)
        assert max(abs(avg[k] - Pw[k]) for k in range(len(f))) < 1e-9


def test_spectrogram_spectrum_scaling_equals_welch():
    # the 'spectrum' (V^2) scaling path must also match a Welch spectrum-scaled PSD
    x = white_noise(4096, 1.0, seed=23)
    f, t, Sxx = spectrogram(x, 1.0, "hann", nperseg=256, noverlap=128, scaling="spectrum")
    avg = [math.fsum(Sxx[fr][k] for fr in range(len(Sxx))) / len(Sxx) for k in range(len(f))]
    fw, Pw = welch(x, 1.0, "hann", nperseg=256, noverlap=128, scaling="spectrum")
    assert max(abs(avg[k] - Pw[k]) for k in range(len(f))) < 1e-9


# ===========================================================================
# 5. PARAMETRIC SPECTRA
# ===========================================================================
def test_levinson_durbin_recovers_ar1():
    phi = 0.6
    r = [phi ** k for k in range(5)]                     # AR(1) autocovariance (r0=1)
    a, sigma2, refl = levinson_durbin(r, 4)
    assert abs(a[0] - phi) < 1e-9
    assert max(abs(a[i]) for i in range(1, 4)) < 1e-9
    assert abs(refl[0] - phi) < 1e-9
    assert abs(sigma2 - (1 - phi * phi)) < 1e-9


def test_levinson_solves_yule_walker_system():
    r = [4.0, 1.5, -0.5, 0.25]
    order = 3
    a, _, _ = levinson_durbin(r, order)
    for m in range(1, order + 1):                        # R a = r  (rows 1..p)
        lhs = math.fsum(a[j - 1] * r[abs(m - j)] for j in range(1, order + 1))
        assert abs(lhs - r[m]) < 1e-9


def test_burg_and_yule_walker_resolve_a_tone():
    # a realistic (lightly noisy) tone -- a perfectly noiseless sine drives Burg's
    # prediction error to machine-zero (poles exactly on the unit circle), which is
    # a degenerate case no AR estimator handles meaningfully.
    fs, n = 1000.0, 512
    tone = sine_wave(123.0, n, fs)
    x = [tone[i] + 0.05 * white_noise(n, 1.0, seed=9)[i] for i in range(n)]
    for a, s2 in (burg(x, 20), ar_yule_walker(x, 20)):
        f, P = ar_psd(a, s2, nfft=1024, fs=fs)
        kpk = max(range(len(P)), key=lambda k: P[k])
        assert abs(f[kpk] - 123.0) < 5.0


def test_burg_and_yule_walker_recover_known_ar_coefficients():
    # both estimators must recover the coefficients of a known AR(4) process from data.
    a_true = [0.3, -0.2, 0.15, -0.1]
    nz = white_noise(8000, 1.0, seed=33)
    y = [0.0] * 8000
    for t in range(8000):
        y[t] = math.fsum(a_true[i] * y[t - 1 - i] for i in range(4) if t - 1 - i >= 0) + nz[t]
    ab, _ = burg(y, 4)
    ay, _ = ar_yule_walker(y, 4)
    assert max(abs(ab[i] - a_true[i]) for i in range(4)) < 0.06
    assert max(abs(ay[i] - a_true[i]) for i in range(4)) < 0.06


def test_levinson_reflection_coeffs_cut_off_after_ar_order():
    # the reflection (PARCOR / partial autocorrelation) coefficients of an AR(2) process
    # must vanish beyond lag 2 -- the defining property of the PACF.
    a2 = [0.5, -0.3]
    nz = white_noise(20000, 1.0, seed=34)
    y = [0.0] * 20000
    for t in range(20000):
        y[t] = math.fsum(a2[i] * y[t - 1 - i] for i in range(2) if t - 1 - i >= 0) + nz[t]
    yd = detrend(y, "constant")
    n = len(yd)
    r = [math.fsum(yd[t] * yd[t - k] for t in range(k, n)) / n for k in range(9)]
    _, _, refl = levinson_durbin(r, 8)
    assert max(abs(v) for v in refl[2:]) < 0.05


def test_hermitian_eig_handles_degenerate_eigenvalues():
    # a repeated eigenvalue (2-D eigenspace) is where Jacobi can fail to converge; the
    # decomposition must still reproduce A on every eigenpair.
    A = [[3 + 0j, 1 + 1j, 1 - 1j],
         [1 - 1j, 3 + 0j, 1 + 1j],
         [1 + 1j, 1 - 1j, 3 + 0j]]                        # circulant Hermitian
    evals, evecs = sp._hermitian_eig(A)
    for j in range(3):
        v = [evecs[i][j] for i in range(3)]
        Av = [sum(A[i][k] * v[k] for k in range(3)) for i in range(3)]
        assert max(abs(Av[i] - evals[j] * v[i]) for i in range(3)) < 1e-8


def test_music_resolves_three_tones():
    fs, n = 1000.0, 400
    x = [sine_wave(60.0, n, fs)[i] + sine_wave(150.0, n, fs)[i]
         + sine_wave(300.0, n, fs)[i] for i in range(n)]
    f, P = music(x, n_signals=6, nfft=1024, fs=fs)       # 3 real tones -> 6 exponentials
    pk = sorted(find_peaks(P), key=lambda i: P[i], reverse=True)[:3]
    got = sorted(round(f[i]) for i in pk)
    assert all(min(abs(g - t) for t in (60, 150, 300)) < 6 for g in got)


def test_music_resolves_two_real_tones():
    fs, n = 1000.0, 256
    x = [sine_wave(50.0, n, fs)[i] + sine_wave(120.0, n, fs)[i] for i in range(n)]
    f, P = music(x, n_signals=4, nfft=512, fs=fs)        # 2 real tones -> 4 exponentials
    pk = sorted(find_peaks(P), key=lambda i: P[i], reverse=True)[:2]
    got = sorted(round(f[i]) for i in pk)
    assert abs(got[0] - 50) <= 2 and abs(got[1] - 120) <= 2


def test_capon_resolves_tone():
    fs, n = 1000.0, 256
    x = sine_wave(90.0, n, fs)
    f, P = capon(x, nfft=512, fs=fs, p=20)
    kpk = max(range(len(P)), key=lambda k: P[k])
    assert abs(f[kpk] - 90.0) < 5.0


def test_capon_and_ar_estimators_are_scale_invariant():
    # capon must not raise "singular" and AR coefficients must not vanish for tiny
    # amplitudes: the scale-blind _EPS thresholds in _herm_inv / levinson_durbin / burg
    # (E and pivots scale with signal POWER ~ c^2) failed on small-amplitude data.
    fs, n = 1000.0, 512
    x = [sine_wave(90.0, n, fs)[i] + 0.2 * white_noise(n, 1.0, seed=3)[i] for i in range(n)]
    _, P0 = capon(x, 512, fs, p=20)
    kp0 = max(range(len(P0)), key=lambda i: P0[i])
    ay0, _ = ar_yule_walker(x, 4)
    ab0, _ = burg(x, 4)
    for c in (1e-8, 1e8):
        xs = [c * v for v in x]
        _, Pc = capon(xs, 512, fs, p=20)                 # must not raise
        assert max(range(len(Pc)), key=lambda i: Pc[i]) == kp0
        ayc, _ = ar_yule_walker(xs, 4)
        abc, _ = burg(xs, 4)
        assert max(abs(ay0[i] - ayc[i]) for i in range(4)) < 1e-6      # coeffs scale-invariant
        assert max(abs(ab0[i] - abc[i]) for i in range(4)) < 1e-6


def test_parametric_estimators_handle_hard_covariances():
    # near-degenerate and coherent sources give ill-conditioned / rank-deficient
    # covariances; MUSIC must stay finite (forward-backward averaging), and high-order
    # AR on a pure sinusoid (poles on the unit circle) must not blow up.
    fs, n = 1000.0, 512
    close = [sine_wave(100.0, n, fs)[i] + sine_wave(102.0, n, fs)[i]
             + 0.05 * white_noise(n, 1.0, seed=1)[i] for i in range(n)]      # 2 Hz apart
    _, Pm = music(close, 4, 2048, fs, p=40)
    assert all(math.isfinite(v) for v in Pm)
    tone = sine_wave(120.0, n, fs)                                           # coherent pair
    coh = [tone[i] + (tone[i - 10] if i >= 10 else 0.0)
           + 0.02 * white_noise(n, 1.0, seed=2)[i] for i in range(n)]
    _, Pc = music(coh, 4, 1024, fs, p=30)
    assert all(math.isfinite(v) for v in Pc)
    pure = sine_wave(77.0, 256, fs)                                          # poles on unit circle
    for order in (20, 50):
        a_y, s_y = ar_yule_walker(pure, order)
        a_b, s_b = burg(pure, order)
        assert all(math.isfinite(v) for v in a_y + a_b) and s_y >= 0 and s_b >= 0


def test_music_resolves_near_degenerate_tones():
    # resolving two tones only 2 Hz apart at order p=24 requires the forward-backward
    # (modified-covariance) estimate; a plain forward-only covariance yields a spurious
    # peak and misses one tone.
    fs, n = 1000.0, 256
    x = [sine_wave(100.0, n, fs)[i] + sine_wave(102.0, n, fs)[i]
         + 0.05 * white_noise(n, 1.0, seed=3)[i] for i in range(n)]
    f, P = music(x, 4, 2048, fs, p=24)
    got = sorted(round(f[i]) for i in sorted(find_peaks(P), key=lambda i: P[i], reverse=True)[:2])
    assert got == [100, 102]


def test_ar_psd_onesided_integrates_to_variance():
    # a one-sided PSD must integrate to the signal variance (interior bins folded x2),
    # consistent with periodogram / welch / multitaper; ar_psd previously integrated to
    # HALF the variance because it did not fold.
    a_true = [0.6, -0.5]
    nz = white_noise(4000, 1.0, seed=71)
    y = [0.0] * 4000
    for t in range(4000):
        y[t] = math.fsum(a_true[i] * y[t - 1 - i] for i in range(2) if t - 1 - i >= 0) + nz[t]
    var = math.fsum(v * v for v in detrend(y, "constant")) / len(y)
    a, s2 = burg(y, 2)
    f1, P1 = ar_psd(a, s2, nfft=2048, fs=1.0, onesided=True)
    f2, P2 = ar_psd(a, s2, nfft=2048, fs=1.0, onesided=False)
    assert abs(math.fsum(P1) * (f1[1] - f1[0]) - var) / var < 0.1     # one-sided -> variance
    assert abs(math.fsum(P2) * (1.0 / 2048) - var) / var < 0.1        # two-sided -> variance


def test_parametric_spectra_stay_finite_on_clean_signals():
    # ar_psd / music / capon must stay FINITE even for a noise-free signal: a sharp
    # pole / null must not emit inf, which would crash a downstream find_peaks.
    fs = 1000.0
    clean = [sine_wave(50.0, 256, fs)[i] + sine_wave(120.0, 256, fs)[i] for i in range(256)]
    a, s2 = burg(clean, 20)
    _, Pa = ar_psd(a, s2, nfft=512, fs=fs)
    _, Pm = music(clean, n_signals=4, nfft=512, fs=fs)
    _, Pc = capon(clean, nfft=512, fs=fs, p=20)
    for P in (Pa, Pm, Pc):
        assert all(math.isfinite(v) for v in P)
        find_peaks(P)                                    # must not raise on non-finite input


def test_hermitian_eig_known_matrix_and_reconstruction():
    A = [[2 + 0j, 1 + 0j], [1 + 0j, 2 + 0j]]             # eigenvalues 1 and 3
    evals, evecs = sp._hermitian_eig(A)
    assert sorted(round(e, 9) for e in evals) == [1.0, 3.0]
    # A v = lambda v for each eigenpair
    for j in range(2):
        v = [evecs[i][j] for i in range(2)]
        Av = [sum(A[i][k] * v[k] for k in range(2)) for i in range(2)]
        lv = [evals[j] * v[i] for i in range(2)]
        assert max(abs(Av[i] - lv[i]) for i in range(2)) < 1e-8


def test_hermitian_eig_complex_hermitian():
    # a genuinely COMPLEX Hermitian matrix (non-zero imaginary off-diagonals): the
    # rotations must actually diagonalise it.  The earlier half-phase rotation kept the
    # eigenvectors unitary but did NOT diagonalise A -> eigenvalues were wrong.
    A = [[2 + 0j, 1 + 1j, 0 + 0j],
         [1 - 1j, 3 + 0j, 2 - 1j],
         [0 + 0j, 2 + 1j, 1 + 0j]]                        # Hermitian, trace = 6
    evals, evecs = sp._hermitian_eig(A)
    assert abs(math.fsum(evals) - 6.0) < 1e-9            # eigenvalues sum to the trace
    for j in range(3):
        v = [evecs[i][j] for i in range(3)]
        Av = [sum(A[i][k] * v[k] for k in range(3)) for i in range(3)]
        assert max(abs(Av[i] - evals[j] * v[i]) for i in range(3)) < 1e-8
        assert abs(math.fsum(abs(v[i]) ** 2 for i in range(3)) - 1.0) < 1e-9


# ===========================================================================
# 6. MULTITAPER / DPSS
# ===========================================================================
def test_dpss_orthonormal_and_symmetric():
    tapers = dpss(128, nw=4.0, k=5)
    for t in tapers:
        assert abs(math.fsum(v * v for v in t) - 1.0) < 1e-9           # unit energy
    for i in range(5):
        for j in range(i + 1, 5):
            dot = math.fsum(tapers[i][k] * tapers[j][k] for k in range(128))
            assert abs(dot) < 1e-6                                     # orthogonal
    t0 = tapers[0]                                                     # order-0 symmetric
    assert max(abs(t0[k] - t0[127 - k]) for k in range(64)) < 1e-6


def test_dpss_concentration_decreases_with_order():
    n, nw, k = 256, 4.0, 5
    W = nw / n
    tapers = dpss(n, nw, k)
    conc = []
    for t in tapers:
        X = rfft(t)
        pw = [abs(v) ** 2 for v in X]
        total = math.fsum(pw)
        band = math.fsum(pw[kk] for kk in range(len(pw)) if kk / n <= W)
        conc.append(band / total)
    assert conc[0] > 0.99
    for i in range(k - 1):
        assert conc[i] >= conc[i + 1] - 1e-9             # monotonically decreasing


def test_multitaper_locates_tone_and_beats_periodogram_variance():
    fs, n = 1000.0, 1024
    tone = sine_wave(77.0, n, fs)
    noisy = [tone[i] + 0.5 * white_noise(n, 1.0, seed=3)[i] for i in range(n)]
    fm, Pm = multitaper(noisy, fs, nw=4.0)
    assert abs(fm[max(range(len(Pm)), key=lambda k: Pm[k])] - 77.0) < 3.0
    # variance: multitaper is smoother than the periodogram in a signal-free band
    x = white_noise(n, 1.0, seed=12)
    _, Pp = periodogram(x, fs=fs, window="boxcar")
    _, Pmt = multitaper(x, fs, nw=4.0)

    def cv(P):
        m = math.fsum(P) / len(P)
        return math.sqrt(math.fsum((p - m) ** 2 for p in P) / len(P)) / m
    assert cv(Pmt) < cv(Pp)


def test_multitaper_twosided_is_genuinely_two_sided():
    # onesided=False must return a FULL-length, symmetric two-sided PSD; the parameter
    # used to be a no-op that returned the one-sided estimate either way.
    n = 1024
    x = white_noise(n, 2.0, seed=41)
    f1, P1 = multitaper(x, 10.0, 4.0, onesided=True)
    f2, P2 = multitaper(x, 10.0, 4.0, onesided=False)
    assert len(P1) == n // 2 + 1 and len(P2) == n
    assert max(abs(P2[k] - P2[n - k]) for k in range(1, n // 2)) < 1e-9   # symmetric (real)
    var = math.fsum(v * v for v in detrend(x, "constant")) / n
    assert abs(math.fsum(P1) * (f1[1] - f1[0]) - var) / var < 0.05         # both integrate
    assert abs(math.fsum(P2) * (10.0 / n) - var) / var < 0.05              # to the variance


# ===========================================================================
# 7. FILTERING
# ===========================================================================
def _fir_response(h, f, fs):
    return sum(h[i] * cmath.exp(-2j * math.pi * f * i / fs) for i in range(len(h)))


def test_firwin_lowpass_dc_gain_unity_and_kills_nyquist():
    fs = 1000.0
    h = firwin(65, 100.0, fs, "hamming", pass_zero=True)
    assert abs(math.fsum(h) - 1.0) < 1e-9                             # H(0) = 1
    assert abs(_fir_response(h, 100.0, fs)) == pytest.approx(0.5, abs=0.15)  # ~ -6dB @ cutoff
    assert abs(_fir_response(h, fs / 2.0, fs)) < 1e-2                 # H(Nyquist) ~ 0


def test_firwin_highpass_kills_dc():
    fs = 1000.0
    h = firwin(65, 100.0, fs, "hamming", pass_zero=False)
    assert abs(math.fsum(h)) < 1e-2                                   # H(0) ~ 0
    assert abs(abs(_fir_response(h, fs / 2.0, fs)) - 1.0) < 1e-2      # H(Nyquist) ~ 1


def test_firwin_bandpass_passes_middle_blocks_edges():
    fs = 1000.0
    h = firwin(129, [100.0, 300.0], fs, "hann", pass_zero=False)
    assert abs(_fir_response(h, 200.0, fs)) > 0.9                     # mid-band passes
    assert abs(_fir_response(h, 20.0, fs)) < 0.1                      # DC side blocked
    assert abs(_fir_response(h, 450.0, fs)) < 0.1                     # HF side blocked


def test_firwin_even_numtaps_rejects_highpass_and_bandstop():
    # a Type II (even-length) FIR is forced to zero gain at Nyquist, so it CANNOT be a
    # high-pass or band-stop; those must raise (rather than silently returning a filter
    # normalised by a ~0 Nyquist gain).  Low-/band-pass with even taps stay valid.
    with pytest.raises(SpectralError):
        firwin(64, 150.0, fs=1000.0, pass_zero=False)                # even high-pass
    with pytest.raises(SpectralError):
        firwin(64, [100.0, 300.0], fs=1000.0, pass_zero=True)        # even band-stop
    assert len(firwin(64, 150.0, fs=1000.0)) == 64                   # even low-pass OK
    assert len(firwin(64, [100.0, 300.0], fs=1000.0, pass_zero=False)) == 64  # band-pass OK


def test_linear_phase_fir_has_constant_group_delay():
    # a symmetric FIR is exactly linear-phase: group delay == (numtaps-1)/2
    h = firwin(51, 0.2, fs=1.0, window="hamming")
    f, gd = group_delay(h, [1.0], worN=256, fs=1.0)
    lo = [gd[k] for k in range(len(f)) if f[k] < 0.08]               # passband only
    assert max(abs(g - 25.0) for g in lo) < 1e-3


def test_butter_section_count():
    assert len(butter(4, 100.0, 1000.0, "low")) == 2                  # two biquads
    assert len(butter(5, 100.0, 1000.0, "low")) == 3                  # + first-order
    assert len(butter(6, 100.0, 1000.0, "high")) == 3


def test_butter_is_minus_3db_at_cutoff_and_unity_at_dc():
    fs, fc, n = 1000.0, 100.0, 4096
    sos = butter(4, fc, fs, "low")
    h = sosfilt(sos, impulse(n))                                      # impulse response
    H0 = math.fsum(h)                                                 # H(0)
    Hc = sum(h[i] * cmath.exp(-2j * math.pi * fc * i / fs) for i in range(n))
    assert abs(H0 - 1.0) < 1e-3                                       # unity DC gain
    assert abs(abs(Hc) - 1.0 / math.sqrt(2.0)) < 1e-2                 # -3 dB at cutoff


def test_butter_highpass_blocks_dc():
    fs, fc, n = 1000.0, 100.0, 4096
    sos = butter(4, fc, fs, "high")
    h = sosfilt(sos, impulse(n))
    assert abs(math.fsum(h)) < 1e-2                                   # H(0) ~ 0


def test_butter_bandpass_via_highpass_lowpass_cascade():
    # the documented way to get a band-pass: cascade a high-pass and a low-pass
    fs, n = 1000.0, 4096
    hp = butter(4, 100.0, fs, "high")
    lp = butter(4, 300.0, fs, "low")
    h = sosfilt(lp, sosfilt(hp, impulse(n)))
    def mag(fq):
        return abs(sum(h[i] * cmath.exp(-2j * math.pi * fq * i / fs) for i in range(n)))
    assert mag(200.0) > 0.7                                           # passband
    assert mag(50.0) < 0.2 and mag(450.0) < 0.3                       # both stopbands


def test_butter_high_order_is_stable():
    # very high order Butterworth (bilinear-transform biquads) must stay finite and
    # exactly -3 dB at cutoff -- no coefficient blow-up.
    fs, fc, n = 1000.0, 100.0, 4096
    for order in (16, 24):
        sos = butter(order, fc, fs, "low")
        assert all(math.isfinite(v) for row in sos for v in row)
        h = sosfilt(sos, impulse(n))
        Hc = abs(sum(h[i] * cmath.exp(-2j * math.pi * fc * i / fs) for i in range(n)))
        assert abs(Hc - 1.0 / math.sqrt(2.0)) < 0.05


def test_lfilter_fir_equals_convolution():
    b = [0.2, 0.3, 0.5, -0.1]
    x = white_noise(64, 1.0, seed=5)
    y = lfilter(b, [1.0], x)
    ref = convolve(b, x, "full")[: len(x)]
    assert _maxabs(y, ref) < 1e-12


def test_lfilter_leading_coefficient_normalised():
    # a[0] != 1 must be divided through: 2 y = x  ->  y = x/2
    y = lfilter([1.0], [2.0], [4.0, 8.0, 2.0])
    assert y == pytest.approx([2.0, 4.0, 1.0])


def test_filtfilt_zero_phase_preserves_symmetry_and_length():
    # a zero-phase filter maps a symmetric signal to a symmetric signal (a causal
    # filter would skew it); use an odd length so the symmetry axis is a sample.
    n = 401
    c = (n - 1) / 2.0
    x = [math.exp(-((i - c) / 20.0) ** 2) for i in range(n)]   # symmetric bump
    h = firwin(31, 0.2, fs=1.0, window="hamming")
    y = filtfilt(h, [1.0], x)
    assert len(y) == n
    assert max(abs(y[i] - y[n - 1 - i]) for i in range(n)) < 1e-9


def test_filtfilt_preserves_symmetry_with_nonzero_edges():
    # zero-phase filtering maps a symmetric signal to a symmetric one.  Crucially the
    # signal is NON-ZERO at the ends, which exposes the trailing-edge odd reflection: the
    # old code duplicated x[-1] (off by one), so the padded signal was asymmetric and the
    # output lost its symmetry.  A FIR filter (no initial-condition transient) makes the
    # symmetry exact, isolating the padding.
    n = 401
    c = (n - 1) / 2.0
    x = [1.0 + 0.5 * math.cos(2 * math.pi * 3 * (i - c) / n) for i in range(n)]  # symmetric, ~1 at ends
    y = filtfilt(firwin(21, 0.3), [1.0], x)
    assert max(abs(y[i] - y[n - 1 - i]) for i in range(n)) < 1e-9


def test_freqz_matches_direct_evaluation():
    b = [0.25, 0.5, 0.25]
    f, H = freqz(b, [1.0], worN=32, fs=2.0)
    for k in range(len(f)):
        assert abs(H[k] - _fir_response(b, f[k], 2.0)) < 1e-9


def test_freqz_and_group_delay_of_a_pure_delay():
    # a pure delay H(z) = z^-m has exact linear phase e^{-j w m} and constant group delay m
    m = 7
    b = [0.0] * m + [1.0]
    fq, H = freqz(b, [1.0], 256, 2.0)
    for k in range(1, len(fq)):
        w = math.pi * fq[k]
        assert abs(H[k] - cmath.exp(-1j * w * m)) < 1e-9
    _, gd = group_delay(b, [1.0], 256, 2.0)
    assert max(abs(gd[k] - m) for k in range(5, 250)) < 1e-3


def test_hilbert_instantaneous_frequency_tracks_a_chirp():
    # the derivative of the analytic signal's unwrapped phase is the instantaneous
    # frequency; for a linear chirp it must rise linearly (sampled away from the edges
    # where the FFT-based Hilbert transform has transients).
    fs, n = 2000.0, 4000
    c = chirp(50.0, 400.0, n, fs, "linear")
    z = hilbert(c)
    ph = unwrap([cmath.phase(v) for v in z])
    inst = [(ph[i + 1] - ph[i]) / (2 * math.pi) * fs for i in range(n - 1)]
    mid1 = sum(inst[900:1100]) / 200                     # t~0.5s -> 50 + 175*0.5 = 137.5
    mid2 = sum(inst[2900:3100]) / 200                    # t~1.5s -> 50 + 175*1.5 = 312.5
    assert abs(mid1 - 137.5) < 12 and abs(mid2 - 312.5) < 12
    assert mid1 < mid2                                    # monotonically increasing


def test_hilbert_analytic_signal_properties():
    fs, n, f0 = 1000.0, 500, 50.0                        # 50 Hz is EXACT bin 25 (periodic)
    x = cosine(f0, n, fs)
    z = hilbert(x)
    assert max(abs(z[i].real - x[i]) for i in range(n)) < 1e-9        # real part = x
    s = sine_wave(f0, n, fs)                                          # 90-deg shift
    assert max(abs(z[i].imag - s[i]) for i in range(n)) < 1e-7        # imag = H{cos} = sin
    env = envelope(x)                                                 # envelope = const
    assert max(abs(env[i] - 1.0) for i in range(n)) < 1e-7


def test_hilbert_has_zero_negative_frequency_content():
    # the DEFINING property of the analytic signal: its spectrum vanishes at negative
    # frequencies (bins n//2+1 .. n-1) and is retained at positive ones.
    rng = random.Random(7)
    for n in (64, 65, 128, 127):
        x = [rng.gauss(0, 1) for _ in range(n)]
        Z = fft(hilbert(x))
        assert max(abs(Z[k]) for k in range(n // 2 + 1, n)) < 1e-11     # negative freqs killed
        assert max(abs(Z[k]) for k in range(1, n // 2)) > 1.0           # positive freqs kept


def test_group_delay_is_additive_for_a_cascade():
    # cascading filters adds their group delays: gd(b1 * b2) == gd(b1) + gd(b2)
    b1, b2 = firwin(21, 0.3), firwin(15, 0.2)
    bc = convolve(b1, b2)
    f, g1 = group_delay(b1, [1.0], 256)
    _, g2 = group_delay(b2, [1.0], 256)
    _, gc = group_delay(bc, [1.0], 256)
    assert max(abs(gc[k] - (g1[k] + g2[k])) for k in range(5, 250)) < 1e-6


# ===========================================================================
# 8. CORRELATION / CONVOLUTION
# ===========================================================================
def test_convolve_polynomial_multiplication():
    assert convolve([1, 1], [1, 1]) == pytest.approx([1, 2, 1])
    assert convolve([1, 2, 3], [0, 1]) == pytest.approx([0, 1, 2, 3])


def test_convolve_modes_lengths():
    a, b = [1, 2, 3, 4, 5], [1, 1, 1]
    assert len(convolve(a, b, "full")) == 7
    assert len(convolve(a, b, "same")) == 5
    assert len(convolve(a, b, "valid")) == 3


def test_convolve_fft_path_matches_direct_for_large_inputs():
    rng = random.Random(21)
    a = [rng.gauss(0, 1) for _ in range(200)]            # 200*200 > 4096 -> FFT path
    b = [rng.gauss(0, 1) for _ in range(200)]
    direct = [0.0] * (len(a) + len(b) - 1)
    for i in range(len(a)):
        for j in range(len(b)):
            direct[i + j] += a[i] * b[j]
    assert _maxabs(convolve(a, b, "full"), direct) < 1e-8


def test_convolution_associativity_and_lfilter_linearity():
    rng = random.Random(70)
    a = [rng.gauss(0, 1) for _ in range(30)]
    b = [rng.gauss(0, 1) for _ in range(12)]
    c = [rng.gauss(0, 1) for _ in range(8)]
    lhs = convolve(convolve(a, b), c)                    # (a*b)*c == a*(b*c)
    rhs = convolve(a, convolve(b, c))
    assert max(abs(lhs[i] - rhs[i]) for i in range(len(lhs))) < 1e-9
    x1 = white_noise(200, 1.0, seed=3)                   # lfilter is linear
    x2 = white_noise(200, 1.0, seed=4)
    bq, aq = [0.2, 0.3, 0.1], [1.0, -0.5, 0.2]
    al, be = 1.7, -2.3
    yc = lfilter(bq, aq, [al * x1[i] + be * x2[i] for i in range(200)])
    y1, y2 = lfilter(bq, aq, x1), lfilter(bq, aq, x2)
    assert max(abs(yc[i] - (al * y1[i] + be * y2[i])) for i in range(200)) < 1e-9


def test_correlate_identity_and_autocorr_matches_direct():
    x = [1.0, 2.0, 3.0, 4.0]
    assert correlate(x, [1.0]) == pytest.approx(x)
    r = autocorrelate(x, maxlag=3, normalize=False)
    direct = [math.fsum(x[t] * x[t - k] for t in range(k, len(x))) for k in range(4)]
    assert _maxabs(r, direct) < 1e-8
    rn = autocorrelate(x, normalize=True)
    assert abs(rn[0] - 1.0) < 1e-12                       # normalised peak at lag 0


# ===========================================================================
# 9. WAVELETS
# ===========================================================================
def test_haar_known_coefficients():
    cA, cD = dwt([1.0, 2.0, 3.0, 4.0], "haar")
    r = math.sqrt(0.5)
    assert cA == pytest.approx([r * 3, r * 7])
    assert cD == pytest.approx([r * -1, r * -1])


def test_dwt_idwt_perfect_reconstruction():
    rng = random.Random(31)
    x = [rng.gauss(0, 1) for _ in range(64)]
    for wav in ("haar", "db2", "db4"):
        cA, cD = dwt(x, wav)
        assert _maxabs(idwt(cA, cD, wav), x) < 1e-9


def test_wavedec_waverec_multilevel_roundtrip():
    rng = random.Random(33)
    x = [rng.gauss(0, 1) for _ in range(128)]
    for wav in ("haar", "db2", "db4"):
        coeffs = wavedec(x, wav, level=3)
        assert len(coeffs) == 4                           # cA3 + cD3 + cD2 + cD1
        assert _maxabs(waverec(coeffs, wav), x) < 1e-9


def test_dwt_preserves_energy():
    rng = random.Random(35)
    x = [rng.gauss(0, 1) for _ in range(64)]
    for wav in ("haar", "db2", "db4"):
        cA, cD = dwt(x, wav)
        e = math.fsum(v * v for v in cA) + math.fsum(v * v for v in cD)
        assert abs(e - math.fsum(v * v for v in x)) < 1e-9


def test_wavelet_filters_are_orthonormal_qmf():
    # the scaling filter h and the quadrature-mirror wavelet filter g must form an
    # orthonormal QMF pair: |h|=1, h _|_ g, and h is orthogonal to its even shifts.
    for wav in ("haar", "db2", "db4"):
        h = sp._WAVELETS[wav]
        g = sp._qmf(h)
        L = len(h)
        assert abs(math.fsum(v * v for v in h) - 1.0) < 1e-12          # unit energy
        assert abs(math.fsum(h[i] * g[i] for i in range(L))) < 1e-12   # h perpendicular to g
        for k in range(1, L // 2):                                     # double-shift orthogonality
            assert abs(math.fsum(h[i] * h[i + 2 * k] for i in range(L - 2 * k))) < 1e-11


def test_wavedec_waverec_reconstructs_arbitrary_length():
    # multi-level DWT/IDWT must reconstruct signals of ANY length, not just powers of
    # two.  Odd-length levels pad by a sample on the way down; the reconstruction has to
    # trim back (db2/db4 previously corrupted the interior via the circular wrap).
    rng = random.Random(41)
    for n in (50, 63, 100, 130, 255):
        x = [rng.gauss(0, 1) for _ in range(n)]
        for wav in ("haar", "db2", "db4"):
            for level in (1, 2, 3):
                xr = waverec(wavedec(x, wav, level=level), wav)
                assert len(xr) >= n
                assert max(abs(xr[i] - x[i]) for i in range(n)) < 1e-9


def test_wavedec_reconstructs_beyond_max_level():
    # requesting more levels than log2(n) must not crash or corrupt the reconstruction
    rng = random.Random(88)
    for nn, lvl in ((8, 5), (16, 6), (100, 8), (5, 6)):
        x = [rng.gauss(0, 1) for _ in range(nn)]
        xr = waverec(wavedec(x, "db2", lvl), "db2")
        assert len(xr) >= nn
        assert max(abs(xr[i] - x[i]) for i in range(nn)) < 1e-9


def test_morlet_symmetry_and_cwt_detects_scale():
    m = morlet(201, w=5.0, s=1.0)                         # odd -> symmetry axis on a sample
    p = len(m)
    assert max(abs(m[i].real - m[p - 1 - i].real) for i in range(p)) < 1e-12   # real even
    assert max(abs(m[i].imag + m[p - 1 - i].imag) for i in range(p)) < 1e-12   # imag odd
    fs, n = 200.0, 400
    x = sine_wave(20.0, n, fs)
    scales = [2.0, 4.0, 8.0, 16.0]
    rows = cwt(x, scales, w=5.0)
    energy = [math.fsum(abs(v) ** 2 for v in row) for row in rows]
    best = max(range(len(scales)), key=lambda i: energy[i])
    # analytic Morlet scale<->freq: f = w*fs/(2*pi*s); the peak scale should bracket 20 Hz
    f_best = 5.0 * fs / (2 * math.pi * scales[best])
    assert 8.0 < f_best < 45.0


def test_cwt_localizes_an_impulse_in_time():
    # the CWT magnitude of an impulse must peak at the impulse's time index, at every
    # scale (time localisation, the complement of the frequency-localisation test).
    imp = impulse(400, 200)
    rows = cwt(imp, [4.0, 8.0, 16.0], w=5.0)
    for row in rows:
        loc = max(range(400), key=lambda t: abs(row[t]))
        assert abs(loc - 200) < 20


# ===========================================================================
# 10. SPECTRAL FEATURES
# ===========================================================================
def test_flatness_and_entropy_of_impulse_are_maximal():
    d = impulse(64, 0)                                    # flat magnitude spectrum
    assert abs(spectral_flatness(d) - 1.0) < 1e-6
    assert abs(spectral_entropy(d) - 1.0) < 1e-6


def test_flatness_and_entropy_of_tone_are_minimal():
    x = cosine(8 * 500.0 / 512, 512, 500.0)              # single bin
    assert spectral_flatness(x) < 1e-2
    assert spectral_entropy(x) < 0.1


def test_spectral_centroid_and_dominant_of_a_tone():
    fs, n = 1000.0, 1000                                 # 150 Hz is EXACT bin 150
    x = cosine(150.0, n, fs)
    assert abs(spectral_centroid(x, fs) - 150.0) < 5.0
    assert abs(dominant_frequency(x, fs) - 150.0) < 2.0


def test_dominant_frequency_parabolic_refinement_beats_the_raw_bin():
    # the parabolic sub-bin refinement must move the estimate TOWARD the true off-bin
    # frequency (closer than the nearest FFT bin).  Exact-bin tests leave delta ~ 0, so a
    # wrong-sign refinement slips past them -- a gap that mutation testing exposed.
    fs, n = 1000.0, 512                                  # bin spacing ~1.95 Hz
    for f0 in (100.3, 137.8, 183.2):                     # deliberately off-bin
        x = [math.cos(2 * math.pi * f0 * i / fs) for i in range(n)]
        est = dominant_frequency(x, fs)
        nearest_bin = round(f0 * n / fs) * fs / n
        assert abs(est - f0) < abs(nearest_bin - f0)     # refinement beats the raw bin
        assert abs(est - f0) < 0.6                        # and lands within ~1/3 of a bin


def test_rolloff_monotone_in_fraction():
    x = white_noise(1024, 1.0, seed=14)
    assert spectral_rolloff(x, 1.0, 0.5) <= spectral_rolloff(x, 1.0, 0.95) + 1e-9


def test_spectral_bandwidth_equals_half_separation_of_symmetric_tones():
    # two equal-amplitude tones at EXACT bins symmetric about the centroid -> the p=2
    # spectral bandwidth (RMS spread) equals their half-separation exactly.
    fs, n = 1000.0, 1000                                 # 100 & 300 Hz are exact bins
    x = [sine_wave(100.0, n, fs)[i] + sine_wave(300.0, n, fs)[i] for i in range(n)]
    assert abs(spectral_centroid(x, fs) - 200.0) < 1e-6
    assert abs(spectral_bandwidth(x, fs) - 100.0) < 1.0   # half-separation (mutant -> ~21)


def test_spectral_rolloff_value_and_strict_monotonicity():
    # for a low-pass signal the roll-off frequency sits well BELOW Nyquist, and increases
    # STRICTLY with the roll fraction (must not collapse to the last bin).
    fs, n = 1000.0, 1024
    lp = [sine_wave(30.0, n, fs)[i] + sine_wave(60.0, n, fs)[i] for i in range(n)]
    assert spectral_rolloff(lp, fs, 0.85) < 100.0         # far below Nyquist (500)
    assert spectral_rolloff(lp, fs, 0.5) < spectral_rolloff(lp, fs, 0.95)   # strict


def test_thd_pure_tone_zero_and_with_known_harmonic():
    fs, n = 2000.0, 2048
    f0 = 20 * fs / n                                      # exact bin
    pure = cosine(f0, n, fs)
    assert thd(pure, fs) < 1e-3
    withharm = [pure[i] + 0.25 * cosine(2 * f0, n, fs)[i] for i in range(n)]
    assert abs(thd(withharm, fs) - 0.25) < 1e-2          # single 2nd harmonic @ 25%


def test_snr_matches_formula():
    sig = [1.0] * 100                                     # power = 100
    noise = [0.1] * 100                                   # power = 1
    assert abs(snr(sig, noise) - 20.0) < 1e-9            # 10 log10(100) = 20 dB


def test_find_peaks_locations_and_filters():
    x = [0, 1, 0, 3, 0, 1, 0, 2, 0]
    assert find_peaks(x) == [1, 3, 5, 7]
    assert find_peaks(x, height=1.5) == [3, 7]
    assert find_peaks(x, distance=3) == [3, 7]            # keep tallest, 3 apart


def test_spectral_flux_zero_for_identical_positive_otherwise():
    a = white_noise(256, 1.0, seed=1)
    b = white_noise(256, 1.0, seed=2)
    assert spectral_flux(a, a) < 1e-9
    assert spectral_flux(a, b) > 0.0


def test_spectral_features_are_scale_invariant():
    # normalised spectral quantities (flatness / entropy / crest / centroid / bandwidth
    # / rolloff / dominant freq / normalised autocorrelation) must NOT depend on the
    # signal amplitude -- an absolute _EPS threshold made them collapse for tiny
    # (e.g. microvolt) signals.  Checked across 16 orders of magnitude.
    fs, n = 1000.0, 2048
    base = [sine_wave(50.0, n, fs)[i] + 0.5 * sine_wave(120.0, n, fs)[i]
            + 0.1 * white_noise(n, 1.0, seed=1)[i] for i in range(n)]
    feats = [
        ("flatness", lambda z: spectral_flatness(z)),
        ("entropy", lambda z: spectral_entropy(z)),
        ("crest", lambda z: spectral_crest(z)),
        ("centroid", lambda z: spectral_centroid(z, fs)),
        ("bandwidth", lambda z: spectral_bandwidth(z, fs)),
        ("rolloff", lambda z: spectral_rolloff(z, fs)),
        ("dominant", lambda z: dominant_frequency(z, fs)),
        ("autocorr3", lambda z: autocorrelate(z, 10, True)[3]),
    ]
    for _, fn in feats:
        b = fn(base)
        for c in (1e-8, 1e8):
            v = fn([c * z for z in base])
            assert abs(v - b) <= 1e-6 * max(1.0, abs(b))
    # a normalised autocorrelation is exactly 1 at lag 0 for ANY non-zero amplitude
    tiny = [1e-8 * z for z in base]
    assert abs(autocorrelate(tiny, 5, True)[0] - 1.0) < 1e-9


# ===========================================================================
# 11. UTILITIES & GENERATORS
# ===========================================================================
def test_detrend_constant_and_linear():
    x = [3.0 + 2.0 * t for t in range(10)]               # a pure line
    assert max(abs(v) for v in detrend(x, "linear")) < 1e-9
    c = detrend([5.0, 7.0, 9.0], "constant")
    assert abs(math.fsum(c)) < 1e-12                      # mean removed


def test_resample_identity_and_tone_preserved():
    x = white_noise(64, 1.0, seed=17)
    assert _maxabs(resample(x, 64), x) < 1e-9
    fs, n = 200.0, 128
    tone = sine_wave(10.0, n, fs)
    up = resample(tone, 256)                              # upsample 2x
    assert abs(dominant_frequency(up, 400.0) - 10.0) < 2.0  # same physical frequency


def test_unwrap_removes_jumps():
    wrapped = [0.0, 3.0, -3.0 + 2 * math.pi, 0.1]         # a 2pi wrap in the middle
    u = unwrap([math.atan2(math.sin(p), math.cos(p)) for p in wrapped])
    assert all(abs(u[i + 1] - u[i]) < math.pi for i in range(len(u) - 1))


def test_mag2db_of_tiny_nonzero_is_finite():
    # only EXACT zero is -inf dB; a tiny-but-nonzero amplitude has a large finite dB
    # (an absolute _EPS cutoff punched artificial -inf holes into a dB spectrum).
    assert math.isfinite(mag2db(1e-100)) and abs(mag2db(1e-100) - (-2000.0)) < 1e-6
    assert mag2db(0.0) == -float("inf")
    out = mag2db([1.0, 1e-50, 0.0])
    assert out[0] == 0.0 and math.isfinite(out[1]) and out[2] == -float("inf")


def test_mag2db_db2mag_roundtrip_and_rms():
    for v in (0.5, 1.0, 2.0, 10.0):
        assert abs(db2mag(mag2db(v)) - v) < 1e-9
    assert abs(mag2db(10.0) - 20.0) < 1e-9               # 20 log10(10)
    A = 2.0
    assert abs(rms(sine_wave(5.0, 4000, 1000.0, amp=A)) - A / math.sqrt(2.0)) < 1e-2


def test_zero_pad_and_impulse():
    assert zero_pad([1.0, 2.0], 5) == [1.0, 2.0, 0.0, 0.0, 0.0]
    assert zero_pad([1.0, 2.0, 3.0], 2) == [1.0, 2.0]
    assert impulse(5, 2) == [0.0, 0.0, 1.0, 0.0, 0.0]


def test_generators_values_and_ranges():
    assert sine_wave(0.0, 5, 1.0) == [0.0] * 5
    sq = square_wave(1.0, 20, 8.0, duty=0.5)
    assert all(v in (-1.0, 1.0) for v in sq)
    saw = sawtooth(1.0, 40, 10.0)
    assert all(-1.0 - 1e-9 <= v <= 1.0 + 1e-9 for v in saw)
    ramp = sawtooth(1.0, 200, 200.0, width=1.0)          # a full -1..+1 ramp each period
    assert max(ramp) > 0.9 and min(ramp) < -0.9          # must span the FULL range


def test_chirp_constant_frequency_and_method_validation():
    # a LOGARITHMIC chirp with f0 == f1 is a constant-frequency tone -- it used to
    # divide by log(f1/f0) = log(1) = 0 and raise ZeroDivisionError.
    fs, n = 1000.0, 500
    c = chirp(100.0, 100.0, n, fs, "logarithmic")
    assert all(math.isfinite(v) for v in c)
    assert abs(dominant_frequency(c, fs) - 100.0) < 3.0
    assert all(math.isfinite(v) for v in chirp(80.0, 80.0, 200, fs, "linear"))
    with pytest.raises(SpectralError):
        chirp(1.0, 2.0, 10, fs, "quadratic")                # unknown method
    with pytest.raises(SpectralError):
        chirp(-1.0, 100.0, 10, fs, "logarithmic")           # log chirp needs f0 > 0


def test_resample_nonpositive_length_returns_empty():
    assert resample([1.0, 2.0, 3.0], 0) == []               # used to IndexError
    assert resample([1.0, 2.0, 3.0], -5) == []


def test_features_stay_finite_on_degenerate_signals():
    # a spike / step / constant / very short signal must never yield NaN or inf
    cases = [impulse(64, 10), [0.0] * 32 + [1.0] * 32, [3.0] * 64, [5.0], [1.0, 2.0]]
    for sg in cases:
        vals = [spectral_centroid(sg), spectral_bandwidth(sg), spectral_flatness(sg),
                spectral_rolloff(sg), spectral_entropy(sg), spectral_crest(sg),
                dominant_frequency(sg), thd(sg), rms(sg)]
        assert all(math.isfinite(v) for v in vals)


def test_chirp_frequency_increases_over_time():
    fs, n = 2000.0, 4000
    c = chirp(20.0, 400.0, n, fs, "linear")
    first = dominant_frequency(c[: n // 2], fs)
    second = dominant_frequency(c[n // 2:], fs)
    assert first < second                                 # sweeps upward


def test_noise_generators_are_deterministic_and_normalised():
    assert white_noise(50, 1.0, seed=42) == white_noise(50, 1.0, seed=42)
    assert white_noise(50, 1.0, seed=1) != white_noise(50, 1.0, seed=2)
    p = pink_noise(4096, seed=7)
    assert pink_noise(4096, seed=7) == p                  # reproducible
    mu = math.fsum(p) / len(p)
    var = math.fsum((v - mu) ** 2 for v in p) / len(p)
    assert abs(mu) < 1e-9 and abs(var - 1.0) < 1e-6       # zero-mean, unit-variance


# ===========================================================================
# 12. INPUT VALIDATION & EDGE CASES
# ===========================================================================
def test_non_finite_input_rejected():
    with pytest.raises(SpectralError):
        periodogram([1.0, float("nan"), 2.0])
    with pytest.raises(SpectralError):
        rfft([1.0, float("inf")])


def test_complex_input_is_rejected_by_real_estimators():
    # the real-signal estimators must reject complex data loudly (via _as_floats),
    # not silently drop the imaginary part.
    for call in (lambda: periodogram([1 + 2j, 3 - 1j, 0.5]),
                 lambda: welch([1 + 2j, 3 - 1j, 0.5, 1.0], nperseg=4),
                 lambda: burg([1 + 1j, 2.0, 3.0, 4.0], 2),
                 lambda: rfft([1 + 0j, 2 + 1j])):
        with pytest.raises(SpectralError):
            call()


def test_bad_option_strings_raise():
    with pytest.raises(SpectralError):
        periodogram([1.0, 2.0, 3.0], scaling="bogus")
    with pytest.raises(SpectralError):
        convolve([1.0], [1.0], mode="bogus")
    with pytest.raises(SpectralError):
        detrend([1.0, 2.0], "quadratic")
    with pytest.raises(SpectralError):
        lfilter([1.0], [0.0], [1.0, 2.0])                 # a[0] == 0
    with pytest.raises(SpectralError):
        butter(4, 100.0, 1000.0, "bandpass")              # only low/high
    with pytest.raises(KeyError):
        get_window("no_such_window", 8)


def test_empty_and_singleton_transforms():
    assert fft([]) == []
    assert ifft([]) == []
    assert kaiser(1) == [1.0] and hann(1) == [1.0]
    assert dpss(1, k=1) == [[1.0]]


# ===========================================================================
# 13. DIFFERENTIAL vs NumPy / SciPy  (skipped when unavailable)
# ===========================================================================
def test_fft_matches_numpy():
    np = pytest.importorskip("numpy")
    rng = random.Random(101)
    for n in (16, 17, 60, 63):
        x = [rng.gauss(0, 1) for _ in range(n)]
        ours = fft(x)
        ref = np.fft.fft(np.array(x))
        assert max(abs(ours[k] - complex(ref[k])) for k in range(n)) < 1e-8


def test_rfft_and_irfft_match_numpy():
    np = pytest.importorskip("numpy")
    rng = random.Random(102)
    x = [rng.gauss(0, 1) for _ in range(64)]
    ours = rfft(x)
    ref = np.fft.rfft(np.array(x))
    assert max(abs(ours[k] - complex(ref[k])) for k in range(len(ours))) < 1e-8
    back = irfft(ours, 64)
    assert max(abs(back[i] - x[i]) for i in range(64)) < 1e-8


def test_windows_match_scipy():
    sig = pytest.importorskip("scipy.signal")
    n = 65
    pairs = [(hann(n), "hann"), (hamming(n), "hamming"), (blackman(n), "blackman"),
             (blackmanharris(n), "blackmanharris"), (nuttall(n), "nuttall"),
             (bartlett(n), "bartlett"), (flattop(n), "flattop")]
    for ours, name in pairs:
        ref = sig.get_window(name, n, fftbins=False)     # symmetric
        assert max(abs(ours[i] - ref[i]) for i in range(n)) < 1e-6
    kref = sig.get_window(("kaiser", 8.6), n, fftbins=False)
    ko = kaiser(n, 8.6)
    assert max(abs(ko[i] - kref[i]) for i in range(n)) < 1e-6


def test_firwin_matches_scipy():
    sig = pytest.importorskip("scipy.signal")
    fs = 1000.0
    cases = [                                            # low / high / band / stop
        (firwin(65, 150.0, fs=fs, window="hamming", pass_zero=True),
         sig.firwin(65, 150.0, fs=fs, window="hamming", pass_zero="lowpass")),
        (firwin(65, 150.0, fs=fs, window="hamming", pass_zero=False),
         sig.firwin(65, 150.0, fs=fs, window="hamming", pass_zero="highpass")),
        (firwin(129, [100.0, 300.0], fs=fs, window="hann", pass_zero=False),
         sig.firwin(129, [100.0, 300.0], fs=fs, window="hann", pass_zero="bandpass")),
        (firwin(129, [100.0, 300.0], fs=fs, window="hann", pass_zero=True),
         sig.firwin(129, [100.0, 300.0], fs=fs, window="hann", pass_zero="bandstop")),
    ]
    for ours, ref in cases:
        assert max(abs(ours[i] - ref[i]) for i in range(len(ours))) < 1e-9


def test_dpss_matches_scipy():
    # several (n, nw, k) -- higher-order tapers at large n used to drift to ~1e-6 when
    # the inverse-iteration shift was too coarse; they now track SciPy to ~1e-13.
    sig = pytest.importorskip("scipy.signal")
    worst = 0.0
    for n, nw, k in [(64, 2.5, 4), (128, 3.0, 5), (256, 4.0, 5), (512, 3.5, 6)]:
        ours = dpss(n, nw, k)
        ref = sig.windows.dpss(n, nw, Kmax=k)
        for j in range(k):
            a = ours[j]
            b = list(ref[j])
            d_plus = max(abs(a[i] - b[i]) for i in range(n))
            d_minus = max(abs(a[i] + b[i]) for i in range(n))
            worst = max(worst, min(d_plus, d_minus))     # equal up to a sign
    assert worst < 1e-9


def test_welch_matches_scipy():
    np = pytest.importorskip("numpy")
    sig = pytest.importorskip("scipy.signal")
    x = white_noise(4096, 1.0, seed=55)
    fo, Po = welch(x, fs=1.0, window="hann", nperseg=256, noverlap=128)
    fr, Pr = sig.welch(np.asarray(x), fs=1.0, window="hann", nperseg=256, noverlap=128,
                       detrend="constant", scaling="density")
    assert max(abs(fo[k] - fr[k]) for k in range(len(fo))) < 1e-9
    assert max(abs(Po[k] - Pr[k]) for k in range(len(Po))) < 1e-9


def test_lfilter_matches_scipy():
    sig = pytest.importorskip("scipy.signal")
    b = [0.2, 0.3, 0.1]
    a = [1.0, -0.5, 0.25]
    x = white_noise(200, 1.0, seed=77)
    ours = lfilter(b, a, x)
    ref = sig.lfilter(b, a, x)
    assert max(abs(ours[i] - ref[i]) for i in range(len(x))) < 1e-9


def test_filtfilt_matches_scipy_including_edges():
    np = pytest.importorskip("numpy")
    sig = pytest.importorskip("scipy.signal")
    x = white_noise(600, 1.0, seed=91)                   # non-zero at the ends
    for order, wn in ((4, 0.1), (3, 0.2), (6, 0.15), (8, 0.25)):
        b, a = sig.butter(order, wn)
        ours = filtfilt(list(b), list(a), x)
        ref = sig.filtfilt(b, a, np.asarray(x))
        assert max(abs(ours[i] - ref[i]) for i in range(len(x))) < 1e-6   # edges too


def test_filtfilt_matches_scipy_when_filter_longer_than_signal():
    np = pytest.importorskip("numpy")
    sig = pytest.importorskip("scipy.signal")
    b, a = sig.butter(6, 0.2)                             # 13 taps
    for nn in (3, 7, 13):
        x = white_noise(nn, 1.0, seed=nn)
        ours = filtfilt(list(b), list(a), x)
        ref = sig.filtfilt(b, a, np.asarray(x),
                           padlen=min(3 * max(len(a), len(b)), nn - 1))
        assert len(ours) == nn
        assert max(abs(ours[i] - ref[i]) for i in range(nn)) < 1e-6


def test_hermitian_eig_is_scale_robust():
    # eigenvalues must track numpy across extreme amplitudes -- the covariance from a
    # very large signal used to overflow |a|^2 (fixed by normalising the matrix first).
    np = pytest.importorskip("numpy")
    rng = random.Random(91)
    for mscale in (1e-10, 1e10, 1e150):
        p = 5
        A = [[0j] * p for _ in range(p)]
        for i in range(p):
            A[i][i] = complex(rng.gauss(0, 1) * mscale, 0.0)
            for j in range(i + 1, p):
                z = complex(rng.gauss(0, 1) * mscale, rng.gauss(0, 1) * mscale)
                A[i][j] = z
                A[j][i] = z.conjugate()
        ev, _ = sp._hermitian_eig(A)
        ref = sorted(np.linalg.eigvalsh(np.array(A)))
        scale = max(abs(e) for e in ref)
        assert max(abs(sorted(ev)[i] - ref[i]) for i in range(p)) < 1e-12 * scale


def test_lfilter_zi_and_initial_condition_match_scipy():
    np = pytest.importorskip("numpy")
    sig = pytest.importorskip("scipy.signal")
    b, a = sig.butter(3, 0.2)
    zi_np = sig.lfilter_zi(b, a)
    zi_ours = sp._lfilter_zi(list(b), list(a))
    assert max(abs(zi_ours[i] - zi_np[i]) for i in range(len(zi_np))) < 1e-9
    x = white_noise(150, 1.0, seed=92)
    yo = lfilter(list(b), list(a), x, zi=[z * x[0] for z in zi_ours])
    yr, _ = sig.lfilter(b, a, np.asarray(x), zi=zi_np * x[0])
    assert max(abs(yo[i] - yr[i]) for i in range(len(x))) < 1e-9


def test_hermitian_eig_matches_numpy():
    np = pytest.importorskip("numpy")
    rng = random.Random(101)
    worst = 0.0
    for _ in range(8):
        p = rng.choice([3, 4, 5, 6])
        A = [[0j] * p for _ in range(p)]
        for i in range(p):
            A[i][i] = complex(rng.gauss(0, 1), 0.0)
            for j in range(i + 1, p):
                z = complex(rng.gauss(0, 1), rng.gauss(0, 1))
                A[i][j] = z
                A[j][i] = z.conjugate()
        ev, _ = sp._hermitian_eig(A)
        ref = sorted(np.linalg.eigvalsh(np.array(A)))
        worst = max(worst, max(abs(sorted(ev)[i] - ref[i]) for i in range(p)))
    assert worst < 1e-8


def test_periodogram_matches_scipy():
    np = pytest.importorskip("numpy")
    sig = pytest.importorskip("scipy.signal")
    x = white_noise(1024, 1.0, seed=61)
    fo, Po = periodogram(x, 2.0, "hann", scaling="density")
    fr, Pr = sig.periodogram(np.asarray(x), 2.0, "hann", detrend="constant", scaling="density")
    assert max(abs(Po[k] - Pr[k]) for k in range(len(Po))) < 1e-9


def test_welch_linear_detrend_matches_scipy():
    np = pytest.importorskip("numpy")
    sig = pytest.importorskip("scipy.signal")
    x = [0.01 * i + math.sin(2 * math.pi * 30 * i / 512) for i in range(4096)]   # trend + tone
    fo, Po = welch(x, 1.0, "hann", 256, 128, detrend_type="linear")
    fr, Pr = sig.welch(np.asarray(x), 1.0, "hann", 256, 128, detrend="linear")
    assert max(abs(Po[k] - Pr[k]) for k in range(len(Po))) < 1e-9


def test_welch_median_matches_scipy():
    np = pytest.importorskip("numpy")
    sig = pytest.importorskip("scipy.signal")
    x = white_noise(4096, 1.0, seed=55)
    fo, Po = welch(x, 1.0, "hann", 256, 128, average="median")
    fr, Pr = sig.welch(np.asarray(x), 1.0, "hann", 256, 128, average="median")
    assert max(abs(Po[k] - Pr[k]) for k in range(len(Po))) < 1e-9


def test_csd_and_coherence_match_scipy():
    np = pytest.importorskip("numpy")
    sig = pytest.importorskip("scipy.signal")
    x = white_noise(8192, 1.0, seed=6)
    nz = white_noise(8192, 1.0, seed=7)
    y = [0.6 * x[i] + 0.4 * nz[i] for i in range(len(x))]
    fo, Po = csd(x, y, 1.0, "hann", 256, 128)
    fr, Pr = sig.csd(np.asarray(x), np.asarray(y), 1.0, "hann", 256, 128)
    assert max(abs(Po[k] - complex(Pr[k])) for k in range(len(Po))) < 1e-9
    fo, Co = coherence(x, y, 1.0, "hann", 256)
    fr, Cr = sig.coherence(np.asarray(x), np.asarray(y), 1.0, "hann", 256)
    assert max(abs(Co[k] - Cr[k]) for k in range(len(Co))) < 1e-9


def test_butter_response_matches_scipy():
    np = pytest.importorskip("numpy")
    sig = pytest.importorskip("scipy.signal")
    for order in (1, 2, 4, 5, 8, 10):                    # incl. first-order & high orders
        for bt in ("low", "high"):
            sos_o = butter(order, 100.0, 1000.0, bt)
            wr, Hr = sig.sosfreqz(sig.butter(order, 100.0, btype=bt, fs=1000.0,
                                             output="sos"), worN=256, fs=1000.0)
            Ho = []
            for fq in wr:
                z = cmath.exp(-2j * math.pi * fq / 1000.0)
                H = 1 + 0j
                for s in sos_o:
                    b0, b1, b2, a0, a1, a2 = s
                    H *= (b0 + b1 * z + b2 * z * z) / (a0 + a1 * z + a2 * z * z)
                Ho.append(H)
            assert max(abs(abs(Ho[k]) - abs(Hr[k])) for k in range(len(wr))) < 1e-9


def test_sosfilt_matches_scipy():
    np = pytest.importorskip("numpy")
    sig = pytest.importorskip("scipy.signal")
    sos = sig.butter(4, 0.2, output="sos")
    x = white_noise(400, 1.0, seed=8)
    yo = sosfilt([list(r) for r in sos], x)
    yr = sig.sosfilt(sos, np.array(x))
    assert max(abs(yo[i] - yr[i]) for i in range(len(x))) < 1e-9


def test_sosfilt_equals_lfilter_for_the_same_transfer_function():
    # applying a filter as a biquad cascade (sosfilt) or as its expanded (b, a)
    # transfer function (lfilter) must give the same output.
    np = pytest.importorskip("numpy")
    sig = pytest.importorskip("scipy.signal")
    sos = sig.butter(5, 0.3, output="sos")
    b, a = sig.sos2tf(sos)
    x = white_noise(300, 1.0, seed=5)
    y_sos = sosfilt([list(r) for r in sos], x)
    y_lf = lfilter(list(b), list(a), x)
    assert max(abs(y_sos[i] - y_lf[i]) for i in range(300)) < 1e-8


def test_hilbert_matches_scipy_even_and_odd():
    np = pytest.importorskip("numpy")
    sig = pytest.importorskip("scipy.signal")
    for n in (256, 255):
        x = white_noise(n, 1.0, seed=9)
        zo = hilbert(x)
        zr = sig.hilbert(np.array(x))
        assert max(abs(zo[i] - complex(zr[i])) for i in range(n)) < 1e-9


def test_resample_matches_scipy():
    np = pytest.importorskip("numpy")
    sig = pytest.importorskip("scipy.signal")
    x = [math.sin(2 * math.pi * 3 * i / 64) + 0.5 * math.cos(2 * math.pi * 10 * i / 64)
         for i in range(64)]
    for num in (128, 40):
        yo = resample(x, num)
        yr = list(sig.resample(np.array(x), num))
        assert max(abs(yo[i] - yr[i]) for i in range(num)) < 1e-9


def test_fuzz_differential_vs_scipy():
    # a compact, reproducible randomized-config sweep pinning welch / firwin / butter /
    # dpss / filtfilt to SciPy across the parameter space.  Config-specific bugs (a
    # particular window x scaling x parity x overlap) hide from hand-picked cases; a
    # 1000-config version of this found none, and this locks the coverage in.
    np = pytest.importorskip("numpy")
    sig = pytest.importorskip("scipy.signal")
    rng = random.Random(20240517)

    for _ in range(10):                                  # welch
        n = rng.choice([1024, 2048])
        x = white_noise(n, 1.0, seed=rng.randint(0, 99999))
        nps = rng.choice([64, 127, 128, 200, 256])
        nov = rng.choice([0, nps // 2, 3 * nps // 4])
        win = rng.choice(["hann", "hamming", "blackman", "boxcar", "nuttall"])
        scaling = rng.choice(["density", "spectrum"])
        det = rng.choice(["constant", "linear"])
        fo, Po = welch(x, 2.0, win, nps, nov, detrend_type=det, scaling=scaling)
        fr, Pr = sig.welch(np.asarray(x), 2.0, win, nperseg=nps, noverlap=nov,
                           detrend=det, scaling=scaling)
        assert max(abs(Po[k] - Pr[k]) for k in range(len(Po))) < 1e-9

    for _ in range(12):                                  # firwin, all four band types
        ntaps = rng.choice([31, 32, 51, 64, 65, 101, 128, 129])
        win = rng.choice(["hann", "hamming", "blackman", "nuttall"])
        band = rng.choice(["low", "high", "bandpass", "bandstop"])
        try:
            if band in ("low", "high"):
                fc = rng.uniform(50, 450)
                pz = band == "low"
                h = firwin(ntaps, fc, 1000.0, win, pass_zero=pz)
                r = sig.firwin(ntaps, fc, fs=1000.0, window=win,
                               pass_zero=("lowpass" if pz else "highpass"))
            else:
                f1, f2 = rng.uniform(50, 200), rng.uniform(250, 450)
                pz = band == "bandstop"
                h = firwin(ntaps, [f1, f2], 1000.0, win, pass_zero=pz)
                r = sig.firwin(ntaps, [f1, f2], fs=1000.0, window=win,
                               pass_zero=("bandstop" if pz else "bandpass"))
            assert max(abs(h[i] - r[i]) for i in range(ntaps)) < 1e-9
        except SpectralError:
            pass                                         # even-numtaps HP/BS: scipy rejects too

    for _ in range(10):                                  # butter, orders 1..10
        order = rng.randint(1, 10)
        fc = rng.uniform(50, 450)
        bt = rng.choice(["low", "high"])
        h = sosfilt(butter(order, fc, 1000.0, bt), impulse(1024))
        wr, Hr = sig.sosfreqz(sig.butter(order, fc, btype=bt, fs=1000.0, output="sos"),
                              worN=128, fs=1000.0)
        Ho = [sum(h[i] * cmath.exp(-2j * math.pi * fq * i / 1000.0) for i in range(len(h)))
              for fq in wr]
        assert max(abs(abs(Ho[k]) - abs(Hr[k])) for k in range(len(wr))) < 1e-7

    for _ in range(8):                                   # dpss
        n = rng.choice([64, 128, 200, 256])
        nw = rng.uniform(2.0, 5.0)
        k = rng.randint(1, int(2 * nw))
        ours, ref = dpss(n, nw, k), sig.windows.dpss(n, nw, Kmax=k)
        for j in range(k):
            a, b = ours[j], list(ref[j])
            dp = max(abs(a[i] - b[i]) for i in range(n))
            dm = max(abs(a[i] + b[i]) for i in range(n))
            assert min(dp, dm) < 1e-9

    for _ in range(8):                                   # filtfilt, incl edges
        order = rng.randint(2, 8)
        wn = rng.uniform(0.05, 0.45)
        n = rng.choice([300, 600])
        x = white_noise(n, 1.0, seed=rng.randint(0, 99999))
        b, a = sig.butter(order, wn)
        ours = filtfilt(list(b), list(a), x)
        ref = sig.filtfilt(b, a, np.asarray(x))
        assert max(abs(ours[i] - ref[i]) for i in range(n)) < 1e-6


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
