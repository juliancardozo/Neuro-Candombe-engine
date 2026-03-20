from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.io.wavfile import write
from scipy.signal import butter, sosfilt

SAMPLE_RATE = 44100


def time_axis(duration: float) -> np.ndarray:
    return np.linspace(0.0, duration, int(SAMPLE_RATE * duration), endpoint=False)


def drum_envelope(
    duration: float,
    attack: float = 0.001,
    hold: float = 0.0,
    decay: float = 10.0,
) -> np.ndarray:
    t = time_axis(duration)
    env = np.exp(-decay * np.maximum(t - attack - hold, 0.0))
    if attack > 0:
        attack_curve = np.clip(t / attack, 0.0, 1.0)
        env *= attack_curve
    if hold > 0:
        env[t < attack + hold] = np.maximum(env[t < attack + hold], 1.0)
    return env


def normalize(signal: np.ndarray, peak: float = 0.96) -> np.ndarray:
    max_val = np.max(np.abs(signal)) or 1.0
    return np.tanh((signal / max_val) * 1.45) * peak


def mix_layers(*layers: np.ndarray) -> np.ndarray:
    max_len = max(layer.size for layer in layers)
    mix = np.zeros(max_len, dtype=np.float64)
    for layer in layers:
        mix[: layer.size] += layer
    return mix


def filtered_noise(
    duration: float,
    low_hz: float,
    high_hz: float,
    decay: float,
    attack: float = 0.0008,
    gain: float = 1.0,
) -> np.ndarray:
    noise = np.random.randn(int(SAMPLE_RATE * duration))
    sos = butter(4, [low_hz, high_hz], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    band = sosfilt(sos, noise)
    return band * drum_envelope(duration, attack=attack, decay=decay) * gain


def sweep_tone(
    start_hz: float,
    end_hz: float,
    duration: float,
    decay: float,
    harmonics: Tuple[Tuple[float, float], ...],
    attack: float = 0.001,
    gain: float = 1.0,
) -> np.ndarray:
    t = time_axis(duration)
    base_curve = np.geomspace(start_hz, end_hz, t.size)
    env = drum_envelope(duration, attack=attack, decay=decay)
    signal = np.zeros_like(t)

    for ratio, amp in harmonics:
        phase = 2 * np.pi * np.cumsum(base_curve * ratio) / SAMPLE_RATE
        signal += np.sin(phase) * amp

    return signal * env * gain


def wood_click(
    duration: float,
    freq_a: float,
    freq_b: float,
    decay: float,
    gain: float = 1.0,
) -> np.ndarray:
    tone = sweep_tone(
        start_hz=freq_a,
        end_hz=freq_b,
        duration=duration,
        decay=decay,
        harmonics=((1.0, 0.92), (1.9, 0.42), (2.8, 0.18)),
        attack=0.0004,
        gain=gain,
    )
    noise = filtered_noise(duration, 1600, 6200, decay=decay * 1.25, attack=0.0002, gain=0.16 * gain)
    return tone + noise


def chico_mano_sample() -> np.ndarray:
    duration = 0.062
    body = sweep_tone(
        start_hz=560,
        end_hz=310,
        duration=duration,
        decay=28,
        harmonics=((1.0, 0.62), (1.28, 0.08), (1.86, 0.03)),
        attack=0.0005,
        gain=0.54,
    )
    shell = sweep_tone(
        start_hz=390,
        end_hz=220,
        duration=0.036,
        decay=34,
        harmonics=((1.0, 0.16), (1.54, 0.03)),
        attack=0.0003,
        gain=0.12,
    )
    hand = filtered_noise(duration, 650, 2100, decay=34, attack=0.0002, gain=0.034)
    wood_a = wood_click(0.014, 1460, 720, decay=90, gain=0.12)
    wood_b = np.pad(wood_click(0.01, 1120, 540, decay=110, gain=0.08), (52, 0))
    return normalize(mix_layers(body, shell, hand, wood_a, wood_b))


def chico_palo_sample() -> np.ndarray:
    duration = 0.048
    body = sweep_tone(
        start_hz=860,
        end_hz=520,
        duration=duration,
        decay=26,
        harmonics=((1.0, 0.46), (1.62, 0.12), (2.2, 0.04)),
        attack=0.0005,
        gain=0.48,
    )
    stick = wood_click(0.018, 2300, 1080, decay=78, gain=0.54)
    shell = sweep_tone(
        start_hz=540,
        end_hz=320,
        duration=0.028,
        decay=34,
        harmonics=((1.0, 0.12),),
        attack=0.0004,
        gain=0.1,
    )
    bark = filtered_noise(duration, 1600, 5200, decay=34, attack=0.0002, gain=0.024)
    return normalize(mix_layers(body, stick, shell, bark))


def chico_sample() -> np.ndarray:
    return normalize(mix_layers(chico_mano_sample(), chico_palo_sample() * 0.72))


def repique_variant(
    body_start: float,
    body_end: float,
    strike_a: float,
    strike_b: float,
    duration: float,
    body_gain: float,
    strike_gain: float,
    ring_gain: float,
    decay: float,
) -> np.ndarray:
    body = sweep_tone(
        start_hz=body_start,
        end_hz=body_end,
        duration=duration,
        decay=decay,
        harmonics=((1.0, 0.84), (1.48, 0.2), (2.0, 0.1)),
        attack=0.0007,
        gain=body_gain,
    )
    ring = sweep_tone(
        start_hz=body_start * 1.55,
        end_hz=body_end * 1.35,
        duration=duration * 0.74,
        decay=decay * 1.45,
        harmonics=((1.0, 0.3), (1.9, 0.12)),
        attack=0.0005,
        gain=ring_gain,
    )
    strike = wood_click(duration * 0.3, strike_a, strike_b, decay=decay * 4.2, gain=strike_gain)
    skin = filtered_noise(duration, 850, 3200, decay=decay * 2.1, attack=0.0004, gain=0.08)
    return normalize(mix_layers(body, ring, strike, skin))


def repique_sample() -> np.ndarray:
    return repique_variant(400, 220, 1460, 820, 0.14, 0.86, 0.2, 0.24, 11.0)


def repique_talk_open() -> np.ndarray:
    return repique_variant(450, 255, 1220, 680, 0.135, 1.0, 0.1, 0.28, 8.4)


def repique_talk_support() -> np.ndarray:
    return repique_variant(330, 205, 2140, 1120, 0.088, 0.3, 0.42, 0.05, 16.0)


def repique_talk_answer() -> np.ndarray:
    return repique_variant(395, 230, 1620, 880, 0.102, 0.58, 0.28, 0.12, 12.0)


def repique_talk_close() -> np.ndarray:
    return repique_variant(470, 275, 1540, 840, 0.12, 0.8, 0.2, 0.18, 9.8)


def repique_call_open() -> np.ndarray:
    return repique_variant(490, 285, 1380, 780, 0.17, 1.04, 0.16, 0.34, 8.6)


def repique_call_support() -> np.ndarray:
    return repique_variant(410, 245, 1680, 920, 0.12, 0.66, 0.24, 0.16, 11.0)


def repique_call_close() -> np.ndarray:
    return repique_variant(380, 220, 2060, 1080, 0.1, 0.34, 0.38, 0.06, 15.0)


def repique_call_tail() -> np.ndarray:
    return repique_variant(320, 185, 1840, 980, 0.085, 0.2, 0.28, 0.02, 17.0)


def piano_sample() -> np.ndarray:
    duration = 0.28
    sub = sweep_tone(
        start_hz=150,
        end_hz=68,
        duration=duration,
        decay=7.5,
        harmonics=((1.0, 0.94), (2.0, 0.14)),
        attack=0.0012,
        gain=1.0,
    )
    shell = sweep_tone(
        start_hz=230,
        end_hz=110,
        duration=0.22,
        decay=9.5,
        harmonics=((1.0, 0.36), (1.7, 0.12)),
        attack=0.0008,
        gain=0.58,
    )
    knock = wood_click(0.065, 760, 360, decay=30, gain=0.24)
    skin = filtered_noise(duration, 180, 1500, decay=18, attack=0.0007, gain=0.09)
    return normalize(mix_layers(sub, shell, knock, skin))


def clave_palmas_sample() -> np.ndarray:
    base = 0.09
    clap_a = filtered_noise(base, 800, 3600, decay=34, attack=0.0003, gain=0.24)
    clap_b = np.pad(filtered_noise(base * 0.72, 1100, 4200, decay=46, attack=0.0002, gain=0.18), (90, 0))
    clap_c = np.pad(filtered_noise(base * 0.58, 1400, 5200, decay=52, attack=0.0002, gain=0.11), (170, 0))
    body = sweep_tone(
        start_hz=640,
        end_hz=300,
        duration=0.045,
        decay=32,
        harmonics=((1.0, 0.22), (1.8, 0.06)),
        attack=0.0004,
        gain=0.18,
    )
    return normalize(mix_layers(clap_a, clap_b, clap_c, body))


def save_wav(filename: Path, data: np.ndarray) -> None:
    audio = np.clip(data, -1.0, 1.0)
    audio = (audio * 32767).astype(np.int16)
    write(str(filename), SAMPLE_RATE, audio)


def main() -> None:
    samples_dir = Path("web/samples")
    samples_dir.mkdir(parents=True, exist_ok=True)

    save_wav(samples_dir / "chico.wav", chico_sample())
    save_wav(samples_dir / "chico_mano.wav", chico_mano_sample())
    save_wav(samples_dir / "chico_palo.wav", chico_palo_sample())

    save_wav(samples_dir / "repique.wav", repique_sample())
    save_wav(samples_dir / "repique_talk_open.wav", repique_talk_open())
    save_wav(samples_dir / "repique_talk_support.wav", repique_talk_support())
    save_wav(samples_dir / "repique_talk_answer.wav", repique_talk_answer())
    save_wav(samples_dir / "repique_talk_close.wav", repique_talk_close())
    save_wav(samples_dir / "repique_call_open.wav", repique_call_open())
    save_wav(samples_dir / "repique_call_support.wav", repique_call_support())
    save_wav(samples_dir / "repique_call_close.wav", repique_call_close())
    save_wav(samples_dir / "repique_call_tail.wav", repique_call_tail())

    save_wav(samples_dir / "piano.wav", piano_sample())
    save_wav(samples_dir / "clave.wav", clave_palmas_sample())


if __name__ == "__main__":
    main()
