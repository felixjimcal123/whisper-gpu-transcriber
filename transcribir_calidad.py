"""Transcribe MP4s in English with independent windows and repetition guards.

Requirements: py -m pip install faster-whisper tqdm
Run from the MP4 directory: py transcribir_calidad.py
"""
import json
import logging
import math
import os
import re
import shutil
from difflib import SequenceMatcher
import site
import warnings
from collections import Counter
from pathlib import Path

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Register pip-installed NVIDIA libraries before importing CTranslate2.
_DLL_HANDLES = []
for base in site.getsitepackages():
    for package in ("cublas", "cudnn", "cuda_nvrtc"):
        folder = Path(base) / "nvidia" / package / "bin"
        if folder.is_dir():
            _DLL_HANDLES.append(os.add_dll_directory(str(folder)))

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from tqdm import tqdm

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

FOLDER = Path(__file__).resolve().parent
MODEL = "large-v3"
RATE = 16000
WINDOW_SECONDS = 24
CONTEXT_SECONDS = 2
VERSION = 1


def loop_detected(text):
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
    n = len(tokens)
    if n < 8:
        return False
    if any(len(list_group) >= 9 for list_group in _word_runs(tokens)):
        return True
    for size in (3, 5, 8):
        if n < size * 4:
            continue
        counts = Counter(tuple(tokens[i:i + size]) for i in range(n - size + 1))
        most = max(counts.values(), default=0)
        if most >= 4 and most * size >= n * 0.35:
            return True
    return False


def _word_runs(words):
    previous, run = None, []
    for word in words:
        if word != previous:
            if run:
                yield run
            run = []
        run.append(word)
        previous = word
    if run:
        yield run


def suspicious_neighbor(text, previous):
    a = " ".join(re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower()))
    b = " ".join(re.findall(r"[a-z]+(?:'[a-z]+)?", previous.lower()))
    return (len(a.split()) >= 12 and len(b.split()) >= 12
            and SequenceMatcher(None, a, b, autojunk=False).ratio() > 0.87)


def timestamp(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"


def recognize(model, audio, read_start, core_start, core_end, language, retry=False):
    arguments = dict(
        language=language,
        task="transcribe",
        beam_size=5,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters={
            "threshold": 0.35,
            "min_speech_duration_ms": 0,
            "min_silence_duration_ms": 400,
            "speech_pad_ms": 250,
        },
        word_timestamps=True,
        hallucination_silence_threshold=1.5,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        max_new_tokens=320,
        temperature=(0.0, 0.2, 0.4) if not retry else (0.2, 0.4, 0.6),
        repetition_penalty=1.0 if not retry else 1.15,
        no_repeat_ngram_size=0 if not retry else 5,
    )
    segments, _ = model.transcribe(audio, **arguments)
    words, quality_warning = [], False
    for seg in segments:
        if seg.compression_ratio > 2.8 and len(seg.text) > 80:
            quality_warning = True
        if seg.words:
            for word in seg.words:
                middle = read_start + (word.start + word.end) / 2
                if core_start <= middle < core_end:
                    words.append(word.word)
        else:
            middle = read_start + (seg.start + seg.end) / 2
            if core_start <= middle < core_end:
                words.append(seg.text)
    text = "".join(words).strip()
    return text, quality_warning or loop_detected(text)


def transcribe_video(model, video, language):
    suffix = "spanish" if language == "es" else "english"
    output = video.with_name(video.stem + f".{suffix}.txt")
    cache = video.parent / ("." + video.stem + ".whisper-checkpoints")
    signature = {"version": VERSION, "size": video.stat().st_size,
                 "mtime_ns": video.stat().st_mtime_ns,
                 "model": MODEL, "window": WINDOW_SECONDS, "language": language}
    manifest = cache / "manifest.json"
    if cache.exists():
        try:
            matches = json.loads(manifest.read_text(encoding="utf-8")) == signature
        except (OSError, ValueError):
            matches = False
        if not matches:
            shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(signature), encoding="utf-8")

    audio = decode_audio(str(video), sampling_rate=RATE)
    total = len(audio)
    core_samples = WINDOW_SECONDS * RATE
    context_samples = CONTEXT_SECONDS * RATE
    count = math.ceil(total / core_samples)

    previous = ""
    with tqdm(total=count, desc="Transcribiendo", unit="fragmento",
              dynamic_ncols=True) as bar:
        for i in range(count):
            checkpoint = cache / f"{i:06d}.txt"
            if checkpoint.exists():
                previous = checkpoint.read_text(encoding="utf-8")
                bar.update(1)
                continue
            start = i * core_samples
            end = min(total, start + core_samples)
            read_start = max(0, start - context_samples)
            read_end = min(total, end + context_samples)
            clip = np.ascontiguousarray(audio[read_start:read_end])
            text, suspicious = recognize(model, clip, read_start / RATE,
                                         start / RATE, end / RATE, language)
            suspicious = suspicious or suspicious_neighbor(text, previous)
            if suspicious:
                second, second_suspicious = recognize(
                    model, clip, read_start / RATE, start / RATE, end / RATE,
                    language, retry=True)
                if not second_suspicious and not suspicious_neighbor(second, previous):
                    text, suspicious = second, False
            if suspicious:
                text = (f"[REVISAR AUDIO {timestamp(start / RATE)}"
                        f" - {timestamp(end / RATE)}: posible fallo de transcripción]")
            pending = checkpoint.with_suffix(".tmp")
            pending.write_text(text + "\n", encoding="utf-8")
            os.replace(pending, checkpoint)
            previous = text
            bar.update(1)

    pending_output = output.with_suffix(".tmp")
    with pending_output.open("w", encoding="utf-8") as file:
        for i in range(count):
            file.write((cache / f"{i:06d}.txt").read_text(encoding="utf-8"))
    os.replace(pending_output, output)
    shutil.rmtree(cache)


def ask_language():
    while True:
        language = input("Idioma de transcripción (es/en): ").strip().lower()
        if language in ("es", "en"):
            return language
        print("Idioma no válido. Introduce 'es' o 'en'.")


def main():
    language = ask_language()
    videos = sorted(FOLDER.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No hay MP4 en {FOLDER}")
    model = WhisperModel(MODEL, device="cuda", device_index=0,
                         compute_type="int8_float32", cpu_threads=4)
    for video in videos:
        transcribe_video(model, video, language)


if __name__ == "__main__":
    main()
