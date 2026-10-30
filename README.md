# Whisper GPU Transcriber

A Python-based, GPU-accelerated video transcription tool powered by **Faster-Whisper** and OpenAI's Whisper large-v3 model.

Designed for reliable transcription of long MP4 recordings, with a focus on transcription quality, fault recovery, and reducing repetitive hallucinations.

### Key Features

- **GPU acceleration:** NVIDIA CUDA inference using Faster-Whisper.
- **Multilingual transcription:** Supports English and Spanish.
- **Chunk-based processing:** Independent 24-second audio windows with overlapping context.
- **Hallucination detection:** Identifies suspicious repetitions and automatically retries problematic segments.
- **Checkpoint recovery:** Saves intermediate results to resume interrupted transcriptions.
- **Voice Activity Detection (VAD):** Filters silence and non-speech regions.
- **Word-level timestamps:** Improves transcription boundaries between audio chunks.
- **Batch processing:** Automatically processes MP4 files in the script directory.
- **Automatic text export:** Generates UTF-8 transcription files for each recording.

### Tech Stack

Python · Faster-Whisper · CTranslate2 · NVIDIA CUDA · NumPy · tqdm

### Use Cases

Technical interviews, recorded meetings, lectures, research sessions, and long-form video transcription.
