# UART Protocol

The firmware outputs structured text over USART1 at **921,600 baud** (8N1).
The host Python script (`board_test.py`) captures and parses this output using
regex patterns.

## Output Structure

A complete run produces output in this order:

```
[INIT] Setting vector table...
[INIT] HAL and clocks configured
[INIT] Enabling caches...
[INIT] Switching to overdrive...
[INIT] Configuring UART...
[INIT] Configuring external memories...
[OK] External memories mapped
[INIT] Configuring NPU...

=== BirdNET-STM32 SD Card Inference ===
[INFO] Sample rate: 24000 Hz, chunk: 3s, FFT: 512, hop: 281, spec: 257x256, classes: 10
[INIT] Initialising NPU network...
[OK] NPU input:  "..."  263168 bytes
[OK] NPU output: "..."  40 bytes
[INIT] Mounting SD card (SDMMC2)...
[OK] SD card mounted
[OK] 10 class labels compiled in
[INIT] Scanning /audio/ for .wav files...
[OK] Found 8 audio files

[1/8] recording_001.wav
  [WAV] 24000 Hz, 16-bit, 1 ch, 72000 samples
  [BENCH] read=12ms stft=28ms npu=4ms total=44ms
  recording_001.wav:
    [1] Common Chiffchaff: 72.3%
    [2] Eurasian Blue Tit: 15.1%

[2/8] recording_002.wav
  [WAV] 24000 Hz, 16-bit, 1 ch, 72000 samples
  [BENCH] read=11ms stft=27ms npu=3ms total=41ms
  recording_002.wav:
    [1] Great Tit: 89.1%

...

=== DONE ===
Processed: 8 / 8 files (0 errors)
Benchmark: read=96ms stft=224ms npu=32ms total=352ms (avg read=12ms stft=28ms npu=4ms total=44ms)
[OK] SD card unmounted. Halting.
```

## Line-by-Line Reference

### Init Lines

| Prefix | Meaning |
|---|---|
| `[INIT]` | Board initialization step in progress |
| `[OK]` | Step completed successfully |
| `[ERROR]` | Fatal error — firmware halts after printing |
| `[WARN]` | Non-fatal warning (e.g., no audio files found) |
| `[INFO]` | Informational (model parameters) |

### Per-File Lines

**File header:**
```
[1/8] recording_001.wav
```
Format: `[index/total] filename` — 1-indexed.

**WAV info:**
```
  [WAV] 24000 Hz, 16-bit, 1 ch, 72000 samples
```
Parsed sample rate, bit depth, channels, and total samples.

**Benchmark:**
```
  [BENCH] read=12ms stft=28ms npu=4ms total=44ms
```
Per-file timing in milliseconds (1 ms resolution from `HAL_GetTick()`):

- `read` — SD card I/O (FatFs `f_read` + PCM16→float32 conversion)
- `stft` — STFT computation on Cortex-M55
- `npu` — NPU inference (including cache flush/invalidate and memcpy)
- `total` — sum of the above

**Skip/error lines:**
```
  [SKIP] Sample rate 22050 != 24000
  [SKIP] Cannot open file
  [SKIP] Invalid WAV format
  [ERROR] Inference failed
```

**Detection results:**
```
  recording_001.wav:
    [1] Common Chiffchaff: 72.3%
    [2] Eurasian Blue Tit: 15.1%
```
Top-K predictions sorted by descending score. Only scores ≥
`APP_SCORE_THRESHOLD` are printed. The score is formatted as `integer.tenths%`
(e.g., `72.3%` = 0.723).

### Summary Lines

**Done marker** — signals end of processing:
```
=== DONE ===
```

**File count:**
```
Processed: 8 / 8 files (0 errors)
```
`processed / total (errors)` — processed + errors = total.

**Aggregate benchmark:**
```
Benchmark: read=96ms stft=224ms npu=32ms total=352ms (avg read=12ms stft=28ms npu=4ms total=44ms)
```
Cumulative timing and per-file averages. Only printed if `processed > 0`.

## Host-Side Parsing

The Python `board_test.py` uses these regex patterns:

```python
# File header
r"^\[(\d+)/(\d+)\]\s+(.+)$"

# WAV info
r"^\s+\[WAV\]\s+(\d+)\s+Hz,\s+(\d+)-bit,\s+(\d+)\s+ch,\s+(\d+)\s+samples"

# Per-file benchmark
r"^\s+\[BENCH\]\s+read=(\d+)ms\s+stft=(\d+)ms\s+npu=(\d+)ms\s+total=(\d+)ms"

# Detection result
r"^\s+\[(\d+)\]\s+(.+?):\s+([\d.]+)%"

# Skip / error
r"^\s+\[SKIP\]\s+(.+)$"
r"^\s+\[ERROR\]\s+(.+)$"

# Done marker
r"^=== DONE ===$"

# Summary
r"^Processed:\s+(\d+)\s*/\s*(\d+)\s+files\s+\((\d+)\s+errors\)"

# Aggregate benchmark
r"^Benchmark:.*?read=(\d+)ms\s+stft=(\d+)ms\s+npu=(\d+)ms\s+total=(\d+)ms\s+"
r"\(avg read=(\d+)ms\s+stft=(\d+)ms\s+npu=(\d+)ms\s+total=(\d+)ms\)"
```

### Parsed Output Structure

The parser produces a dictionary:

```python
{
    "files": [
        {
            "index": 1,
            "filename": "recording_001.wav",
            "detections": [
                {"rank": 1, "label": "Common Chiffchaff", "score": 72.3},
                {"rank": 2, "label": "Eurasian Blue Tit", "score": 15.1},
            ],
            "bench": {"read_ms": 12, "stft_ms": 28, "npu_ms": 4, "total_ms": 44},
            "error": None,
        },
        # ...
    ],
    "processed": 8,
    "total": 8,
    "errors": 0,
    "benchmark": {
        "read_ms": 96, "stft_ms": 224, "npu_ms": 32, "total_ms": 352,
        "avg_read_ms": 12, "avg_stft_ms": 28, "avg_npu_ms": 4, "avg_total_ms": 44,
    },
}
```

## Real-Time Factor

The host displays a **real-time factor** (RTF) after the run:

```
Real-time factor: 68.2x (3.0s audio processed in 44ms avg)
```

RTF = `chunk_duration / avg_total_ms × 1000`. Values > 1 mean faster than
real-time. Typical values are 50–75× for a 3 s chunk.

## Timeout Behavior

The host waits for the `=== DONE ===` marker or a configurable timeout
(default 300 s). If the timeout expires:

- Partial results are still parsed and displayed.
- A warning is printed indicating incomplete processing.
- Exit code is non-zero.

Common causes of timeout: firmware crash (bus fault, assertion), SD card not
inserted, wrong serial port, baud rate mismatch.
