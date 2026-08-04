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
[INIT] No overdrive (CPU @ 600 MHz, NPU @ 800 MHz)...
[INIT] Configuring UART...
[INIT] Configuring external memories...
[OK] External memories mapped
[INIT] Configuring NPU...

=== BirdNET-STM32 SD Card Inference ===
[INFO] Frontend: raw (waveform → NPU)
[INFO] Sample rate: 24000 Hz, chunk: 2.500s (60000 samples), classes: 25
[INIT] Initialising NPU network...
[OK] NPU input:  "..."  240000 bytes
[OK] NPU output: "..."  100 bytes
[INIT] Mounting SD card (SDMMC2)...
[OK] SD card mounted
[OK] 25 class labels compiled in
[INIT] Scanning /audio/ for .wav files...
[OK] Found 8 audio files

[1/8] recording_001.wav
  [WAV] 24000 Hz, 16-bit, 1 ch, 72000 samples
  [BENCH] read=72ms stft=0ms npu=13ms total=85ms
  recording_001.wav:
    [1] Common Chiffchaff: 72.3%
    [2] Eurasian Blue Tit: 15.1%

[2/8] recording_002.wav
  [WAV] 24000 Hz, 16-bit, 1 ch, 72000 samples
  [BENCH] read=72ms stft=0ms npu=13ms total=85ms
  recording_002.wav:
    [1] Great Tit: 89.1%

...

=== DONE ===
Processed: 8 / 8 files (0 errors)
Benchmark: read=576ms stft=0ms npu=104ms total=680ms (avg read=72ms stft=0ms npu=13ms total=85ms)
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
  [BENCH] read=72ms stft=0ms npu=13ms total=85ms
```
Per-file timing in milliseconds (1 ms resolution from `HAL_GetTick()`):

- `read` — SD card I/O (FatFs `f_read` + PCM16→float32 conversion)
- `stft` — frontend CPU time (STFT for hybrid/librosa; zero for raw)
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
Benchmark: read=576ms stft=0ms npu=104ms total=680ms (avg read=72ms stft=0ms npu=13ms total=85ms)
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
    "results": [
        {
            "file": "recording_001.wav",
            "detections": [
                {"label": "Common Chiffchaff", "score": 0.723},
                {"label": "Eurasian Blue Tit", "score": 0.151},
            ],
            "bench": {"read_ms": 72, "stft_ms": 0, "npu_ms": 13, "total_ms": 85},
        },
        # ...
    ],
    "processed": 8,
    "total": 8,
    "errors": 0,
    "benchmark": {
        "avg_read_ms": 72, "avg_stft_ms": 0, "avg_npu_ms": 13, "avg_total_ms": 85,
    },
}
```

## Real-Time Factor

The host displays a **real-time factor** (RTF) after the run:

```
Real-time factor: 0.0340x (29x faster than real-time)
```

RTF = `avg_total_ms / (chunk_duration × 1000)`. Values below 1 mean faster than
real time; the host prints the reciprocal as the speedup.

## Timeout Behavior

The host waits for the `=== DONE ===` marker or a configurable timeout
(default 300 s). If the timeout expires:

- Partial results are still parsed and displayed.
- A warning is printed indicating incomplete processing.
- The command returns the partial parsed result after printing a warning.

Common causes of timeout: firmware crash (bus fault, assertion), SD card not
inserted, wrong serial port, baud rate mismatch.
