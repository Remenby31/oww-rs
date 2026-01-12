# oww-rs

A high-performance Rust port of [OpenWakeWord](https://github.com/dscripka/openWakeWord) for lightweight wake word detection.

## Features

- **27x faster** than Python implementation (without VAD)
- **Single binary** - no Python runtime required
- **Same ONNX models** - fully compatible with OpenWakeWord models
- **Silero VAD integration** - reduces false positives
- **Low latency** - ~55 microseconds per inference

## Performance Comparison

Benchmarked on AMD Ryzen (single thread, 80ms audio chunks):

| Implementation | Without VAD | With VAD |
|----------------|-------------|----------|
| Python (OpenWakeWord) | 1.501 ms | 1.741 ms |
| **Rust (oww-rs)** | **0.055 ms** | **0.211 ms** |
| Speedup | **27x** | **8x** |

## Binary Size

- **oww-rs**: ~20 MB (includes ONNX Runtime)
- **Models**: ~5.5 MB total
  - melspectrogram.onnx: 1.1 MB
  - embedding_model.onnx: 1.3 MB
  - hey_jarvis.onnx: 1.3 MB
  - silero_vad.onnx: 1.8 MB

Compare to Python OpenWakeWord: ~200 MB (Python + numpy + onnxruntime + dependencies)

## Installation

### From source

```bash
git clone https://github.com/yourrepo/oww-rs
cd oww-rs
cargo build --release
```

### Download models

The models are compatible with OpenWakeWord. Download them from the official repository or use:

```bash
# Download models to models/ directory
mkdir -p models
# melspectrogram and embedding are shared across all wake words
curl -L -o models/melspectrogram.onnx "https://github.com/dscripka/openWakeWord/raw/main/openwakeword/resources/models/melspectrogram.onnx"
curl -L -o models/embedding_model.onnx "https://github.com/dscripka/openWakeWord/raw/main/openwakeword/resources/models/embedding_model.onnx"
# Wake word specific model (example: hey_jarvis)
curl -L -o models/hey_jarvis.onnx "https://huggingface.co/davidscripka/openwakeword/resolve/main/hey_jarvis_v0.1.onnx"
# Optional: Silero VAD for reduced false positives
curl -L -o models/silero_vad.onnx "https://github.com/dscripka/openWakeWord/raw/main/openwakeword/resources/models/silero_vad.onnx"
```

## Usage

### Basic usage

```bash
# Run with default settings (threshold=0.5, VAD enabled)
./target/release/oww-rs --model-dir models

# Show detection scores continuously
./target/release/oww-rs --model-dir models --verbose

# Execute command on detection
./target/release/oww-rs --model-dir models --command "notify-send 'Wake word detected!'"

# Adjust threshold (lower = more sensitive, higher = fewer false positives)
./target/release/oww-rs --model-dir models --threshold 0.3
```

### CLI Options

```
USAGE:
    oww-rs [OPTIONS]

OPTIONS:
    -m, --model-dir <MODEL_DIR>    Path to models directory [default: models]
    -t, --threshold <THRESHOLD>    Detection threshold (0.0 - 1.0) [default: 0.5]
    -c, --command <COMMAND>        Command to execute on detection
        --vad                      Enable VAD (Voice Activity Detection)
        --cooldown <COOLDOWN>      Cooldown after detection (seconds) [default: 2.0]
        --verbose                  Show scores continuously
    -h, --help                     Print help
```

### As a library

```rust
use oww_rs::WakeWordDetector;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    // Create detector with VAD enabled
    let mut detector = WakeWordDetector::new(
        Path::new("models"),
        0.5,  // threshold
        true  // use_vad
    )?;

    // Process audio chunks (16kHz, mono, i16)
    let audio: Vec<i16> = get_audio_chunk(); // 1280 samples = 80ms
    let score = detector.predict(&audio)?;

    if detector.detected() {
        println!("Wake word detected! Score: {}", score);
    }

    Ok(())
}
```

## Architecture

```
Audio 16kHz int16
       |
       v
+------------------+
|  Melspectrogram  |  <- melspectrogram.onnx
| [1,1280] -> [N,32]|
+--------+---------+
         |
         v  normalize: spec/10 + 2
+------------------+
|    Embedding     |  <- embedding_model.onnx
| [B,76,32,1]->[B,96]|
+--------+---------+
         |
         v  sliding window (step=8)
+------------------+
|  Classification  |  <- hey_jarvis.onnx
| [1,16,96] -> [1,1]|
+--------+---------+
         |
         v
    Score 0-1
```

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sample rate | 16000 Hz | Audio sample rate |
| Chunk size | 1280 samples | 80ms per chunk |
| Mel bins | 32 | Mel spectrogram bins |
| Embedding window | 76 frames | Input frames for embedding |
| Embedding step | 8 frames | Sliding window step |
| Embedding dim | 96 | Embedding vector size |
| Classification frames | 16 | Input embeddings for classifier |

## Running Benchmarks

```bash
# Run Rust benchmarks
cargo bench

# Run Python comparison (requires OpenWakeWord installed)
python benchmark_python.py
```

## License

MIT

## Credits

- [OpenWakeWord](https://github.com/dscripka/openWakeWord) - Original Python implementation
- [ort](https://github.com/pykeio/ort) - Rust ONNX Runtime bindings
- [Silero VAD](https://github.com/snakers4/silero-vad) - Voice Activity Detection model
