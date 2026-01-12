pub mod audio;
pub mod detector;
pub mod embedding;
pub mod preprocessing;
pub mod vad;

// Critical constants - must match OpenWakeWord exactly
pub const SAMPLE_RATE: u32 = 16000;
pub const CHUNK_SIZE: usize = 1280; // 80ms at 16kHz
pub const MEL_BINS: usize = 32;
pub const EMBEDDING_WINDOW: usize = 76; // frames
pub const EMBEDDING_STEP: usize = 8; // frames
pub const EMBEDDING_DIM: usize = 96;
pub const CLASSIFICATION_FRAMES: usize = 16;

// VAD constants
pub const VAD_CHUNK_SIZE: usize = 480; // 30ms at 16kHz
pub const VAD_HIDDEN_DIM: usize = 64;

pub use detector::WakeWordDetector;
