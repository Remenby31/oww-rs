use anyhow::{Context, Result};
use ndarray::Array3;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::collections::VecDeque;
use std::path::Path;

use crate::embedding::EmbeddingExtractor;
use crate::preprocessing::Preprocessor;
use crate::vad::VoiceActivityDetector;
use crate::{CLASSIFICATION_FRAMES, EMBEDDING_DIM, EMBEDDING_WINDOW};

pub struct WakeWordDetector {
    preprocessor: Preprocessor,
    embedding_extractor: EmbeddingExtractor,
    classifier: Session,
    vad: Option<VoiceActivityDetector>,
    prediction_buffer: VecDeque<f32>,
    threshold: f32,
    warmup_frames: usize,
    frame_count: usize,
}

impl WakeWordDetector {
    pub fn new(model_dir: &Path, threshold: f32, use_vad: bool) -> Result<Self> {
        let melspec_path = model_dir.join("melspectrogram.onnx");
        let embedding_path = model_dir.join("embedding_model.onnx");
        let classifier_path = model_dir.join("hey_jarvis.onnx");

        let preprocessor = Preprocessor::new(&melspec_path, EMBEDDING_WINDOW * 2)?;
        let embedding_extractor = EmbeddingExtractor::new(&embedding_path, 120)?;

        let classifier = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .commit_from_file(&classifier_path)
            .context("Failed to load classifier model")?;

        let vad = if use_vad {
            let vad_path = model_dir.join("silero_vad.onnx");
            Some(VoiceActivityDetector::new(&vad_path, 0.5)?)
        } else {
            None
        };

        Ok(Self {
            preprocessor,
            embedding_extractor,
            classifier,
            vad,
            prediction_buffer: VecDeque::with_capacity(30),
            threshold,
            warmup_frames: 5,
            frame_count: 0,
        })
    }

    /// Process audio and return detection score
    pub fn predict(&mut self, audio: &[i16]) -> Result<f32> {
        // 1. VAD check (if enabled)
        if let Some(vad) = &mut self.vad {
            vad.predict(audio)?;
        }

        // 2. Preprocessing: audio -> mel frames
        let consumed = self.preprocessor.process(audio)?;

        if consumed == 0 {
            return Ok(self.last_score());
        }

        // 3. Get mel frames and extract embeddings
        if let Some(mel_frames) = self.preprocessor.get_mel_frames(EMBEDDING_WINDOW) {
            self.embedding_extractor.extract(&mel_frames)?;
        }

        // 4. Check if we have enough embeddings for classification
        if self.embedding_extractor.embedding_count() < CLASSIFICATION_FRAMES {
            return Ok(0.0);
        }

        // 5. Get embeddings and classify
        let embeddings = self
            .embedding_extractor
            .get_embeddings(CLASSIFICATION_FRAMES)
            .context("Not enough embeddings")?;

        // Reshape to [1, 16, 96]
        let (data, _offset) = embeddings.into_raw_vec_and_offset();
        let input_array = Array3::from_shape_vec(
            (1, CLASSIFICATION_FRAMES, EMBEDDING_DIM),
            data,
        )?;

        // Run classifier
        let tensor = Tensor::from_array(input_array)?;
        let outputs = self.classifier.run(ort::inputs![tensor])?;

        let output: ndarray::ArrayViewD<f32> = outputs[0]
            .try_extract_array()
            .context("Failed to extract classifier output")?;

        let shape = output.shape();

        // Handle different output shapes
        let mut score = if shape.len() == 3 {
            output[[0, 0, 0]]
        } else if shape.len() == 2 {
            output[[0, 0]]
        } else {
            output[[0]]
        };

        // 6. Apply warmup (zero first N frames)
        self.frame_count += 1;
        if self.frame_count <= self.warmup_frames {
            score = 0.0;
        }

        // 7. Apply VAD filter
        if let Some(vad) = &self.vad {
            if !vad.is_speech() {
                score = 0.0;
            }
        }

        // 8. Update prediction buffer
        if self.prediction_buffer.len() >= 30 {
            self.prediction_buffer.pop_front();
        }
        self.prediction_buffer.push_back(score);

        Ok(score)
    }

    /// Check if wake word was detected (score > threshold)
    pub fn detected(&self) -> bool {
        self.prediction_buffer
            .back()
            .map(|&s| s > self.threshold)
            .unwrap_or(false)
    }

    /// Get last prediction score
    pub fn last_score(&self) -> f32 {
        self.prediction_buffer.back().copied().unwrap_or(0.0)
    }

    /// Reset all buffers
    pub fn reset(&mut self) {
        self.preprocessor.reset();
        self.embedding_extractor.reset();
        if let Some(vad) = &mut self.vad {
            vad.reset();
        }
        self.prediction_buffer.clear();
        self.frame_count = 0;
    }

    /// Get threshold
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Set threshold
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }
}
