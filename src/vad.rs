use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Array3};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::collections::VecDeque;
use std::path::Path;

use crate::{SAMPLE_RATE, VAD_CHUNK_SIZE, VAD_HIDDEN_DIM};

pub struct VoiceActivityDetector {
    model: Session,
    h: Array3<f32>,
    c: Array3<f32>,
    sample_rate: i64,
    prediction_buffer: VecDeque<f32>,
    threshold: f32,
}

impl VoiceActivityDetector {
    pub fn new(model_path: &Path, threshold: f32) -> Result<Self> {
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .commit_from_file(model_path)
            .context("Failed to load VAD model")?;

        let h = Array3::<f32>::zeros((2, 1, VAD_HIDDEN_DIM));
        let c = Array3::<f32>::zeros((2, 1, VAD_HIDDEN_DIM));

        Ok(Self {
            model,
            h,
            c,
            sample_rate: SAMPLE_RATE as i64,
            prediction_buffer: VecDeque::with_capacity(125),
            threshold,
        })
    }

    pub fn predict(&mut self, audio: &[i16]) -> Result<f32> {
        let frame_size = VAD_CHUNK_SIZE;
        let mut frame_predictions = Vec::new();

        for chunk_start in (0..audio.len()).step_by(frame_size) {
            let chunk_end = (chunk_start + frame_size).min(audio.len());
            let chunk = &audio[chunk_start..chunk_end];

            if chunk.len() < frame_size {
                break;
            }

            let normalized: Vec<f32> = chunk.iter().map(|&s| s as f32 / 32767.0).collect();
            let input_array = Array2::from_shape_vec((1, frame_size), normalized)?;
            let sr_array = Array1::from_elem(1, self.sample_rate);

            let input_tensor = Tensor::from_array(input_array)?;
            let h_tensor = Tensor::from_array(self.h.clone())?;
            let c_tensor = Tensor::from_array(self.c.clone())?;
            let sr_tensor = Tensor::from_array(sr_array)?;

            let outputs = self.model.run(ort::inputs![
                input_tensor, sr_tensor, h_tensor, c_tensor
            ])?;

            let out: ndarray::ArrayViewD<f32> = outputs[0]
                .try_extract_array()
                .context("Failed to extract VAD output")?;

            let new_h: ndarray::ArrayViewD<f32> = outputs[1]
                .try_extract_array()
                .context("Failed to extract VAD h state")?;

            let new_c: ndarray::ArrayViewD<f32> = outputs[2]
                .try_extract_array()
                .context("Failed to extract VAD c state")?;

            for i in 0..2 {
                for j in 0..VAD_HIDDEN_DIM {
                    self.h[[i, 0, j]] = new_h[[i, 0, j]];
                    self.c[[i, 0, j]] = new_c[[i, 0, j]];
                }
            }

            let score = out[[0, 0]];
            frame_predictions.push(score);
        }

        let mean_score = if frame_predictions.is_empty() {
            0.0
        } else {
            frame_predictions.iter().sum::<f32>() / frame_predictions.len() as f32
        };

        if self.prediction_buffer.len() >= 125 {
            self.prediction_buffer.pop_front();
        }
        self.prediction_buffer.push_back(mean_score);

        Ok(mean_score)
    }

    pub fn is_speech(&self) -> bool {
        let len = self.prediction_buffer.len();
        if len < 7 {
            return false;
        }

        let start = len.saturating_sub(7);
        let end = len.saturating_sub(4);

        let max_score = self
            .prediction_buffer
            .range(start..end)
            .copied()
            .fold(0.0f32, f32::max);

        max_score >= self.threshold
    }

    pub fn reset(&mut self) {
        self.h.fill(0.0);
        self.c.fill(0.0);
        self.prediction_buffer.clear();
    }

    pub fn last_score(&self) -> f32 {
        self.prediction_buffer.back().copied().unwrap_or(0.0)
    }
}
