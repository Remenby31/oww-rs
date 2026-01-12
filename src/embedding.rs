use anyhow::{Context, Result};
use ndarray::{Array2, Array4};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::collections::VecDeque;
use std::path::Path;

use crate::{EMBEDDING_DIM, EMBEDDING_STEP, EMBEDDING_WINDOW, MEL_BINS};

pub struct EmbeddingExtractor {
    model: Session,
    embedding_buffer: VecDeque<[f32; EMBEDDING_DIM]>,
    max_embeddings: usize,
}

impl EmbeddingExtractor {
    pub fn new(model_path: &Path, max_embeddings: usize) -> Result<Self> {
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .commit_from_file(model_path)
            .context("Failed to load embedding model")?;

        Ok(Self {
            model,
            embedding_buffer: VecDeque::with_capacity(max_embeddings),
            max_embeddings,
        })
    }

    pub fn extract(&mut self, mel_frames: &Array2<f32>) -> Result<usize> {
        let n_frames = mel_frames.shape()[0];

        if n_frames < EMBEDDING_WINDOW {
            return Ok(0);
        }

        let mut windows = Vec::new();
        let mut window_start = 0;

        while window_start + EMBEDDING_WINDOW <= n_frames {
            windows.push(window_start);
            window_start += EMBEDDING_STEP;
        }

        if windows.is_empty() {
            return Ok(0);
        }

        let batch_size = windows.len();
        let mut batch_data = Vec::with_capacity(batch_size * EMBEDDING_WINDOW * MEL_BINS);

        for &start in &windows {
            for frame_idx in start..start + EMBEDDING_WINDOW {
                for bin in 0..MEL_BINS {
                    batch_data.push(mel_frames[[frame_idx, bin]]);
                }
            }
        }

        let input = Array4::from_shape_vec(
            (batch_size, EMBEDDING_WINDOW, MEL_BINS, 1),
            batch_data,
        )?;
        let tensor = Tensor::from_array(input)?;

        let outputs = self.model.run(ort::inputs![tensor])?;

        let output: ndarray::ArrayViewD<f32> = outputs[0]
            .try_extract_array()
            .context("Failed to extract embedding output")?;

        let output_shape = output.shape();

        for batch_idx in 0..output_shape[0] {
            let mut embedding = [0.0f32; EMBEDDING_DIM];
            for dim in 0..EMBEDDING_DIM {
                embedding[dim] = output[[batch_idx, dim]];
            }

            if self.embedding_buffer.len() >= self.max_embeddings {
                self.embedding_buffer.pop_front();
            }
            self.embedding_buffer.push_back(embedding);
        }

        Ok(windows.len())
    }

    pub fn get_embeddings(&self, n: usize) -> Option<Array2<f32>> {
        if self.embedding_buffer.len() < n {
            return None;
        }

        let start = self.embedding_buffer.len() - n;
        let mut data = Vec::with_capacity(n * EMBEDDING_DIM);

        for i in start..self.embedding_buffer.len() {
            data.extend_from_slice(&self.embedding_buffer[i]);
        }

        Array2::from_shape_vec((n, EMBEDDING_DIM), data).ok()
    }

    pub fn embedding_count(&self) -> usize {
        self.embedding_buffer.len()
    }

    pub fn reset(&mut self) {
        self.embedding_buffer.clear();
    }
}
