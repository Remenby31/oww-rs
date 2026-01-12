use anyhow::{Context, Result};
use ndarray::Array2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::collections::VecDeque;
use std::path::Path;

use crate::{CHUNK_SIZE, MEL_BINS};

pub struct Preprocessor {
    melspec_model: Session,
    mel_buffer: VecDeque<[f32; MEL_BINS]>,
    raw_buffer: Vec<i16>,
    max_mel_frames: usize,
}

impl Preprocessor {
    pub fn new(model_path: &Path, max_mel_frames: usize) -> Result<Self> {
        let melspec_model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .commit_from_file(model_path)
            .context("Failed to load melspectrogram model")?;

        Ok(Self {
            melspec_model,
            mel_buffer: VecDeque::with_capacity(max_mel_frames),
            raw_buffer: Vec::with_capacity(CHUNK_SIZE * 2),
            max_mel_frames,
        })
    }

    pub fn process(&mut self, audio: &[i16]) -> Result<usize> {
        self.raw_buffer.extend_from_slice(audio);

        let mut consumed = 0;

        while self.raw_buffer.len() >= CHUNK_SIZE {
            let chunk: Vec<i16> = self.raw_buffer.drain(..CHUNK_SIZE).collect();
            self.process_chunk(&chunk)?;
            consumed += CHUNK_SIZE;
        }

        Ok(consumed)
    }

    fn process_chunk(&mut self, audio: &[i16]) -> Result<()> {
        let audio_f32: Vec<f32> = audio.iter().map(|&s| s as f32).collect();
        let input = Array2::from_shape_vec((1, audio.len()), audio_f32)?;
        let tensor = Tensor::from_array(input)?;

        let outputs = self.melspec_model.run(ort::inputs![tensor])?;

        let output: ndarray::ArrayViewD<f32> = outputs[0]
            .try_extract_array()
            .context("Failed to extract melspec output")?;

        let shape = output.shape();

        if shape.len() == 3 {
            let n_frames = shape[1];

            for frame_idx in 0..n_frames {
                let mut mel_frame = [0.0f32; MEL_BINS];
                for bin in 0..MEL_BINS {
                    let val = output[[0, frame_idx, bin]];
                    mel_frame[bin] = val / 10.0 + 2.0;
                }

                if self.mel_buffer.len() >= self.max_mel_frames {
                    self.mel_buffer.pop_front();
                }
                self.mel_buffer.push_back(mel_frame);
            }
        }

        Ok(())
    }

    pub fn get_mel_frames(&self, n_frames: usize) -> Option<Array2<f32>> {
        if self.mel_buffer.len() < n_frames {
            return None;
        }

        let start = self.mel_buffer.len() - n_frames;
        let mut data = Vec::with_capacity(n_frames * MEL_BINS);

        for i in start..self.mel_buffer.len() {
            data.extend_from_slice(&self.mel_buffer[i]);
        }

        Array2::from_shape_vec((n_frames, MEL_BINS), data).ok()
    }

    pub fn mel_frame_count(&self) -> usize {
        self.mel_buffer.len()
    }

    pub fn reset(&mut self) {
        self.mel_buffer.clear();
        self.raw_buffer.clear();
    }
}
