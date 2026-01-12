use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, SampleRate, Stream, StreamConfig};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};

use crate::{CHUNK_SIZE, SAMPLE_RATE};

pub struct AudioCapture {
    _stream: Stream,
    receiver: Receiver<Vec<i16>>,
}

impl AudioCapture {
    pub fn new() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .context("No input device available")?;

        // Find a config that matches our requirements
        let supported_configs = device
            .supported_input_configs()
            .context("Failed to get supported configs")?;

        let config = supported_configs
            .filter(|c| c.channels() == 1 && c.sample_format() == SampleFormat::I16)
            .find(|c| {
                c.min_sample_rate().0 <= SAMPLE_RATE && c.max_sample_rate().0 >= SAMPLE_RATE
            })
            .map(|c| c.with_sample_rate(SampleRate(SAMPLE_RATE)))
            .or_else(|| {
                // Fallback: use any mono config and we'll handle conversion
                device
                    .supported_input_configs()
                    .ok()?
                    .find(|c| c.channels() == 1)
                    .map(|c| c.with_sample_rate(SampleRate(SAMPLE_RATE)))
            })
            .context("No suitable audio config found")?;

        let (sender, receiver) = mpsc::channel();
        let buffer = Arc::new(Mutex::new(Vec::with_capacity(CHUNK_SIZE * 2)));

        let stream = Self::build_stream(&device, &config.config(), sender, buffer)?;
        stream.play().context("Failed to start audio stream")?;

        Ok(Self {
            _stream: stream,
            receiver,
        })
    }

    fn build_stream(
        device: &cpal::Device,
        config: &StreamConfig,
        sender: Sender<Vec<i16>>,
        buffer: Arc<Mutex<Vec<i16>>>,
    ) -> Result<Stream> {
        let err_fn = |err| eprintln!("Audio stream error: {}", err);

        let stream = device.build_input_stream(
            config,
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                let mut buf = buffer.lock().unwrap();
                buf.extend_from_slice(data);

                // Send complete chunks
                while buf.len() >= CHUNK_SIZE {
                    let chunk: Vec<i16> = buf.drain(..CHUNK_SIZE).collect();
                    let _ = sender.send(chunk);
                }
            },
            err_fn,
            None,
        )?;

        Ok(stream)
    }

    /// Try to read a chunk of audio (non-blocking)
    pub fn try_read(&self) -> Option<Vec<i16>> {
        self.receiver.try_recv().ok()
    }

    /// Read a chunk of audio (blocking)
    pub fn read(&self) -> Result<Vec<i16>> {
        self.receiver.recv().context("Audio channel closed")
    }
}

/// Audio capture with f32 format support (some devices don't support i16)
pub struct AudioCaptureF32 {
    _stream: Stream,
    receiver: Receiver<Vec<i16>>,
}

impl AudioCaptureF32 {
    pub fn new() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .context("No input device available")?;

        let config = StreamConfig {
            channels: 1,
            sample_rate: SampleRate(SAMPLE_RATE),
            buffer_size: cpal::BufferSize::Fixed(CHUNK_SIZE as u32),
        };

        let (sender, receiver) = mpsc::channel();
        let buffer = Arc::new(Mutex::new(Vec::with_capacity(CHUNK_SIZE * 2)));

        let stream = Self::build_stream_f32(&device, &config, sender, buffer)?;
        stream.play().context("Failed to start audio stream")?;

        Ok(Self {
            _stream: stream,
            receiver,
        })
    }

    fn build_stream_f32(
        device: &cpal::Device,
        config: &StreamConfig,
        sender: Sender<Vec<i16>>,
        buffer: Arc<Mutex<Vec<i16>>>,
    ) -> Result<Stream> {
        let err_fn = |err| eprintln!("Audio stream error: {}", err);

        let stream = device.build_input_stream(
            config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // Convert f32 [-1.0, 1.0] to i16
                let samples: Vec<i16> = data
                    .iter()
                    .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
                    .collect();

                let mut buf = buffer.lock().unwrap();
                buf.extend_from_slice(&samples);

                // Send complete chunks
                while buf.len() >= CHUNK_SIZE {
                    let chunk: Vec<i16> = buf.drain(..CHUNK_SIZE).collect();
                    let _ = sender.send(chunk);
                }
            },
            err_fn,
            None,
        )?;

        Ok(stream)
    }

    /// Try to read a chunk of audio (non-blocking)
    pub fn try_read(&self) -> Option<Vec<i16>> {
        self.receiver.try_recv().ok()
    }

    /// Read a chunk of audio (blocking)
    pub fn read(&self) -> Result<Vec<i16>> {
        self.receiver.recv().context("Audio channel closed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_capture_init() {
        // This test may fail on systems without audio input
        let result = AudioCapture::new();
        // Just check it doesn't panic, may fail without hardware
        println!("AudioCapture init result: {:?}", result.is_ok());
    }
}
