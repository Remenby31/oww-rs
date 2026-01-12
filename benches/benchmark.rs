use criterion::{criterion_group, criterion_main, Criterion};
use oww_rs::WakeWordDetector;
use std::path::PathBuf;

fn benchmark_inference(c: &mut Criterion) {
    let model_dir = PathBuf::from("models");
    let mut detector = WakeWordDetector::new(&model_dir, 0.5, false).unwrap();

    // Create 80ms of silence (1280 samples at 16kHz)
    let audio: Vec<i16> = vec![0i16; 1280];

    c.bench_function("inference_80ms_chunk", |b| {
        b.iter(|| {
            detector.predict(&audio).unwrap()
        })
    });
}

fn benchmark_inference_with_vad(c: &mut Criterion) {
    let model_dir = PathBuf::from("models");
    let mut detector = WakeWordDetector::new(&model_dir, 0.5, true).unwrap();

    // Create 80ms of silence (1280 samples at 16kHz)
    let audio: Vec<i16> = vec![0i16; 1280];

    c.bench_function("inference_80ms_with_vad", |b| {
        b.iter(|| {
            detector.predict(&audio).unwrap()
        })
    });
}

criterion_group!(benches, benchmark_inference, benchmark_inference_with_vad);
criterion_main!(benches);
