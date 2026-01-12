use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

use oww_rs::audio::AudioCaptureF32;
use oww_rs::WakeWordDetector;

#[derive(Parser)]
#[command(name = "oww-rs")]
#[command(about = "Rust wake word detector - OpenWakeWord port")]
struct Args {
    /// Path to models directory
    #[arg(short, long, default_value = "models")]
    model_dir: PathBuf,

    /// Detection threshold (0.0 - 1.0)
    #[arg(short, long, default_value = "0.5")]
    threshold: f32,

    /// Command to execute on detection
    #[arg(short, long)]
    command: Option<String>,

    /// Enable VAD (Voice Activity Detection)
    #[arg(long, default_value = "true")]
    vad: bool,

    /// Cooldown after detection (seconds)
    #[arg(long, default_value = "2.0")]
    cooldown: f32,

    /// Show scores continuously
    #[arg(long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("oww-rs - Rust Wake Word Detector");
    println!("================================");
    println!("Model dir: {}", args.model_dir.display());
    println!("Threshold: {}", args.threshold);
    println!("VAD: {}", if args.vad { "enabled" } else { "disabled" });
    println!();

    // Initialize detector
    print!("Loading models...");
    let start = Instant::now();
    let mut detector = WakeWordDetector::new(&args.model_dir, args.threshold, args.vad)?;
    println!(" done ({:.2}s)", start.elapsed().as_secs_f32());

    // Initialize audio capture
    print!("Initializing audio...");
    let audio = AudioCaptureF32::new()?;
    println!(" done");

    println!();
    println!("Listening for 'hey jarvis'... (Ctrl+C to quit)");
    println!();

    let cooldown_duration = Duration::from_secs_f32(args.cooldown);
    let mut last_detection = Instant::now() - cooldown_duration;

    loop {
        if let Some(chunk) = audio.try_read() {
            let score = detector.predict(&chunk)?;

            if args.verbose {
                print!("\rScore: {:.3}", score);
                std::io::Write::flush(&mut std::io::stdout())?;
            }

            // Check detection with cooldown
            let now = Instant::now();
            if detector.detected() && now.duration_since(last_detection) >= cooldown_duration {
                last_detection = now;

                if args.verbose {
                    println!();
                }
                println!(">>> DETECTED! (score: {:.3}) <<<", score);

                // Execute command if provided
                if let Some(cmd) = &args.command {
                    println!("Executing: {}", cmd);
                    Command::new("sh").arg("-c").arg(cmd).spawn()?;
                }
            }
        }

        // Small sleep to avoid busy loop
        std::thread::sleep(Duration::from_millis(10));
    }
}
