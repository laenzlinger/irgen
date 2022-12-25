use std::ops::{Add, Div};

use hound::WavReader;
use num::complex::ComplexFloat;
use rustfft::{num_complex::Complex64, FftPlanner};

const SEGMENT_SIZE: usize = 131072; // 2^17
const MIN_DURATION_SECONDS: u32 = 30;
const SKIP_START_SECONDS: u32 = 6;

pub fn generate_from_wav() -> u8 {
    let mut reader = WavReader::open("test/gibson.wav").expect("Failed to open WAV file");
    let spec = reader.spec();
    if spec.channels != 2 {
        panic!("only stereo wav files are supported");
    }
    if spec.sample_format == hound::SampleFormat::Float {
        panic!("float format is not supported");
    }
    let duration: f32 = reader.duration() as f32 / spec.sample_rate as f32;
    if duration < MIN_DURATION_SECONDS as f32 {
        panic!("sample needs to be at least {MIN_DURATION_SECONDS}s long, but was {duration:.2}s");
    }
    reader.seek(spec.sample_rate * SKIP_START_SECONDS).unwrap();

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(SEGMENT_SIZE);
    let ifft = planner.plan_fft_inverse(SEGMENT_SIZE);
    let mut mic = vec![Complex64::new(0.0, 0.0); SEGMENT_SIZE];
    let mut pickup = vec![Complex64::new(0.0, 0.0); SEGMENT_SIZE];
    let mut acc = vec![Complex64::new(0.0, 0.0); SEGMENT_SIZE];

    let samples = reader.samples::<i32>();

    let mut i = 0;
    let mut ch1 = true;
    let mut count: u8 = 0;
    for sample in samples {
        let value = Complex64::new(sample.unwrap() as f64, 0f64);
        if ch1 {
            mic[i] = value;
            ch1 = false;
        } else {
            pickup[i] = value;
            ch1 = true;
            i += 1;
        }
        if i == SEGMENT_SIZE {
            i = 0;
            fft.process(&mut mic);
            fft.process(&mut pickup);
            accumulate(&mic, &pickup, &mut acc);
            count += 1
        }
    }

    if count == 0 {
        panic!("No segments were processed");
    }

    normalize(count, &mut acc);
    ifft.process(&mut acc);
    write(&acc);
    count
}

fn accumulate(mic: &[Complex64], pickup: &[Complex64], acc: &mut [Complex64]) {
    for i in 0..SEGMENT_SIZE {
        let d = mic[i].div(pickup[i]);
        acc[i] = d.add(acc[i]);
    }
}

fn normalize(count: u8, acc: &mut [Complex64]) {
    let c = Complex64::new(count as f64, 0f64);
    for i in 0..SEGMENT_SIZE {
        acc[i] = acc[i].div(c)
    }
}

fn write(acc: &[Complex64]) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create("test/out.wav", spec).unwrap();
    for t in acc {
        writer.write_sample(t.abs() as f32).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = generate_from_wav();
        assert_eq!(result, 4);
    }
}
