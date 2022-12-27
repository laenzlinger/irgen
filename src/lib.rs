use std::ops::{Add, Div};

use hound::WavReader;
use num::complex::ComplexFloat;
use rustfft::{num_complex::Complex64, FftPlanner};

// Algorithm
const SEGMENT_SIZE: usize = 131072; // 2^17
const IR_SIZE: usize = 2048;
const ONE: Complex64 = Complex64::new(1.0, 0f64);
const MINUS_65_DB: f64 = 0.0005623413251903491;

// wav file handling
const SCALE_24_BIT_PCM: f64 = 8388608.0;
const SCALE_16_BIT_PCM: f64 = std::i16::MAX as f64;
const MIN_DURATION_SECONDS: u32 = 30;

struct Segement {
    mic: Vec<Complex64>,
    pickup: Vec<Complex64>,
}

pub fn generate_from_wav(input_file: String, output_file: String) -> u64 {
    let mut reader = WavReader::open(input_file).expect("Failed to open WAV file");
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

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(SEGMENT_SIZE);
    let mut acc = vec![Complex64::new(0.0, 0.0); SEGMENT_SIZE];
    let mut segment = Segement {
        mic: vec![Complex64::new(0.0, 0.0); SEGMENT_SIZE],
        pickup: vec![Complex64::new(0.0, 0.0); SEGMENT_SIZE],
    };

    let samples = reader
        .samples::<i32>()
        .filter_map(|s| s.ok()) // ignore the errors while reading
        .map(|s| Complex64::new(s as f64 / SCALE_24_BIT_PCM, 0f64)); // normalize 24bit to +-1.0

    let mut i = 0;
    let mut ch1 = true;
    let mut segment_nr: u8 = 0;
    let mut count: u8 = 0;
    let mut nzcount: u64 = 0;
    for sample in samples {
        if ch1 {
            segment.pickup[i] = sample;
            ch1 = false;
        } else {
            segment.mic[i] = sample;
            ch1 = true;
            i += 1;
        }
        if i == SEGMENT_SIZE {
            // FIXME check for clipping and too_low
            if segment_nr > 1 && count < 4 {
                apply_window(&mut segment.mic);
                apply_window(&mut segment.pickup);
                fft.process(&mut segment.mic);
                fft.process(&mut segment.pickup);
                nzcount += apply_near_zero(&mut segment.mic, &mut segment.pickup);
                accumulate(&segment.mic, &segment.pickup, &mut acc);
                count += 1;
            }
            i = 0;
            segment_nr += 1;
        }
    }

    if count == 0 {
        panic!("No segments were processed");
    }

    let ifft = planner.plan_fft_inverse(SEGMENT_SIZE);
    ifft.process(&mut acc);
    normalize(&mut acc, (count as usize * SEGMENT_SIZE) as f64);
    write(output_file, &acc[0..IR_SIZE]);
    nzcount / (count as u64 * 2)
}

fn accumulate(mic: &[Complex64], pickup: &[Complex64], acc: &mut [Complex64]) {
    for i in 0..acc.len() {
        let d = mic[i].div(pickup[i]);
        acc[i] = acc[i].add(d);
    }
}

fn normalize(acc: &mut [Complex64], dividend: f64) {
    let c = Complex64::new(dividend, 0f64);
    for i in 0..acc.len() {
        acc[i] = acc[i].div(c)
    }
}

fn apply_window(s: &mut [Complex64]) {
    let mut window = apodize::blackman_iter(s.len());
    for i in 0..s.len() {
        let w = window.next().unwrap();
        s[i] = Complex64::new(s[i].re() * w, 0f64);
    }
}

fn apply_near_zero(mic: &mut [Complex64], pickup: &mut [Complex64]) -> u64 {
    let mut count: u64 = 0;
    let near_zero = max(pickup);
    for i in 0..mic.len() {
        if pickup[i].abs() < near_zero {
            pickup[i] = ONE;
            mic[i] = ONE;
            count += 1;
        }
    }
    count
}

fn max(samples: &mut [Complex64]) -> f64 {
    samples
        .into_iter()
        .map(|c| c.abs())
        .reduce(f64::max)
        .unwrap()
        * MINUS_65_DB
}

fn write(filename: String, samples: &[Complex64]) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(filename, spec).unwrap();
    for s in samples.into_iter() {
        let sample = (s.re() * SCALE_16_BIT_PCM) as i32;
        writer.write_sample(sample).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = generate_from_wav(
            String::from("test/gibson.wav"),
            String::from("test/out.wav"),
        );
        assert_eq!(result, 16560);
    }

    #[test]
    fn normalize_works() {
        let mut acc = vec![Complex64::new(6.0, 2.0); SEGMENT_SIZE];

        normalize(&mut acc, 2.0);
        assert_eq!(acc[0].re(), 3.0);
        assert_eq!(acc[0].im(), 1.0);
    }

    #[test]
    fn test_write() {
        let mut acc = vec![Complex64::new(0.0, 0.0); SEGMENT_SIZE];
        for j in 0..SEGMENT_SIZE {
            let val = std::f64::consts::PI * 2.0 * j as f64 / 44.1;
            acc[j] = Complex64::new(0.1 * ((val.sin() + 1.0) / 2.0), 0f64);
        }
        write(String::from("test/sin.wav"), &acc)
    }
}
