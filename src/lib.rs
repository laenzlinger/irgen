use std::ops::{Add, Div};

use hound::WavReader;
use num::complex::ComplexFloat;
use rustfft::{num_complex::Complex64, FftPlanner};

const SEGMENT_SIZE: usize = 131072; // 2^17
const IR_SIZE: usize = 2048; // 2^17
const MIN_DURATION_SECONDS: u32 = 30;
const ONE: Complex64 = Complex64::new(1.0, 0f64);
const MINUS_65_DB: f64 = 0.00056234132519;
const Q: f64 = 2.0 / 16777215.0;

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
    //   reader.seek(spec.sample_rate * SKIP_START_SECONDS).unwrap();

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(SEGMENT_SIZE);
    let ifft = planner.plan_fft_inverse(SEGMENT_SIZE);
    let mut mic = vec![Complex64::new(0.0, 0.0); SEGMENT_SIZE];
    let mut pickup = vec![Complex64::new(0.0, 0.0); SEGMENT_SIZE];
    let mut acc = vec![Complex64::new(0.0, 0.0); SEGMENT_SIZE];

    let samples = reader.samples::<i32>();

    let mut i = 0;
    let mut ch1 = true;
    let mut segment: u8 = 0;
    let mut count: u8 = 0;
    for sample in samples {
        let value = Complex64::new((sample.unwrap() as f64) * Q - 1.0, 0f64);
        if ch1 {
            pickup[i] = value;
            ch1 = false;
        } else {
            mic[i] = value;
            ch1 = true;
            i += 1;
        }
        if i == SEGMENT_SIZE {
            i = 0;
            // FIXME check for clipping and too_low
            if segment > 1 && count < 4 {
                apply_window(&mut mic);
                apply_window(&mut pickup);
                fft.process(&mut mic);
                fft.process(&mut pickup);
                apply_near_zero(&mut mic, &mut pickup);
                accumulate(&mic, &pickup, &mut acc);
                count += 1;
            }
            segment += 1;
        }
    }

    if count == 0 {
        panic!("No segments were processed");
    }

    normalize(count, &mut acc);
    ifft.process(&mut acc);
    write(String::from("test/out.wav"), &acc);
    count
}

fn accumulate(mic: &[Complex64], pickup: &[Complex64], acc: &mut [Complex64]) {
    for i in 0..SEGMENT_SIZE {
        let d = mic[i].div(pickup[i]);
        acc[i] = acc[i].add(d);
    }
}

fn normalize(count: u8, acc: &mut [Complex64]) {
    let c = Complex64::new(count as f64, 0f64);
    for i in 0..SEGMENT_SIZE {
        acc[i] = acc[i].div(c)
    }
}

fn apply_window(s: &mut [Complex64]) {
    let mut window = apodize::hanning_iter(SEGMENT_SIZE);
    for i in 0..SEGMENT_SIZE {
        let w = window.next().unwrap();
        s[i] = Complex64::new(s[i].re() * w, 0f64);
    }
}

fn apply_near_zero(mic: &mut [Complex64], pickup: &mut [Complex64]) {
    let mut near_zero = 0f64;
    for i in 0..SEGMENT_SIZE {
        let abs = pickup[i].abs();
        if abs > near_zero {
            near_zero = abs
        }
    }
    near_zero = near_zero * MINUS_65_DB;

    for i in 0..SEGMENT_SIZE {
        if pickup[i].abs() < near_zero {
            pickup[i] = ONE;
            mic[i] = ONE;
        }
    }
}

fn write(filename: String, acc: &[Complex64]) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(filename, spec).unwrap();
    for i in 1..IR_SIZE {
        writer.write_sample(acc[i].abs() as f32).unwrap();
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

    #[test]
    fn normalize_works() {
        let mut acc = vec![Complex64::new(6.0, 2.0); SEGMENT_SIZE];

        normalize(2, &mut acc);
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
