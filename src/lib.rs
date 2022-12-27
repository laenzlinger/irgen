use std::ops::{Add, Div};
use std::sync::Arc;

use hound::WavReader;
use num::complex::ComplexFloat;
use rustfft::{num_complex::Complex64, Fft, FftPlanner};

// Algorithm
const SEGMENT_SIZE: usize = 131072; // 2^17
const IR_SIZE: usize = 2048;
const ONE: Complex64 = Complex64::new(1.0, 0f64);
const MINUS_65_DB: f64 = 0.0005623413251903491;

// wav file handling
const SCALE_24_BIT_PCM: f64 = 8388608.0;
const SCALE_16_BIT_PCM: f64 = std::i16::MAX as f64;
const MIN_DURATION_SECONDS: u32 = 30;

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
    let mut segment = Segment::new(&mut planner);
    let mut acc = Accumulator::new(&mut planner);

    let samples = reader
        .samples::<i32>()
        .filter_map(|s| s.ok()) // ignore the errors while reading
        .map(|s| Complex64::new(s as f64 / SCALE_24_BIT_PCM, 0f64)); // normalize 24bit to +-1.0

    let mut i = 0;
    let mut ch1 = true;
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
            segment.process(&mut acc);
            i = 0;
        }
    }

    acc.process(output_file)
}

struct Segment {
    count: u8,
    mic: Vec<Complex64>,
    pickup: Vec<Complex64>,
    fft: Arc<dyn Fft<f64>>,
}

impl Segment {
    fn new(planner: &mut FftPlanner<f64>) -> Segment {
        let fft = planner.plan_fft_forward(SEGMENT_SIZE);
        Segment {
            count: 0,
            mic: vec![Complex64::new(0.0, 0.0); SEGMENT_SIZE],
            pickup: vec![Complex64::new(0.0, 0.0); SEGMENT_SIZE],
            fft,
        }
    }

    pub fn process(&mut self, acc: &mut Accumulator) -> bool {
        self.count += 1;
        if self.count < 3 || acc.count > 3 {
            return true;
        }
        // FIXME check for clipping and too_low
        self.apply_window();
        self.fft.process(&mut self.mic);
        self.fft.process(&mut self.pickup);
        let near_zero_count = self.apply_near_zero();
        acc.accumulate(&self, near_zero_count);
        return false;
    }

    fn apply_window(&mut self) {
        let mut window = apodize::blackman_iter(self.mic.len());
        for i in 0..self.mic.len() {
            let w = window.next().unwrap();
            self.mic[i] = Complex64::new(self.mic[i].re() * w, 0f64);
            self.pickup[i] = Complex64::new(self.pickup[i].re() * w, 0f64);
        }
    }

    fn apply_near_zero(&mut self) -> u64 {
        let mut count: u64 = 0;
        let near_zero = max(&self.pickup);
        for i in 0..self.mic.len() {
            if self.pickup[i].abs() < near_zero {
                self.pickup[i] = ONE;
                self.mic[i] = ONE;
                count += 1;
            }
        }
        count
    }
}

struct Accumulator {
    count: u8,
    near_zero_count: u64,
    result: Vec<Complex64>,
    ifft: Arc<dyn Fft<f64>>,
}

impl Accumulator {
    fn new(planner: &mut FftPlanner<f64>) -> Accumulator {
        let ifft = planner.plan_fft_inverse(SEGMENT_SIZE);
        Accumulator {
            count: 0,
            near_zero_count: 0,
            result: vec![Complex64::new(0.0, 0.0); SEGMENT_SIZE],
            ifft,
        }
    }

    pub fn process(&mut self, filename: String) -> u64 {
        // validate the number of segments accumulated
        if self.count == 0 {
            panic!("No segments were processed");
        }

        self.ifft.process(&mut self.result);
        self.normalize();
        self.write(filename);
        self.near_zero_count / (self.count as u64 * 2)
    }

    fn accumulate(&mut self, s: &Segment, near_zero_count: u64) {
        for i in 0..self.result.len() {
            let d = s.mic[i].div(s.pickup[i]);
            self.result[i] = self.result[i].add(d);
        }
        self.count += 1;
        self.near_zero_count += near_zero_count
    }

    fn normalize(&mut self) {
        let dividend = (self.count as usize * self.result.len()) as f64;
        let c = Complex64::new(dividend, 0f64);
        for i in 0..self.result.len() {
            self.result[i] = self.result[i].div(c)
        }
    }

    fn write(&self, filename: String) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 48000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(filename, spec).unwrap();
        for s in self.result[0..IR_SIZE].into_iter() {
            let sample = (s.re() * SCALE_16_BIT_PCM) as i32;
            writer.write_sample(sample).unwrap();
        }
    }
}

fn max(samples: &[Complex64]) -> f64 {
    samples
        .into_iter()
        .map(|c| c.abs())
        .reduce(f64::max)
        .unwrap()
        * MINUS_65_DB
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
}
