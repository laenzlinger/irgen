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
    let mut samples = reader
        .samples::<i32>()
        .filter_map(|s| s.ok()) // ignore the errors while reading
        .map(|s| s as f64 / SCALE_24_BIT_PCM); // normalize 24bit to +-1.0

    let mut generator = Generator::new();
    let mut done = false;
    while !done {
        done = samples
            .next()
            .zip(samples.next())
            .map(Frame::new)
            .map_or(true, |frame| generator.process(frame));
    }

    generator.write(output_file);
    generator.avg_near_zero_count()
}

struct Frame {
    pickup: f64,
    mic: f64,
}

impl Frame {
    fn new(frame: (f64, f64)) -> Frame {
        Frame {
            pickup: frame.0,
            mic: frame.1,
        }
    }
}

struct Generator {
    segment: Segment,
    accu: Accumulator,
    frame_count: usize,
}

impl Generator {
    pub fn new() -> Generator {
        let mut planner = FftPlanner::<f64>::new();
        let segment = Segment::new(&mut planner);
        let accu = Accumulator::new(&mut planner);

        Generator {
            segment,
            accu,
            frame_count: 0,
        }
    }

    pub fn process(&mut self, frame: Frame) -> bool {
        if self.accu.done() {
            return true;
        }
        self.segment.mic[self.frame_count] = Complex64::new(frame.mic, 0f64);
        self.segment.pickup[self.frame_count] = Complex64::new(frame.pickup, 0f64);
        self.frame_count += 1;
        if self.frame_count == SEGMENT_SIZE {
            self.frame_count = 0;
            let done = self.segment.process(&mut self.accu);
            if done {
                self.accu.process();
                return true;
            }
        }
        return false;
    }

    pub fn avg_near_zero_count(&self) -> u64 {
        self.accu.avg_near_zero_count()
    }

    pub fn write(&self, file_name: String) {
        self.accu.write(file_name);
    }
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

    fn process(&mut self, accu: &mut Accumulator) -> bool {
        self.count += 1;
        if self.count < 3 {
            return false;
        }
        if accu.done() {
            return true;
        }

        // FIXME check for clipping and too_low
        self.apply_window();
        self.fft.process(&mut self.mic);
        self.fft.process(&mut self.pickup);
        let near_zero_count = self.apply_near_zero();
        accu.accumulate(&self, near_zero_count);
        accu.done()
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

    fn process(&mut self) {
        // validate the number of segments accumulated
        if self.count == 0 {
            panic!("No segments were processed");
        }

        self.ifft.process(&mut self.result);
        self.normalize();
    }

    fn avg_near_zero_count(&self) -> u64 {
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

    fn done(&self) -> bool {
        self.count > 3
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
