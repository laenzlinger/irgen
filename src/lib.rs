use hound::{SampleFormat, WavReader};
use num::Zero;
use std::ops::{Add, Div};
use std::sync::Arc;

use num::complex::ComplexFloat;
use rustfft::{num_complex::Complex64, Fft, FftPlanner};

// wav file handling
pub const SCALE_24_BIT_PCM: f64 = 8388608.0;
pub const SCALE_16_BIT_PCM: f64 = std::i16::MAX as f64;

const ONE: Complex64 = Complex64::new(1.0, 0f64);

pub struct Frame {
    pickup: f64,
    mic: f64,
}

impl Frame {
    pub fn new(frame: (f64, f64)) -> Frame {
        Frame {
            pickup: frame.0,
            mic: frame.1,
        }
    }
}

pub struct Thresholds {
    clip: f64,
    too_low: f64,
    near_zero: f64,
}

pub struct Options {
    segment_size: usize,
    ir_size: usize,
    sample_rate: u32,
    thresholds: Thresholds,
}

impl Default for Thresholds {
    fn default() -> Self {
        Thresholds {
            clip: 0.999,
            too_low: 0.178,
            near_zero: 0.0005623413251903491, // -65dB
        }
    }
}

impl Default for Options {
    fn default() -> Self {
        Options {
            segment_size: 131072, // 2^17
            sample_rate: 48000,
            ir_size: 2048,
            thresholds: Default::default(),
        }
    }
}

pub struct Generator {
    segment: Segment,
    accu: Accumulator,
    options: Options,
}

impl Default for Generator {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl Generator {
    pub fn new(options: Options) -> Generator {
        let mut planner = FftPlanner::<f64>::new();
        let segment = Segment::new(&mut planner, &options);
        let accu = Accumulator::new(&mut planner, options.segment_size);

        Generator {
            segment,
            accu,
            options,
        }
    }

    pub fn process(&mut self, frame: Frame) -> bool {
        if self.accu.done() {
            return true;
        }
        self.segment.add(frame, &mut self.accu, &self.options)
    }

    pub fn avg_near_zero_count(&self) -> u64 {
        self.accu.avg_near_zero_count()
    }

    pub fn write(&self, file_name: String) {
        self.accu.write(file_name, &self.options);
    }
}

struct Segment {
    mic: Vec<Complex64>,
    pickup: Vec<Complex64>,
    fft: Arc<dyn Fft<f64>>,
    count: u32,
    frame_count: usize,
}

impl Segment {
    fn new(planner: &mut FftPlanner<f64>, options: &Options) -> Segment {
        let fft = planner.plan_fft_forward(options.segment_size);
        Segment {
            count: 0,
            frame_count: 0,
            mic: vec![Complex64::zero(); options.segment_size],
            pickup: vec![Complex64::zero(); options.segment_size],
            fft,
        }
    }

    fn add(&mut self, frame: Frame, accu: &mut Accumulator, options: &Options) -> bool {
        self.mic[self.frame_count] = Complex64::new(frame.mic, 0f64);
        self.pickup[self.frame_count] = Complex64::new(frame.pickup, 0f64);
        self.frame_count += 1;
        let ready = self.frame_count == self.mic.len();
        if ready {
            self.frame_count = 0;
            let done = self.process(accu, options);
            if done {
                accu.process();
                return true;
            }
        }
        false
    }

    fn process(&mut self, accu: &mut Accumulator, options: &Options) -> bool {
        self.count += 1;
        if accu.done() {
            return true;
        }
        if self.count < 3 {
            return false;
        }
        if !self.is_valid(&options.thresholds) {
            return false;
        }

        self.apply_window();
        self.fft.process(&mut self.mic);
        self.fft.process(&mut self.pickup);
        let near_zero_count = self.apply_near_zero(&options.thresholds);
        accu.accumulate(self, near_zero_count);
        accu.done()
    }

    fn is_valid(&mut self, thresholds: &Thresholds) -> bool {
        let max = max(&self.mic).max(max(&self.pickup));
        let clip = max > thresholds.clip;
        let too_low = max < thresholds.too_low;
        !(clip || too_low)
    }
    fn apply_window(&mut self) {
        let mut window = apodize::blackman_iter(self.mic.len());
        for i in 0..self.mic.len() {
            let w = window.next().unwrap();
            self.mic[i] = Complex64::new(self.mic[i].re() * w, 0f64);
            self.pickup[i] = Complex64::new(self.pickup[i].re() * w, 0f64);
        }
    }

    fn apply_near_zero(&mut self, thresholds: &Thresholds) -> u64 {
        let mut count: u64 = 0;
        let near_zero = max(&self.pickup) * thresholds.near_zero;
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
    count: u32,
    near_zero_count: u64,
    result: Vec<Complex64>,
    ifft: Arc<dyn Fft<f64>>,
}

impl Accumulator {
    fn new(planner: &mut FftPlanner<f64>, segment_size: usize) -> Accumulator {
        let ifft = planner.plan_fft_inverse(segment_size);
        Accumulator {
            count: 0,
            near_zero_count: 0,
            result: vec![Complex64::zero(); segment_size],
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

    fn write(&self, filename: String, options: &Options) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: options.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(filename, spec).unwrap();
        for s in self.result[0..options.ir_size].iter() {
            let sample = (s.re() * SCALE_16_BIT_PCM) as i32;
            writer.write_sample(sample).unwrap();
        }
    }
}

fn max(samples: &[Complex64]) -> f64 {
    samples.iter().map(|c| c.abs()).reduce(f64::max).unwrap()
}

pub fn generate_from_wav(input_file: String, output_file: String) -> u64 {
    let mut reader = WavReader::open(input_file).expect("Failed to open WAV file.");
    let spec = reader.spec();
    if spec.channels != 2 {
        panic!("Only stereo .wav files are supported.");
    }
    let scale_factor = match spec.sample_format {
        SampleFormat::Int => match spec.bits_per_sample {
            24 => SCALE_24_BIT_PCM,
            16 => SCALE_16_BIT_PCM,
            _ => panic!("Input .waf contains unsupported 'bits per sample' value."),
        },
        SampleFormat::Float => 1.0,
    };

    let mut samples = reader
        .samples::<i32>()
        .filter_map(|s| s.ok()) // ignore the errors while reading
        .map(|s| s as f64 / scale_factor); // normalize to +-1.0

    let options = Options {
        sample_rate: spec.sample_rate,
        ..Default::default()
    };
    let mut generator = Generator::new(options);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_create() {
        let generator: Generator = Default::default();
        assert_eq!(generator.options.sample_rate, 48000)
    }
}
