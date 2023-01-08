use easyfft::num_complex::Complex64;
use easyfft::num_complex::ComplexFloat;
use easyfft::prelude::*;
use hound::{SampleFormat, WavReader};
use num::Zero;
use std::ops::{Add, Div};

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
    /// Above this value the input is considered to be clipping.
    /// A segment is ignored in case its maximum is above this value.
    ///
    /// Range [0.0, 1.0]
    clip: f64,

    /// Below this value the input is considered to be too small.
    /// A segment is ignored in case its maximum is below this value.
    /// Range [0.0, 1.0]
    too_low: f64,

    /// In case a frequency of the pickup is below this value (relative to the maximum)
    /// it is ignored.
    near_zero: f64,
}

pub struct Options {
    /// The number of samples which are analyzed.
    segment_size: usize,

    /// The size of the output impulse response.
    ir_size: usize,

    /// The sample rate of the input/output data.
    /// Relevant only to write the ouptut to a file.
    sample_rate: u32,

    /// The amount of bits per sample. (16 or 24).
    /// Relevant only to write the ouptut to a file.
    bits_per_sample: u16,

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
            bits_per_sample: 16,
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
        let segment = Segment::new(&options);
        let accu = Accumulator::new(options.segment_size);

        Generator {
            segment,
            accu,
            options,
        }
    }

    /// Process the given Frame of data.
    ///
    /// Returns true if the generator has received enough data to generate the output.
    pub fn process(&mut self, frame: Frame) -> bool {
        if self.accu.done() {
            return true;
        }
        self.segment.add(frame, &mut self.accu, &self.options)
    }

    /// Returns the result of the Generator.
    ///
    /// Panics if the Generator did not yet process enough data.
    pub fn result(&self) -> Result {
        Result {
            avg_near_zero_count: self.accu.avg_near_zero_count(),
            segment_count: self.accu.count,
            impulse_response: self.accu.result.iter().map(|s| s.re()).collect(),
        }
    }

    /// Write the result of the Generator to a .wav file.
    ///
    /// # Arguments
    ///
    /// * `file_name` - A string that contains the path/file name.
    ///
    /// Panics if the Generator did not yet process enough data.
    pub fn write(&self, file_name: String) {
        self.accu.write(file_name, &self.options);
    }
}

pub struct Result {
    pub avg_near_zero_count: u64,
    pub segment_count: u32,
    pub impulse_response: Vec<f64>,
}

struct Segment {
    mic: Vec<Complex64>,
    pickup: Vec<Complex64>,
    count: u32,
    frame_count: usize,
}

impl Segment {
    fn new(options: &Options) -> Segment {
        Segment {
            count: 0,
            frame_count: 0,
            mic: vec![Complex64::zero(); options.segment_size],
            pickup: vec![Complex64::zero(); options.segment_size],
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
        self.mic.fft_mut();
        self.pickup.fft_mut();
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
}

impl Accumulator {
    fn new(segment_size: usize) -> Accumulator {
        Accumulator {
            count: 0,
            near_zero_count: 0,
            result: vec![Complex64::zero(); segment_size],
        }
    }

    fn process(&mut self) {
        // validate the number of segments accumulated
        if self.count == 0 {
            panic!("No segments were processed");
        }

        self.result.ifft_mut();
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
            bits_per_sample: options.bits_per_sample,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(filename, spec).unwrap();
        let scale_factor = scale_factor(spec.bits_per_sample);
        for s in self.result[0..options.ir_size]
            .iter()
            .map(|s| (s.re() * scale_factor) as i32)
        {
            writer.write_sample(s).unwrap();
        }
    }
}

fn max(samples: &[Complex64]) -> f64 {
    samples.iter().map(|c| c.abs()).reduce(f64::max).unwrap()
}

pub const SCALE_24_BIT_PCM: f64 = 8388608.0;
pub const SCALE_16_BIT_PCM: f64 = std::i16::MAX as f64;

fn scale_factor(bits_per_sample: u16) -> f64 {
    match bits_per_sample {
        24 => SCALE_24_BIT_PCM,
        16 => SCALE_16_BIT_PCM,
        _ => panic!("Input .waf contains unsupported 'bits per sample' value."),
    }
}

/// Convenience function to run the Generator with from a .wav file as data source.
pub fn generate_from_wav(
    input_file: String,
    output_file: Option<String>,
    options: Options,
) -> Result {
    let mut reader = WavReader::open(input_file).expect("Failed to open WAV file.");
    let spec = reader.spec();
    if spec.channels != 2 {
        panic!("Only stereo .wav files are supported.");
    }
    let scale_factor = match spec.sample_format {
        SampleFormat::Int => scale_factor(spec.bits_per_sample),
        SampleFormat::Float => 1.0,
    };

    let mut samples = reader
        .samples::<i32>()
        .filter_map(|s| s.ok()) // ignore the errors while reading
        .map(|s| s as f64 / scale_factor); // normalize to +-1.0

    let options = Options {
        sample_rate: spec.sample_rate,
        ..options
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
    if let Some(of) = output_file {
        generator.write(of)
    }
    generator.result()
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
