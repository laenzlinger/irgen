use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Stereo input .waf data (left: pickup, right: mic)
    #[arg(short, long)]
    input_file: String,

    /// Output IR .wav file.
    #[arg(short, long)]
    output_file: String,
}

pub fn main() {
    let args = Args::parse();
    irgen::generate_from_wav(args.input_file, args.output_file, Default::default());
}
