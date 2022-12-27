use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Stereo .waf file (left: pickup, right: mic)
    #[arg(short, long)]
    input_file: String,

    /// File to write the IR wav file.
    #[arg(short, long)]
    output_file: String,
}

pub fn main() {
    let args = Args::parse();
    irgen::generate_from_wav(args.input_file, args.output_file);
}
