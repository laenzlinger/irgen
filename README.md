# irgen
Create IR files to enhance the signal of an acoustic instrument pickups.

Library for creating Impulse Response (IR) files by comparing the
sound of a microphone with the sound of the pickup.

The following algorithms are implemented:

* [John Fields open source algorithm](http://acousticir.free.fr/spip.php?article136&var_mode=calcul)

WARNING: The API is still under development and will very likely change
(especially in case new algorithms would be added).
Therefore, Ppease don't expect any API stability at the moment.

## Installation

Pre-Requesite: Install [Rust](https://www.rust-lang.org/tools/install)

```
cargo install irgen
```

## Command Line Interface

Run the help:
```
irgen -h
create IR files to enahnce the signal of an acoustic instrument pickups.

Usage: irgen --input-file <INPUT_FILE> --output-file <OUTPUT_FILE>

Options:
  -i, --input-file <INPUT_FILE>    Stereo input .waf data (left: pickup, right: mic)
  -o, --output-file <OUTPUT_FILE>  Output IR .wav file
  -h, --help                       Print help information
  -V, --version                    Print version information
```
