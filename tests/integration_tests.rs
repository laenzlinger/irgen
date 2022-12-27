use irgen;

#[test]
fn test_generator() {
    let result = irgen::generate_from_wav(
        String::from("tests/data/gibson.wav"),
        String::from("tests/data/out.wav"),
    );
    assert_eq!(result, 16560);
}
