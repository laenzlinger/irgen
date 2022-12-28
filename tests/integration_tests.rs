#[test]
fn test_generator() {
    let result = irgen::generate_from_wav(
        String::from("tests/data/gibson.wav"),
        String::from("tests/data/out.wav"),
    );
    assert_eq!(result.avg_near_zero_count, 16560);
    assert_eq!(result.segment_count, 4);
    assert_eq!(result.impulse_response.len(), 131072);
    assert_eq!(
        result.impulse_response[0..10],
        [
            0.2483326857782041,
            -0.15221656751203486,
            0.015689695799776474,
            -0.010657662902112286,
            0.004957049143985034,
            -0.009458970441859112,
            -0.017333266402560735,
            0.0019741073802161664,
            -0.005646281461241869,
            -0.01762116898876799
        ]
    );
}
