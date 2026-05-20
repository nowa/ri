use ri_agent_core::*;

fn expected_buffer_tail(content: &str, max_bytes: usize) -> &str {
    let bytes = content.as_bytes();
    if bytes.len() <= max_bytes {
        return content;
    }
    let mut start = bytes.len() - max_bytes;
    while start < bytes.len() && (bytes[start] & 0xc0) == 0x80 {
        start += 1;
    }
    std::str::from_utf8(&bytes[start..]).expect("valid tail")
}

fn sampled_byte_limits(content: &str) -> Vec<usize> {
    let total_bytes = content.len();
    let candidates = [
        0,
        1,
        2,
        3,
        4,
        5,
        8,
        total_bytes.saturating_div(2).saturating_sub(1),
        total_bytes / 2,
        total_bytes / 2 + 1,
        total_bytes.saturating_sub(8),
        total_bytes.saturating_sub(5),
        total_bytes.saturating_sub(4),
        total_bytes.saturating_sub(3),
        total_bytes.saturating_sub(2),
        total_bytes.saturating_sub(1),
        total_bytes,
        total_bytes + 1,
        total_bytes + 4,
    ];
    let mut values = candidates.to_vec();
    values.sort_unstable();
    values.dedup();
    values
}

fn assert_matches_buffer_tail(content: &str, byte_limits: &[usize]) {
    for &max_bytes in byte_limits {
        let result = truncate_tail(
            content,
            TruncationOptions {
                max_bytes,
                max_lines: 10,
            },
        );
        let expected = expected_buffer_tail(content, max_bytes);
        assert_eq!(
            result.content, expected,
            "tail mismatch input={content:?} max_bytes={max_bytes}"
        );
        assert!(
            result.output_bytes <= max_bytes,
            "output exceeded byte limit input={content:?} max_bytes={max_bytes} output_bytes={}",
            result.output_bytes
        );
    }
}

#[test]
fn truncate_counts_utf8_bytes_without_buffer_dependencies() {
    let content = "aé🙂\nb";
    let result = truncate_head(
        content,
        TruncationOptions {
            max_bytes: 100,
            max_lines: 10,
        },
    );

    assert!(!result.truncated);
    assert_eq!(result.total_bytes, 9);
    assert_eq!(result.output_bytes, 9);
    assert_eq!(result.content, content);
}

#[test]
fn truncate_head_uses_complete_utf8_lines_and_reports_first_line_overflow() {
    let result = truncate_head(
        "éé\nabc",
        TruncationOptions {
            max_bytes: 4,
            max_lines: 10,
        },
    );
    assert_eq!(result.content, "éé");
    assert!(result.truncated);
    assert_eq!(result.truncated_by, Some(TruncatedBy::Bytes));
    assert_eq!(result.output_bytes, 4);
    assert!(!result.first_line_exceeds_limit);

    let first_line_overflow = truncate_head(
        "éé\nabc",
        TruncationOptions {
            max_bytes: 3,
            max_lines: 10,
        },
    );
    assert_eq!(first_line_overflow.content, "");
    assert!(first_line_overflow.truncated);
    assert_eq!(first_line_overflow.truncated_by, Some(TruncatedBy::Bytes));
    assert!(first_line_overflow.first_line_exceeds_limit);
}

#[test]
fn truncate_tail_keeps_utf8_suffix_and_marks_partial_last_line() {
    let result = truncate_tail(
        "aé🙂b",
        TruncationOptions {
            max_bytes: 5,
            max_lines: 10,
        },
    );
    assert_eq!(result.content, "🙂b");
    assert!(result.truncated);
    assert_eq!(result.truncated_by, Some(TruncatedBy::Bytes));
    assert!(result.last_line_partial);
    assert_eq!(result.output_bytes, 5);
}

#[test]
fn truncate_tail_drops_oversized_trailing_character_when_it_cannot_fit() {
    let oversized = truncate_tail(
        "abc🙂",
        TruncationOptions {
            max_bytes: 3,
            max_lines: 10,
        },
    );
    assert_eq!(oversized.content, "");
    assert!(oversized.truncated);
    assert_eq!(oversized.truncated_by, Some(TruncatedBy::Bytes));
    assert!(oversized.last_line_partial);
    assert_eq!(oversized.output_bytes, 0);
}

#[test]
fn truncate_tail_matches_buffer_semantics_for_multibyte_edges() {
    let inputs = ["", "a", "é", "aéb", "中🙂", "👩‍💻", "abc🙂", "🙂🙂b"];
    for input in inputs {
        assert_matches_buffer_tail(input, &sampled_byte_limits(input));
    }
}

#[test]
fn truncate_tail_matches_buffer_semantics_across_deterministic_fuzz_cases() {
    let alphabet = [
        "a", "\u{7f}", "\u{80}", "é", "\u{7ff}", "\u{800}", "中", "\u{d7ff}", "🙂", "\u{e000}",
        "\u{ffff}",
    ];

    fn check(prefix: String, depth: usize, alphabet: &[&str]) {
        let limits = sampled_byte_limits(&prefix);
        assert_matches_buffer_tail(&prefix, &limits);
        if depth == 0 {
            return;
        }
        for character in alphabet {
            check(format!("{prefix}{character}"), depth - 1, alphabet);
        }
    }
    check(String::new(), 2, &alphabet);

    let mut seed = 0x1234_5678_u32;
    for _ in 0..250 {
        let mut input = String::new();
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let length = (seed % 40) as usize;
        for _ in 0..length {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            input.push_str(alphabet[(seed as usize) % alphabet.len()]);
        }
        let limits = sampled_byte_limits(&input);
        assert_matches_buffer_tail(&input, &limits);
    }
}

#[test]
fn truncate_line_and_format_size_match_harness_helpers() {
    assert_eq!(format_size(42), "42B");
    assert_eq!(format_size(1536), "1.5KB");
    assert_eq!(format_size(2 * 1024 * 1024), "2.0MB");

    let short = truncate_line("short", GREP_MAX_LINE_LENGTH);
    assert_eq!(short.text, "short");
    assert!(!short.was_truncated);

    let long = truncate_line("abcdef", 3);
    assert_eq!(long.text, "abc... [truncated]");
    assert!(long.was_truncated);
}
