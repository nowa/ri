use serde::de::DeserializeOwned;
use serde_json::Value;

fn is_control_character(ch: char) -> bool {
    matches!(ch as u32, 0x00..=0x1f)
}

fn escape_control_character(ch: char) -> String {
    match ch {
        '\u{08}' => "\\b".to_owned(),
        '\u{0c}' => "\\f".to_owned(),
        '\n' => "\\n".to_owned(),
        '\r' => "\\r".to_owned(),
        '\t' => "\\t".to_owned(),
        _ => format!("\\u{:04x}", ch as u32),
    }
}

fn parse_hex_quad(chars: &[char], start: usize) -> Option<(String, u16)> {
    let digits: String = chars.iter().skip(start).take(4).collect();
    if digits.len() == 4 && digits.chars().all(|d| d.is_ascii_hexdigit()) {
        let value = u16::from_str_radix(&digits, 16).ok()?;
        Some((digits, value))
    } else {
        None
    }
}

fn is_high_surrogate(value: u16) -> bool {
    (0xd800..=0xdbff).contains(&value)
}

fn is_low_surrogate(value: u16) -> bool {
    (0xdc00..=0xdfff).contains(&value)
}

pub fn repair_json(json: &str) -> String {
    let mut repaired = String::new();
    let mut in_string = false;
    let chars: Vec<char> = json.chars().collect();
    let mut index = 0;

    while index < chars.len() {
        let ch = chars[index];

        if !in_string {
            repaired.push(ch);
            if ch == '"' {
                in_string = true;
            }
            index += 1;
            continue;
        }

        if ch == '"' {
            repaired.push(ch);
            in_string = false;
            index += 1;
            continue;
        }

        if ch == '\\' {
            let Some(next) = chars.get(index + 1).copied() else {
                repaired.push_str("\\\\");
                index += 1;
                continue;
            };

            if next == 'u' {
                if let Some((digits, value)) = parse_hex_quad(&chars, index + 2) {
                    if is_high_surrogate(value) {
                        if chars.get(index + 6) == Some(&'\\')
                            && chars.get(index + 7) == Some(&'u')
                            && let Some((low_digits, low_value)) = parse_hex_quad(&chars, index + 8)
                            && is_low_surrogate(low_value)
                        {
                            repaired.push_str("\\u");
                            repaired.push_str(&digits);
                            repaired.push_str("\\u");
                            repaired.push_str(&low_digits);
                            index += 12;
                            continue;
                        }

                        repaired.push_str("\\uFFFD");
                        index += 6;
                        continue;
                    }
                    if is_low_surrogate(value) {
                        repaired.push_str("\\uFFFD");
                        index += 6;
                        continue;
                    }

                    repaired.push_str("\\u");
                    repaired.push_str(&digits);
                    index += 6;
                    continue;
                }
            }

            if matches!(next, '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u') {
                repaired.push('\\');
                repaired.push(next);
                index += 2;
                continue;
            }

            repaired.push_str("\\\\");
            index += 1;
            continue;
        }

        if is_control_character(ch) {
            repaired.push_str(&escape_control_character(ch));
        } else {
            repaired.push(ch);
        }
        index += 1;
    }

    repaired
}

pub fn parse_json_with_repair<T: DeserializeOwned>(json: &str) -> serde_json::Result<T> {
    match serde_json::from_str(json) {
        Ok(value) => Ok(value),
        Err(first) => {
            let repaired = repair_json(json);
            if repaired != json {
                serde_json::from_str(&repaired)
            } else {
                Err(first)
            }
        }
    }
}

pub fn parse_streaming_json(partial_json: Option<&str>) -> Value {
    let Some(partial_json) = partial_json else {
        return Value::Object(Default::default());
    };
    if partial_json.trim().is_empty() {
        return Value::Object(Default::default());
    }

    if let Ok(value) = parse_json_with_repair(partial_json) {
        return value;
    }
    if let Some(candidate) = complete_partial_json(partial_json)
        && let Ok(value) = parse_json_with_repair(&candidate)
    {
        return value;
    }
    Value::Object(Default::default())
}

fn complete_partial_json(partial_json: &str) -> Option<String> {
    let repaired = repair_json(partial_json);
    let mut completed = String::with_capacity(repaired.len() + 8);
    let mut closers = Vec::<char>::new();
    let mut in_string = false;
    let mut escaped = false;

    for ch in repaired.chars() {
        completed.push(ch);
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => closers.push('}'),
            '[' => closers.push(']'),
            '}' | ']' => {
                completed.pop();
                trim_trailing_json_whitespace(&mut completed);
                if completed.ends_with(',') {
                    completed.pop();
                }
                completed.push(ch);
                if closers.pop() != Some(ch) {
                    return None;
                }
            }
            _ => {}
        }
    }

    if in_string {
        completed.push('"');
    }

    while let Some(closer) = closers.pop() {
        trim_trailing_json_whitespace(&mut completed);
        if completed.ends_with(':') {
            return None;
        }
        if completed.ends_with(',') {
            completed.pop();
        }
        completed.push(closer);
    }

    Some(completed)
}

fn trim_trailing_json_whitespace(value: &mut String) {
    while value.ends_with(char::is_whitespace) {
        value.pop();
    }
}

pub fn short_hash(input: &str) -> String {
    let mut h1: u32 = 0xdead_beef;
    let mut h2: u32 = 0x41c6_ce57;
    for unit in input.encode_utf16() {
        let ch = unit as u32;
        h1 = (h1 ^ ch).wrapping_mul(2_654_435_761);
        h2 = (h2 ^ ch).wrapping_mul(1_597_334_677);
    }
    h1 = ((h1 ^ (h1 >> 16)).wrapping_mul(2_246_822_507))
        ^ ((h2 ^ (h2 >> 13)).wrapping_mul(3_266_489_909));
    h2 = ((h2 ^ (h2 >> 16)).wrapping_mul(2_246_822_507))
        ^ ((h1 ^ (h1 >> 13)).wrapping_mul(3_266_489_909));
    format!("{}{}", to_base36(h2), to_base36(h1))
}

pub fn sanitize_surrogates(text: &str) -> String {
    text.to_owned()
}

fn to_base36(mut value: u32) -> String {
    if value == 0 {
        return "0".to_owned();
    }
    let mut out = Vec::new();
    while value > 0 {
        let digit = (value % 36) as u8;
        out.push(match digit {
            0..=9 => (b'0' + digit) as char,
            _ => (b'a' + digit - 10) as char,
        });
        value /= 36;
    }
    out.iter().rev().collect()
}
