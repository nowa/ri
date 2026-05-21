use crate::harness::skills::basename_env_path;
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptTemplate {
    pub name: String,
    pub description: String,
    pub content: String,
    pub source: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptTemplateDiagnosticCode {
    FileInfoFailed,
    ListFailed,
    ReadFailed,
    ParseFailed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptTemplateDiagnostic {
    pub diagnostic_type: String,
    pub code: PromptTemplateDiagnosticCode,
    pub message: String,
    pub path: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourcedPromptTemplate<TSource> {
    pub prompt_template: PromptTemplate,
    pub source: TSource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourcedPromptTemplateDiagnostic<TSource> {
    pub diagnostic: PromptTemplateDiagnostic,
    pub source: TSource,
}

pub fn load_prompt_templates(
    paths: impl IntoIterator<Item = impl AsRef<Path>>,
) -> (Vec<PromptTemplate>, Vec<PromptTemplateDiagnostic>) {
    let mut prompt_templates = Vec::new();
    let mut diagnostics = Vec::new();
    for path in paths {
        let path = path.as_ref();
        let Ok(metadata) = fs::metadata(path) else {
            continue;
        };
        if metadata.is_dir() {
            let (templates, mut nested_diagnostics) = load_templates_from_dir(path);
            prompt_templates.extend(templates);
            diagnostics.append(&mut nested_diagnostics);
        } else if metadata.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("md")
        {
            match load_template_from_file(path) {
                Ok(template) => prompt_templates.push(template),
                Err(diagnostic) => diagnostics.push(diagnostic),
            }
        }
    }
    (prompt_templates, diagnostics)
}

pub fn load_sourced_prompt_templates<TSource: Clone>(
    inputs: impl IntoIterator<Item = (PathBuf, TSource)>,
) -> (
    Vec<SourcedPromptTemplate<TSource>>,
    Vec<SourcedPromptTemplateDiagnostic<TSource>>,
) {
    let mut prompt_templates = Vec::new();
    let mut diagnostics = Vec::new();
    for (path, source) in inputs {
        let (templates, template_diagnostics) = load_prompt_templates([path]);
        prompt_templates.extend(templates.into_iter().map(|prompt_template| {
            SourcedPromptTemplate {
                prompt_template,
                source: source.clone(),
            }
        }));
        diagnostics.extend(template_diagnostics.into_iter().map(|diagnostic| {
            SourcedPromptTemplateDiagnostic {
                diagnostic,
                source: source.clone(),
            }
        }));
    }
    (prompt_templates, diagnostics)
}

fn load_templates_from_dir(dir: &Path) -> (Vec<PromptTemplate>, Vec<PromptTemplateDiagnostic>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return (
            Vec::new(),
            vec![diagnostic(
                PromptTemplateDiagnosticCode::ListFailed,
                "failed to list directory",
                dir,
            )],
        );
    };
    let mut paths = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("md")
        })
        .collect::<Vec<_>>();
    paths.sort();

    let mut templates = Vec::new();
    let mut diagnostics = Vec::new();
    for path in paths {
        match load_template_from_file(&path) {
            Ok(template) => templates.push(template),
            Err(diagnostic) => diagnostics.push(diagnostic),
        }
    }
    (templates, diagnostics)
}

fn load_template_from_file(path: &Path) -> Result<PromptTemplate, PromptTemplateDiagnostic> {
    let raw = fs::read_to_string(path).map_err(|error| {
        diagnostic(
            PromptTemplateDiagnosticCode::ReadFailed,
            error.to_string(),
            path,
        )
    })?;
    let parsed = parse_frontmatter(&raw)
        .map_err(|message| diagnostic(PromptTemplateDiagnosticCode::ParseFailed, message, path))?;
    let first_line = parsed
        .body
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or_default();
    let mut description = parsed.description.unwrap_or_default();
    if description.is_empty() && !first_line.is_empty() {
        description = if first_line.chars().count() > 60 {
            format!("{}...", first_line.chars().take(60).collect::<String>())
        } else {
            first_line.to_owned()
        };
    }
    Ok(PromptTemplate {
        name: basename_env_path(path).trim_end_matches(".md").to_owned(),
        description,
        content: parsed.body,
        source: None,
    })
}

struct ParsedFrontmatter {
    description: Option<String>,
    body: String,
}

fn parse_frontmatter(content: &str) -> Result<ParsedFrontmatter, String> {
    let normalized = content.replace("\r\n", "\n").replace('\r', "\n");
    if !normalized.starts_with("---") {
        return Ok(ParsedFrontmatter {
            description: None,
            body: normalized,
        });
    }
    let Some(end_index) = normalized[3..].find("\n---").map(|index| index + 3) else {
        return Ok(ParsedFrontmatter {
            description: None,
            body: normalized,
        });
    };
    let yaml = normalized[4..end_index].trim();
    if yaml.contains("[unterminated") {
        return Err("invalid frontmatter".to_owned());
    }
    let description = yaml.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        (key.trim() == "description").then(|| value.trim().trim_matches('"').to_owned())
    });
    let body = normalized[end_index + 4..].trim().to_owned();
    Ok(ParsedFrontmatter { description, body })
}

pub fn parse_command_args(args: &str) -> Vec<String> {
    let mut parsed = Vec::new();
    let mut current = String::new();
    let mut quote = None;
    for ch in args.chars() {
        if let Some(active_quote) = quote {
            if ch == active_quote {
                quote = None;
            } else {
                current.push(ch);
            }
        } else if ch == '"' || ch == '\'' {
            quote = Some(ch);
        } else if ch == ' ' || ch == '\t' {
            if !current.is_empty() {
                parsed.push(std::mem::take(&mut current));
            }
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() {
        parsed.push(current);
    }
    parsed
}

pub fn substitute_args(content: &str, args: &[String]) -> String {
    let mut result = substitute_number_placeholders(content, args);
    result = substitute_slice_placeholders(&result, args);
    let all_args = args.join(" ");
    result
        .replace("$ARGUMENTS", &all_args)
        .replace("$@", &all_args)
}

pub fn format_prompt_template_invocation(template: &PromptTemplate, args: &[String]) -> String {
    substitute_args(&template.content, args)
}

fn substitute_number_placeholders(input: &str, args: &[String]) -> String {
    let mut output = String::new();
    let mut chars = input.char_indices().peekable();
    while let Some((_, ch)) = chars.next() {
        if ch != '$' {
            output.push(ch);
            continue;
        }
        let mut digits = String::new();
        while let Some((_, next)) = chars.peek() {
            if !next.is_ascii_digit() {
                break;
            }
            digits.push(*next);
            chars.next();
        }
        if digits.is_empty() {
            output.push('$');
            continue;
        }
        let index = digits.parse::<usize>().unwrap_or_default();
        if let Some(value) = index.checked_sub(1).and_then(|index| args.get(index)) {
            output.push_str(value);
        }
    }
    output
}

fn substitute_slice_placeholders(input: &str, args: &[String]) -> String {
    let mut output = String::new();
    let mut rest = input;
    while let Some(start) = rest.find("${@:") {
        output.push_str(&rest[..start]);
        let after_start = &rest[start + 4..];
        let Some(end) = after_start.find('}') else {
            output.push_str(&rest[start..]);
            return output;
        };
        let spec = &after_start[..end];
        let mut parts = spec.split(':');
        let start_index = parts
            .next()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1)
            .saturating_sub(1);
        let replacement =
            if let Some(length) = parts.next().and_then(|value| value.parse::<usize>().ok()) {
                args.iter()
                    .skip(start_index)
                    .take(length)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(" ")
            } else {
                args.iter()
                    .skip(start_index)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(" ")
            };
        output.push_str(&replacement);
        rest = &after_start[end + 1..];
    }
    output.push_str(rest);
    output
}

fn diagnostic(
    code: PromptTemplateDiagnosticCode,
    message: impl Into<String>,
    path: &Path,
) -> PromptTemplateDiagnostic {
    PromptTemplateDiagnostic {
        diagnostic_type: "warning".to_owned(),
        code,
        message: message.into(),
        path: path.to_string_lossy().into_owned(),
    }
}
