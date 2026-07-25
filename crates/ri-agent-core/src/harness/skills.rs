use std::{
    fs,
    path::{Path, PathBuf},
};
use yaml_rust::{Yaml, YamlLoader};

const MAX_NAME_LENGTH: usize = 64;
const MAX_DESCRIPTION_LENGTH: usize = 1024;
const IGNORE_FILE_NAMES: &[&str] = &[".gitignore", ".ignore", ".fdignore"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub content: String,
    pub file_path: String,
    pub source: Option<String>,
    pub disable_model_invocation: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkillDiagnosticCode {
    FileInfoFailed,
    ListFailed,
    ReadFailed,
    ParseFailed,
    InvalidMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillDiagnostic {
    pub diagnostic_type: String,
    pub code: SkillDiagnosticCode,
    pub message: String,
    pub path: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourcedSkill<TSource> {
    pub skill: Skill,
    pub source: TSource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourcedSkillDiagnostic<TSource> {
    pub diagnostic: SkillDiagnostic,
    pub source: TSource,
}

pub fn format_skill_invocation(skill: &Skill, additional_instructions: Option<&str>) -> String {
    let skill_block = format!(
        "<skill name=\"{}\" location=\"{}\">\nReferences are relative to {}.\n\n{}\n</skill>",
        skill.name,
        skill.file_path,
        dirname_env_path(&skill.file_path),
        skill.content
    );
    match additional_instructions {
        Some(instructions) if !instructions.is_empty() => {
            format!("{skill_block}\n\n{instructions}")
        }
        _ => skill_block,
    }
}

pub fn load_skills<I, P>(dirs: I) -> (Vec<Skill>, Vec<SkillDiagnostic>)
where
    I: IntoIterator<Item = P>,
    P: AsRef<Path>,
{
    let mut skills = Vec::new();
    let mut diagnostics = Vec::new();
    for dir in dirs {
        let dir = dir.as_ref();
        match fs::metadata(dir) {
            Ok(metadata) if metadata.is_dir() => {
                let mut ignore = IgnoreRules::default();
                let (mut loaded, mut warnings) = load_skills_from_dir(dir, true, dir, &mut ignore);
                skills.append(&mut loaded);
                diagnostics.append(&mut warnings);
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => diagnostics.push(diagnostic(
                SkillDiagnosticCode::FileInfoFailed,
                error.to_string(),
                dir,
            )),
        }
    }
    (skills, diagnostics)
}

pub fn load_sourced_skills<I, P, TSource>(
    inputs: I,
) -> (
    Vec<SourcedSkill<TSource>>,
    Vec<SourcedSkillDiagnostic<TSource>>,
)
where
    I: IntoIterator<Item = (P, TSource)>,
    P: AsRef<Path>,
    TSource: Clone,
{
    let mut skills = Vec::new();
    let mut diagnostics = Vec::new();
    for (path, source) in inputs {
        let (loaded, warnings) = load_skills([path]);
        skills.extend(loaded.into_iter().map(|skill| SourcedSkill {
            skill,
            source: source.clone(),
        }));
        diagnostics.extend(
            warnings
                .into_iter()
                .map(|diagnostic| SourcedSkillDiagnostic {
                    diagnostic,
                    source: source.clone(),
                }),
        );
    }
    (skills, diagnostics)
}

pub(crate) fn dirname_env_path(path: &str) -> String {
    let normalized = path.trim_end_matches('/');
    // pi: slashIndex <= 0 ? "/" : ... — a slash-less path resolves to "/".
    match normalized.rfind('/') {
        None | Some(0) => "/".to_owned(),
        Some(index) => normalized[..index].to_owned(),
    }
}

pub(crate) fn basename_env_path(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_owned()
}

fn load_skills_from_dir(
    dir: &Path,
    include_root_files: bool,
    root_dir: &Path,
    ignore: &mut IgnoreRules,
) -> (Vec<Skill>, Vec<SkillDiagnostic>) {
    let mut skills = Vec::new();
    let mut diagnostics = Vec::new();
    let mut entries = match sorted_entries(dir) {
        Ok(entries) => entries,
        Err(error) => {
            diagnostics.push(diagnostic(
                SkillDiagnosticCode::ListFailed,
                error.to_string(),
                dir,
            ));
            return (skills, diagnostics);
        }
    };

    add_ignore_rules(dir, root_dir, ignore, &mut diagnostics);

    if let Some(skill_path) = entries
        .iter()
        .map(|entry| entry.path())
        .find(|path| path.file_name().and_then(|name| name.to_str()) == Some("SKILL.md"))
    {
        if fs::metadata(&skill_path)
            .map(|metadata| metadata.is_file())
            .unwrap_or(false)
            && !ignore.ignores(&relative_env_path(root_dir, &skill_path))
        {
            let (skill, mut warnings) = load_skill_from_file(&skill_path);
            if let Some(skill) = skill {
                skills.push(skill);
            }
            diagnostics.append(&mut warnings);
            return (skills, diagnostics);
        }
    }

    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') || name == "node_modules" {
            continue;
        }
        let path = entry.path();
        let Ok(metadata) = fs::metadata(&path) else {
            continue;
        };
        let relative_path = relative_env_path(root_dir, &path);
        let ignore_path = if metadata.is_dir() {
            format!("{relative_path}/")
        } else {
            relative_path
        };
        if ignore.ignores(&ignore_path) {
            continue;
        }
        if metadata.is_dir() {
            let (mut loaded, mut warnings) = load_skills_from_dir(&path, false, root_dir, ignore);
            skills.append(&mut loaded);
            diagnostics.append(&mut warnings);
        } else if include_root_files && metadata.is_file() && name.ends_with(".md") {
            let (skill, mut warnings) = load_skill_from_file(&path);
            if let Some(skill) = skill {
                skills.push(skill);
            }
            diagnostics.append(&mut warnings);
        }
    }

    (skills, diagnostics)
}

fn sorted_entries(dir: &Path) -> Result<Vec<fs::DirEntry>, std::io::Error> {
    let mut entries = fs::read_dir(dir)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    Ok(entries)
}

fn add_ignore_rules(
    dir: &Path,
    root_dir: &Path,
    ignore: &mut IgnoreRules,
    diagnostics: &mut Vec<SkillDiagnostic>,
) {
    let relative_dir = relative_env_path(root_dir, dir);
    let prefix = if relative_dir.is_empty() {
        String::new()
    } else {
        format!("{relative_dir}/")
    };

    for filename in IGNORE_FILE_NAMES {
        let path = dir.join(filename);
        let metadata = match fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                diagnostics.push(diagnostic(
                    SkillDiagnosticCode::FileInfoFailed,
                    error.to_string(),
                    &path,
                ));
                continue;
            }
        };
        if !metadata.is_file() {
            continue;
        }
        let content = match fs::read_to_string(&path) {
            Ok(content) => content,
            Err(error) => {
                diagnostics.push(diagnostic(
                    SkillDiagnosticCode::ReadFailed,
                    error.to_string(),
                    &path,
                ));
                continue;
            }
        };
        for line in content.lines() {
            if let Some(pattern) = prefix_ignore_pattern(line, &prefix) {
                ignore.add(pattern);
            }
        }
    }
}

fn prefix_ignore_pattern(line: &str, prefix: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() || (trimmed.starts_with('#') && !trimmed.starts_with("\\#")) {
        return None;
    }

    let mut pattern = line.trim_end().to_owned();
    let negated = if let Some(rest) = pattern.strip_prefix('!') {
        pattern = rest.to_owned();
        true
    } else {
        if let Some(rest) = pattern.strip_prefix("\\!") {
            pattern = rest.to_owned();
        } else if let Some(rest) = pattern.strip_prefix("\\#") {
            pattern = format!("#{rest}");
        }
        false
    };
    if let Some(rest) = pattern.strip_prefix('/') {
        pattern = rest.to_owned();
    }
    let prefixed = if prefix.is_empty() {
        pattern
    } else {
        format!("{prefix}{pattern}")
    };
    Some(if negated {
        format!("!{prefixed}")
    } else {
        prefixed
    })
}

#[derive(Default)]
struct IgnoreRules {
    rules: Vec<IgnoreRule>,
}

impl IgnoreRules {
    fn add(&mut self, pattern: String) {
        let (negated, pattern) = pattern
            .strip_prefix('!')
            .map(|pattern| (true, pattern.to_owned()))
            .unwrap_or((false, pattern));
        self.rules.push(IgnoreRule { pattern, negated });
    }

    fn ignores(&self, relative_path: &str) -> bool {
        let mut ignored = false;
        for rule in &self.rules {
            if rule.matches(relative_path.trim_start_matches("./")) {
                ignored = !rule.negated;
            }
        }
        ignored
    }
}

struct IgnoreRule {
    pattern: String,
    negated: bool,
}

impl IgnoreRule {
    fn matches(&self, relative_path: &str) -> bool {
        let raw_pattern = self.pattern.trim_start_matches("./");
        if raw_pattern.is_empty() {
            return false;
        }
        let directory_only = raw_pattern.ends_with('/');
        let pattern = raw_pattern.trim_end_matches('/');
        let relative_path = relative_path.trim_start_matches("./");
        let is_dir = relative_path.ends_with('/');
        let relative_path = relative_path.trim_end_matches('/');
        if relative_path.is_empty() {
            return false;
        }

        // A leading slash anchors the pattern to the root (gitignore).
        let (anchored_by_slash, pattern) = match pattern.strip_prefix('/') {
            Some(rest) => (true, rest),
            None => (false, pattern),
        };
        if pattern.is_empty() {
            return false;
        }
        let anchored = anchored_by_slash || pattern.contains('/');

        let mut pattern_components = pattern.split('/').collect::<Vec<_>>();
        if !anchored {
            // A pattern without a slash matches at any depth (gitignore
            // basename semantics): an implicit leading `**/`.
            pattern_components.insert(0, "**");
        }
        let path_components = relative_path.split('/').collect::<Vec<_>>();

        // A trailing `/**` matches everything inside the directory but not
        // the directory itself (gitignore); with a directory-only slash
        // (`a/**/`) the globstar keeps its zero-or-more meaning instead.
        if !directory_only && pattern_components.last() == Some(&"**") {
            let head = &pattern_components[..pattern_components.len() - 1];
            return match_components(head, &path_components).prefix;
        }

        let outcome = match_components(&pattern_components, &path_components);
        if outcome.prefix {
            // The pattern matched a leading directory of the path, so the
            // path lives inside an ignored directory.
            return true;
        }
        outcome.full && (!directory_only || is_dir)
    }
}

#[derive(Clone, Copy, Default)]
struct ComponentMatch {
    /// The pattern consumed the entire path.
    full: bool,
    /// The pattern matched a proper leading directory prefix of the path.
    prefix: bool,
}

fn match_components(pattern: &[&str], path: &[&str]) -> ComponentMatch {
    let Some((&head, rest)) = pattern.split_first() else {
        return ComponentMatch {
            full: path.is_empty(),
            prefix: !path.is_empty(),
        };
    };
    if head == "**" {
        // A standalone `**` spans zero or more path components (gitignore
        // globstar).
        let mut outcome = match_components(rest, path);
        if let Some((_, path_rest)) = path.split_first() {
            let skipped = match_components(pattern, path_rest);
            outcome.full |= skipped.full;
            outcome.prefix |= skipped.prefix;
        }
        return outcome;
    }
    let Some((&first, path_rest)) = path.split_first() else {
        return ComponentMatch::default();
    };
    if component_pattern_matches(head, first) {
        match_components(rest, path_rest)
    } else {
        ComponentMatch::default()
    }
}

fn has_glob(pattern: &str) -> bool {
    pattern.contains('*') || pattern.contains('?') || pattern.contains('[')
}

fn component_pattern_matches(pattern: &str, text: &str) -> bool {
    if !has_glob(pattern) {
        return pattern == text;
    }
    component_glob_matches(pattern.as_bytes(), text.as_bytes())
}

/// Byte-wise glob over one path component: `*`/`?` wildcards plus `[...]`
/// character classes with `!`/`^` negation and ranges (gitignore).
fn component_glob_matches(pattern: &[u8], text: &[u8]) -> bool {
    let Some((&first, rest)) = pattern.split_first() else {
        return text.is_empty();
    };
    match first {
        b'*' => {
            // Consecutive asterisks inside a component behave like a single
            // one (gitignore: "other consecutive asterisks are considered
            // regular asterisks").
            let mut rest = rest;
            while rest.first() == Some(&b'*') {
                rest = &rest[1..];
            }
            (0..=text.len()).any(|skip| component_glob_matches(rest, &text[skip..]))
        }
        b'?' => !text.is_empty() && component_glob_matches(rest, &text[1..]),
        b'[' => {
            // An unterminated class matches nothing (npm `ignore`).
            let Some(class) = CharacterClass::parse(pattern) else {
                return false;
            };
            !text.is_empty()
                && class.matches(text[0])
                && component_glob_matches(&pattern[class.length..], &text[1..])
        }
        _ => !text.is_empty() && text[0] == first && component_glob_matches(rest, &text[1..]),
    }
}

struct CharacterClass {
    length: usize,
    negated: bool,
    ranges: Vec<(u8, u8)>,
}

impl CharacterClass {
    /// Parse a `[...]` class at the start of `pattern` (gitignore rules:
    /// `!`/`^` negation, `a-z` ranges, a leading `]` is literal).
    fn parse(pattern: &[u8]) -> Option<Self> {
        let mut index = 1;
        let negated = matches!(pattern.get(index), Some(b'!' | b'^'));
        if negated {
            index += 1;
        }
        let mut ranges = Vec::new();
        let mut first = true;
        loop {
            let &byte = pattern.get(index)?;
            if byte == b']' && !first {
                return Some(Self {
                    length: index + 1,
                    negated,
                    ranges,
                });
            }
            first = false;
            index += 1;
            if pattern.get(index) == Some(&b'-')
                && pattern.get(index + 1).is_some_and(|&end| end != b']')
            {
                let end = pattern[index + 1];
                index += 2;
                // Out-of-order ranges are dropped (npm `ignore`
                // sanitizeRange).
                if byte <= end {
                    ranges.push((byte, end));
                }
            } else {
                ranges.push((byte, byte));
            }
        }
    }

    fn matches(&self, byte: u8) -> bool {
        let inside = self
            .ranges
            .iter()
            .any(|&(start, end)| start <= byte && byte <= end);
        inside != self.negated
    }
}

fn load_skill_from_file(path: &Path) -> (Option<Skill>, Vec<SkillDiagnostic>) {
    let mut diagnostics = Vec::new();
    let raw = match fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(error) => {
            diagnostics.push(diagnostic(
                SkillDiagnosticCode::ReadFailed,
                error.to_string(),
                path,
            ));
            return (None, diagnostics);
        }
    };

    let parsed = match parse_skill_frontmatter(&raw) {
        Ok(parsed) => parsed,
        Err(message) => {
            diagnostics.push(diagnostic(SkillDiagnosticCode::ParseFailed, message, path));
            return (None, diagnostics);
        }
    };

    let file_path = display_path(path);
    let skill_dir = dirname_env_path(&file_path);
    let parent_dir_name = basename_env_path(Path::new(&skill_dir));
    // pi: frontmatterName || parentDirName — an empty frontmatter name also
    // falls back to the directory name.
    let name = parsed
        .name
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| parent_dir_name.clone());
    let description = parsed.description;

    for message in validate_description(description.as_deref()) {
        diagnostics.push(diagnostic(
            SkillDiagnosticCode::InvalidMetadata,
            message,
            path,
        ));
    }
    for message in validate_name(&name, &parent_dir_name) {
        diagnostics.push(diagnostic(
            SkillDiagnosticCode::InvalidMetadata,
            message,
            path,
        ));
    }

    let Some(description) = description.filter(|description| !description.trim().is_empty()) else {
        return (None, diagnostics);
    };

    (
        Some(Skill {
            name,
            description,
            content: parsed.body,
            file_path,
            source: None,
            disable_model_invocation: parsed.disable_model_invocation,
        }),
        diagnostics,
    )
}

fn validate_name(name: &str, parent_dir_name: &str) -> Vec<String> {
    let mut errors = Vec::new();
    if name != parent_dir_name {
        errors.push(format!(
            "name \"{name}\" does not match parent directory \"{parent_dir_name}\""
        ));
    }
    if name.chars().count() > MAX_NAME_LENGTH {
        errors.push(format!(
            "name exceeds {MAX_NAME_LENGTH} characters ({})",
            name.chars().count()
        ));
    }
    if !name
        .chars()
        .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-')
    {
        errors.push(
            "name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)"
                .to_owned(),
        );
    }
    if name.starts_with('-') || name.ends_with('-') {
        errors.push("name must not start or end with a hyphen".to_owned());
    }
    if name.contains("--") {
        errors.push("name must not contain consecutive hyphens".to_owned());
    }
    errors
}

fn validate_description(description: Option<&str>) -> Vec<String> {
    let mut errors = Vec::new();
    match description {
        Some(description) if !description.trim().is_empty() => {
            if description.chars().count() > MAX_DESCRIPTION_LENGTH {
                errors.push(format!(
                    "description exceeds {MAX_DESCRIPTION_LENGTH} characters ({})",
                    description.chars().count()
                ));
            }
        }
        _ => errors.push("description is required".to_owned()),
    }
    errors
}

#[derive(Debug)]
struct ParsedSkill {
    name: Option<String>,
    description: Option<String>,
    disable_model_invocation: bool,
    body: String,
}

fn parse_skill_frontmatter(content: &str) -> Result<ParsedSkill, String> {
    let normalized = content.replace("\r\n", "\n").replace('\r', "\n");
    if !normalized.starts_with("---") {
        return Ok(ParsedSkill {
            name: None,
            description: None,
            disable_model_invocation: false,
            body: normalized,
        });
    }
    let Some(end_index) = normalized[3..].find("\n---").map(|index| index + 3) else {
        return Ok(ParsedSkill {
            name: None,
            description: None,
            disable_model_invocation: false,
            body: normalized,
        });
    };

    let yaml = &normalized[4..end_index];
    let docs = YamlLoader::load_from_str(yaml).map_err(|error| error.to_string())?;
    let value = docs.first();
    let body = normalized[end_index + 4..].trim().to_owned();
    Ok(ParsedSkill {
        name: yaml_string_field(value, "name"),
        description: yaml_string_field(value, "description"),
        disable_model_invocation: yaml_bool_field(value, "disable-model-invocation")
            .unwrap_or(false),
        body,
    })
}

fn yaml_string_field(value: Option<&Yaml>, key: &str) -> Option<String> {
    let Some(Yaml::Hash(mapping)) = value else {
        return None;
    };
    let key = Yaml::String(key.to_owned());
    match mapping.get(&key) {
        Some(Yaml::String(value)) => Some(value.clone()),
        _ => None,
    }
}

fn yaml_bool_field(value: Option<&Yaml>, key: &str) -> Option<bool> {
    let Some(Yaml::Hash(mapping)) = value else {
        return None;
    };
    let key = Yaml::String(key.to_owned());
    match mapping.get(&key) {
        Some(Yaml::Boolean(value)) => Some(*value),
        _ => None,
    }
}

fn display_path(path: &Path) -> String {
    path_to_unix(path)
}

fn diagnostic(
    code: SkillDiagnosticCode,
    message: impl Into<String>,
    path: &Path,
) -> SkillDiagnostic {
    SkillDiagnostic {
        diagnostic_type: "warning".to_owned(),
        code,
        message: message.into(),
        path: display_path(path),
    }
}

fn relative_env_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .ok()
        .map(path_to_unix)
        .unwrap_or_else(|| display_path(path))
        .trim_start_matches('/')
        .to_owned()
}

fn path_to_unix(path: &Path) -> String {
    path.components()
        .collect::<PathBuf>()
        .to_string_lossy()
        .replace('\\', "/")
}
