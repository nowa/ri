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
    let Some(index) = normalized.rfind('/') else {
        return ".".to_owned();
    };
    if index == 0 {
        "/".to_owned()
    } else {
        normalized[..index].to_owned()
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

        if directory_only {
            return matches_directory_pattern(pattern, relative_path, is_dir);
        }

        if pattern.contains('/') {
            if path_pattern_matches(pattern, relative_path) {
                return true;
            }
            !has_glob(pattern) && relative_path.starts_with(&format!("{pattern}/"))
        } else {
            relative_path
                .split('/')
                .any(|component| component_pattern_matches(pattern, component))
                || relative_path.starts_with(&format!("{pattern}/"))
        }
    }
}

fn matches_directory_pattern(pattern: &str, relative_path: &str, is_dir: bool) -> bool {
    if pattern.contains('/') {
        if path_pattern_matches(pattern, relative_path) {
            return is_dir || relative_path.starts_with(&format!("{pattern}/"));
        }
        return !has_glob(pattern) && relative_path.starts_with(&format!("{pattern}/"));
    }

    let mut components = relative_path.split('/').peekable();
    while let Some(component) = components.next() {
        if component_pattern_matches(pattern, component) && (is_dir || components.peek().is_some())
        {
            return true;
        }
    }
    false
}

fn path_pattern_matches(pattern: &str, relative_path: &str) -> bool {
    let pattern_components = pattern.split('/').collect::<Vec<_>>();
    let path_components = relative_path.split('/').collect::<Vec<_>>();
    if pattern_components.len() != path_components.len() {
        return false;
    }
    pattern_components
        .iter()
        .zip(path_components)
        .all(|(pattern, component)| component_pattern_matches(pattern, component))
}

fn has_glob(pattern: &str) -> bool {
    pattern.contains('*') || pattern.contains('?')
}

fn component_pattern_matches(pattern: &str, text: &str) -> bool {
    if !has_glob(pattern) {
        return pattern == text;
    }
    let pattern = pattern.as_bytes();
    let text = text.as_bytes();
    let (mut p, mut t) = (0, 0);
    let mut star = None;
    let mut match_after_star = 0;

    while t < text.len() {
        if p < pattern.len() && (pattern[p] == text[t] || pattern[p] == b'?') {
            p += 1;
            t += 1;
        } else if p < pattern.len() && pattern[p] == b'*' {
            star = Some(p);
            match_after_star = t;
            p += 1;
        } else if let Some(star_index) = star {
            p = star_index + 1;
            match_after_star += 1;
            t = match_after_star;
        } else {
            return false;
        }
    }

    while p < pattern.len() && pattern[p] == b'*' {
        p += 1;
    }
    p == pattern.len()
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
    let name = parsed.name.unwrap_or_else(|| parent_dir_name.clone());
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
