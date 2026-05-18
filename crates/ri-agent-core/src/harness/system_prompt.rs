use super::Skill;

pub const DEFAULT_SYSTEM_PROMPT: &str =
    "You are a helpful AI assistant with access to tools and project context.";

pub fn system_prompt_with_context(base: &str, context: impl AsRef<str>) -> String {
    let context = context.as_ref();
    if context.trim().is_empty() {
        base.to_owned()
    } else {
        format!("{base}\n\n{context}")
    }
}

pub fn format_skills_for_system_prompt(skills: &[Skill]) -> String {
    let visible = skills
        .iter()
        .filter(|skill| !skill.disable_model_invocation)
        .collect::<Vec<_>>();
    if visible.is_empty() {
        return String::new();
    }

    let mut prompt = "The following skills provide specialized instructions for specific tasks.\n\
Read the full skill file when the task matches its description.\n\
When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.\n\n\
<available_skills>"
        .to_owned();

    for skill in visible {
        prompt.push_str(&format!(
            "\n  <skill>\n    <name>{}</name>\n    <description>{}</description>\n    <location>{}</location>\n  </skill>",
            escape_xml(&skill.name),
            escape_xml(&skill.description),
            escape_xml(&skill.file_path),
        ));
    }
    prompt.push_str("\n</available_skills>");
    prompt
}

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}
