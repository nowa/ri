use ri_agent_core::*;
use std::{fs, path::PathBuf};

#[cfg(unix)]
use std::os::unix::fs as unix_fs;

fn temp_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ri-resources-test-{}", uuidv7()));
    fs::create_dir_all(&dir).expect("temp dir");
    dir
}

#[test]
fn formats_skill_and_prompt_template_invocations() {
    let skill = Skill {
        name: "inspect".to_owned(),
        description: "Inspect things".to_owned(),
        content: "Use inspection tools.".to_owned(),
        file_path: "/project/.pi/skills/inspect/SKILL.md".to_owned(),
        source: None,
        disable_model_invocation: false,
    };

    assert_eq!(
        format_skill_invocation(&skill, Some("Check errors.")),
        "<skill name=\"inspect\" location=\"/project/.pi/skills/inspect/SKILL.md\">\nReferences are relative to /project/.pi/skills/inspect.\n\nUse inspection tools.\n</skill>\n\nCheck errors."
    );

    let template = PromptTemplate {
        name: "review".to_owned(),
        description: String::new(),
        content: "Review $1 with $ARGUMENTS".to_owned(),
        source: None,
    };
    assert_eq!(
        format_prompt_template_invocation(&template, &["a.ts".to_owned(), "care".to_owned()]),
        "Review a.ts with a.ts care"
    );
}

#[test]
fn formats_visible_skills_for_system_prompt() {
    let visible = Skill {
        name: "visible".to_owned(),
        description: "Use <this> & that".to_owned(),
        content: "visible content".to_owned(),
        file_path: "/skills/visible/SKILL.md".to_owned(),
        source: None,
        disable_model_invocation: false,
    };
    let disabled = Skill {
        name: "hidden".to_owned(),
        description: "Hidden".to_owned(),
        content: "hidden content".to_owned(),
        file_path: "/skills/hidden/SKILL.md".to_owned(),
        source: None,
        disable_model_invocation: true,
    };
    let second = Skill {
        name: "second".to_owned(),
        description: "Second skill".to_owned(),
        content: "second content".to_owned(),
        file_path: "/skills/second/SKILL.md".to_owned(),
        source: None,
        disable_model_invocation: false,
    };

    assert_eq!(
        format_skills_for_system_prompt(&[visible, disabled.clone(), second]),
        "The following skills provide specialized instructions for specific tasks.\nRead the full skill file when the task matches its description.\nWhen a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.\n\n<available_skills>\n  <skill>\n    <name>visible</name>\n    <description>Use &lt;this&gt; &amp; that</description>\n    <location>/skills/visible/SKILL.md</location>\n  </skill>\n  <skill>\n    <name>second</name>\n    <description>Second skill</description>\n    <location>/skills/second/SKILL.md</location>\n  </skill>\n</available_skills>"
    );
    assert_eq!(format_skills_for_system_prompt(&[disabled]), "");

    let escaped = format_skills_for_system_prompt(&[Skill {
        name: "a&b".to_owned(),
        description: "Quote \"double\" and 'single'".to_owned(),
        content: "content".to_owned(),
        file_path: "/skills/<bad>&\"quote\"/SKILL.md".to_owned(),
        source: None,
        disable_model_invocation: false,
    }]);
    assert!(escaped.contains("<name>a&amp;b</name>"));
    assert!(
        escaped
            .contains("<description>Quote &quot;double&quot; and &apos;single&apos;</description>")
    );
    assert!(
        escaped.contains("<location>/skills/&lt;bad&gt;&amp;&quot;quote&quot;/SKILL.md</location>")
    );
}

#[test]
fn loads_skills_from_skill_files_and_root_markdown() {
    let root = temp_dir();
    fs::create_dir_all(root.join(".agents/skills/example")).expect("example");
    fs::write(
        root.join(".agents/skills/example/SKILL.md"),
        "---\nname: example\ndescription: Example skill\ndisable-model-invocation: true\n---\nUse this skill.\n",
    )
    .expect("skill");

    let (skills, diagnostics) = load_skills([root.join(".agents/skills")]);
    assert!(diagnostics.is_empty());
    assert_eq!(
        skills,
        vec![Skill {
            name: "example".to_owned(),
            description: "Example skill".to_owned(),
            content: "Use this skill.".to_owned(),
            file_path: root
                .join(".agents/skills/example/SKILL.md")
                .to_string_lossy()
                .to_string(),
            source: None,
            disable_model_invocation: true,
        }]
    );

    let root_files = temp_dir();
    fs::create_dir_all(root_files.join("skills/nested")).expect("nested");
    fs::write(
        root_files.join("skills/root.md"),
        "---\ndescription: Root skill\n---\nRoot content",
    )
    .expect("root md");
    fs::write(
        root_files.join("skills/nested/ignored.md"),
        "---\ndescription: Ignored\n---\nIgnored content",
    )
    .expect("ignored");

    let (skills, diagnostics) = load_skills([root_files.join("skills")]);
    assert!(diagnostics.is_empty());
    assert_eq!(skills.len(), 1);
    assert_eq!(skills[0].name, "skills");
    assert_eq!(skills[0].description, "Root skill");
    assert_eq!(skills[0].content, "Root content");
}

#[cfg(unix)]
#[test]
fn loads_skills_through_symlinked_directories() {
    let root = temp_dir();
    fs::create_dir_all(root.join("actual/example")).expect("actual");
    fs::write(
        root.join("actual/example/SKILL.md"),
        "---\nname: example\ndescription: Example skill\n---\nUse this skill.",
    )
    .expect("skill");
    unix_fs::symlink(root.join("actual"), root.join("skills-link")).expect("symlink");

    let (skills, diagnostics) = load_skills([root.join("skills-link")]);
    assert!(diagnostics.is_empty());
    assert_eq!(skills.len(), 1);
    assert_eq!(skills[0].name, "example");
    assert_eq!(
        skills[0].file_path,
        root.join("skills-link/example/SKILL.md")
            .to_string_lossy()
            .to_string()
    );
}

#[test]
fn load_skills_honors_ignore_files() {
    let root = temp_dir();
    let skills_dir = root.join("skills");
    fs::create_dir_all(skills_dir.join("ignored-skill")).expect("ignored skill");
    fs::create_dir_all(skills_dir.join("nested/local-ignored")).expect("local ignored");
    fs::create_dir_all(skills_dir.join("nested/visible")).expect("visible");

    fs::write(
        skills_dir.join(".gitignore"),
        "ignored-skill/\nroot-ignored.md\n*.tmp.md\n!keep.tmp.md\n",
    )
    .expect("gitignore");
    fs::write(skills_dir.join("nested/.ignore"), "local-ignored/\n").expect("nested ignore");
    fs::write(
        skills_dir.join("ignored-skill/SKILL.md"),
        "---\nname: ignored-skill\ndescription: ignored skill\n---\nIgnored",
    )
    .expect("ignored skill file");
    fs::write(
        skills_dir.join("nested/local-ignored/SKILL.md"),
        "---\nname: local-ignored\ndescription: local ignored\n---\nIgnored",
    )
    .expect("local ignored file");
    fs::write(
        skills_dir.join("nested/visible/SKILL.md"),
        "---\nname: visible\ndescription: visible nested\n---\nVisible",
    )
    .expect("visible file");
    fs::write(
        skills_dir.join("root-ignored.md"),
        "---\ndescription: ignored root\n---\nIgnored root",
    )
    .expect("ignored root");
    fs::write(
        skills_dir.join("drop.tmp.md"),
        "---\ndescription: ignored wildcard\n---\nIgnored wildcard",
    )
    .expect("ignored wildcard");
    fs::write(
        skills_dir.join("keep.tmp.md"),
        "---\ndescription: kept negated wildcard\n---\nKept wildcard",
    )
    .expect("kept wildcard");
    fs::write(
        skills_dir.join("root-visible.md"),
        "---\ndescription: visible root\n---\nVisible root",
    )
    .expect("visible root");

    let (skills, diagnostics) = load_skills([skills_dir]);
    assert!(diagnostics.is_empty());
    let mut descriptions = skills
        .iter()
        .map(|skill| skill.description.as_str())
        .collect::<Vec<_>>();
    descriptions.sort_unstable();
    assert_eq!(
        descriptions,
        vec!["kept negated wildcard", "visible nested", "visible root"]
    );
}

#[test]
fn sourced_skills_preserve_source_and_attach_diagnostics() {
    let root = temp_dir();
    fs::create_dir_all(root.join("user/example")).expect("example");
    fs::write(
        root.join("user/example/SKILL.md"),
        "---\nname: example\ndescription: Example skill\n---\nUse this skill.",
    )
    .expect("skill");

    let (skills, diagnostics) = load_sourced_skills([(root.join("user"), "user".to_owned())]);
    assert!(diagnostics.is_empty());
    assert_eq!(skills.len(), 1);
    assert_eq!(skills[0].source, "user");
    assert_eq!(skills[0].skill.name, "example");

    fs::create_dir_all(root.join("broken/broken")).expect("broken");
    fs::write(
        root.join("broken/broken/SKILL.md"),
        "---\nname: broken\n---\nMissing description.",
    )
    .expect("broken skill");
    let (skills, diagnostics) = load_sourced_skills([(root.join("broken"), "user".to_owned())]);
    assert!(skills.is_empty());
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].source, "user");
    assert_eq!(diagnostics[0].diagnostic.diagnostic_type, "warning");
    assert_eq!(
        diagnostics[0].diagnostic.path,
        root.join("broken/broken/SKILL.md").to_string_lossy()
    );
    assert_eq!(
        diagnostics[0].diagnostic.code,
        SkillDiagnosticCode::InvalidMetadata
    );
    assert_eq!(diagnostics[0].diagnostic.message, "description is required");
}

#[test]
fn load_skills_reports_pi_metadata_validation_warnings_without_dropping_skill() {
    let root = temp_dir();
    fs::create_dir_all(root.join("skills/bad_name")).expect("skill dir");
    let long_description = "d".repeat(1025);
    fs::write(
        root.join("skills/bad_name/SKILL.md"),
        format!("---\nname: Bad_Name\ndescription: {long_description}\n---\nUse it."),
    )
    .expect("skill");

    let (skills, diagnostics) = load_skills([root.join("skills")]);

    assert_eq!(skills.len(), 1);
    assert_eq!(skills[0].name, "Bad_Name");
    assert_eq!(skills[0].description, long_description);
    assert_eq!(
        diagnostics
            .iter()
            .map(|diagnostic| (
                diagnostic.diagnostic_type.as_str(),
                &diagnostic.code,
                diagnostic.message.as_str()
            ))
            .collect::<Vec<_>>(),
        vec![
            (
                "warning",
                &SkillDiagnosticCode::InvalidMetadata,
                "description exceeds 1024 characters (1025)"
            ),
            (
                "warning",
                &SkillDiagnosticCode::InvalidMetadata,
                "name \"Bad_Name\" does not match parent directory \"bad_name\""
            ),
            (
                "warning",
                &SkillDiagnosticCode::InvalidMetadata,
                "name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)"
            ),
        ]
    );
}

#[test]
fn loads_prompt_templates_non_recursively_from_dirs_and_files() {
    let root = temp_dir();
    fs::create_dir_all(root.join("a/nested")).expect("a");
    fs::create_dir_all(root.join("b")).expect("b");
    fs::write(
        root.join("a/one.md"),
        "---\ndescription: One template\n---\nHello $1",
    )
    .expect("one");
    fs::write(root.join("a/nested/ignored.md"), "Ignored").expect("ignored");
    fs::write(root.join("b/two.md"), "First line description\nBody").expect("two");

    let (templates, diagnostics) = load_prompt_templates([root.join("a"), root.join("b")]);
    assert!(diagnostics.is_empty());
    assert_eq!(
        templates,
        vec![
            PromptTemplate {
                name: "one".to_owned(),
                description: "One template".to_owned(),
                content: "Hello $1".to_owned(),
                source: None,
            },
            PromptTemplate {
                name: "two".to_owned(),
                description: "First line description".to_owned(),
                content: "First line description\nBody".to_owned(),
                source: None,
            },
        ]
    );

    let (explicit, _) = load_prompt_templates([root.join("a/one.md")]);
    assert_eq!(explicit[0].name, "one");
}

#[cfg(unix)]
#[test]
fn loads_prompt_templates_from_symlinked_markdown_files() {
    let root = temp_dir();
    fs::write(
        root.join("target.md"),
        "---\ndescription: Target\n---\nTarget body",
    )
    .expect("target");
    unix_fs::symlink(root.join("target.md"), root.join("link.md")).expect("symlink");

    let (templates, diagnostics) =
        load_prompt_templates([root.join("target.md"), root.join("link.md")]);

    assert!(diagnostics.is_empty());
    assert_eq!(
        templates,
        vec![
            PromptTemplate {
                name: "target".to_owned(),
                description: "Target".to_owned(),
                content: "Target body".to_owned(),
                source: None,
            },
            PromptTemplate {
                name: "link".to_owned(),
                description: "Target".to_owned(),
                content: "Target body".to_owned(),
                source: None,
            },
        ]
    );
}

#[test]
fn sourced_prompt_templates_preserve_source_and_attach_diagnostics() {
    let root = temp_dir();
    fs::create_dir_all(root.join("prompts")).expect("prompts");
    fs::write(
        root.join("prompts/example.md"),
        "---\ndescription: Example\n---\nExample body",
    )
    .expect("example");
    let (templates, diagnostics) =
        load_sourced_prompt_templates([(root.join("prompts"), "project".to_owned())]);
    assert!(diagnostics.is_empty());
    assert_eq!(templates[0].source, "project");
    assert_eq!(templates[0].prompt_template.name, "example");

    fs::write(
        root.join("broken.md"),
        "---\ndescription: [unterminated\n---\nBody",
    )
    .expect("broken");
    let (templates, diagnostics) =
        load_sourced_prompt_templates([(root.join("broken.md"), "user".to_owned())]);
    assert!(templates.is_empty());
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].source, "user");
    assert_eq!(
        diagnostics[0].diagnostic.path,
        root.join("broken.md").to_string_lossy()
    );
    assert_eq!(
        diagnostics[0].diagnostic.code,
        PromptTemplateDiagnosticCode::ParseFailed
    );
}

#[test]
fn prompt_template_argument_substitution_matches_pi_placeholders() {
    let content = "$1 ${@:2} $ARGUMENTS $10";
    let template = PromptTemplate {
        name: "one".to_owned(),
        description: String::new(),
        content: content.to_owned(),
        source: None,
    };
    let args = [
        "hello world".to_owned(),
        "test".to_owned(),
        "three".to_owned(),
        "four".to_owned(),
        "five".to_owned(),
        "six".to_owned(),
        "seven".to_owned(),
        "eight".to_owned(),
        "nine".to_owned(),
        "ten".to_owned(),
    ];
    assert_eq!(
        format_prompt_template_invocation(&template, &args),
        "hello world test three four five six seven eight nine ten hello world test three four five six seven eight nine ten ten"
    );
    assert_eq!(substitute_args("$10 $2 $99", &args), "ten test ");
    assert_eq!(
        parse_command_args("one 'two words' \"three words\""),
        vec!["one", "two words", "three words"]
    );
}
