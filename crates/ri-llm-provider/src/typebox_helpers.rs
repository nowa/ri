use serde_json::{Map, Value};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StringEnumOptions {
    pub description: Option<String>,
    pub default: Option<String>,
}

impl StringEnumOptions {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn default_value(mut self, default: impl Into<String>) -> Self {
        self.default = Some(default.into());
        self
    }
}

pub fn string_enum_schema<I, S>(values: I, options: StringEnumOptions) -> Value
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let enum_values = values
        .into_iter()
        .map(|value| Value::String(value.into()))
        .collect::<Vec<_>>();

    let mut schema = Map::new();
    schema.insert("type".to_owned(), Value::String("string".to_owned()));
    schema.insert("enum".to_owned(), Value::Array(enum_values));
    if let Some(description) = options.description.filter(|value| !value.is_empty()) {
        schema.insert("description".to_owned(), Value::String(description));
    }
    if let Some(default) = options.default.filter(|value| !value.is_empty()) {
        schema.insert("default".to_owned(), Value::String(default));
    }
    Value::Object(schema)
}
