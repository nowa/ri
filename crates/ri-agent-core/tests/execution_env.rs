use ri_agent_core::*;
use std::{
    fs,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

#[cfg(unix)]
use std::os::unix::fs as unix_fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

fn temp_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ri-env-test-{}", uuidv7()));
    fs::create_dir_all(&dir).expect("temp dir");
    dir
}

#[test]
fn local_execution_env_reads_writes_lists_and_removes_files() {
    let root = temp_dir();
    let env = LocalExecutionEnv::new(&root);

    assert_eq!(
        env.absolute_path("nested/child"),
        root.join("nested/child").to_string_lossy()
    );
    assert_eq!(
        env.join_path([
            root.clone(),
            PathBuf::from("nested"),
            PathBuf::from("child")
        ]),
        root.join("nested/child").to_string_lossy()
    );

    env.create_dir("nested/child", CreateDirOptions::default())
        .expect("create dir");
    env.write_file("nested/child/file.txt", b"hel")
        .expect("write");
    env.append_file("nested/child/file.txt", b"lo")
        .expect("append");
    assert_eq!(
        env.read_text_file("nested/child/file.txt").expect("read"),
        "hello"
    );
    assert_eq!(
        env.read_text_lines("nested/child/file.txt", 1)
            .expect("lines"),
        vec!["hello".to_owned()]
    );
    assert_eq!(
        env.read_binary_file("nested/child/file.txt")
            .expect("binary"),
        b"hello"
    );

    let entries = env.list_dir("nested/child").expect("list");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].name, "file.txt");
    assert_eq!(
        entries[0].path,
        root.join("nested/child/file.txt").to_string_lossy()
    );
    assert_eq!(entries[0].kind, FileKind::File);
    assert_eq!(entries[0].size, 5);
    assert!(entries[0].mtime_ms > 0);

    assert!(env.exists("nested/child/file.txt").expect("exists"));
    env.remove("nested/child/file.txt", RemoveOptions::default())
        .expect("remove");
    assert!(!env.exists("nested/child/file.txt").expect("exists"));
}

#[cfg(unix)]
#[test]
fn local_execution_env_reports_symlinks_without_following_them() {
    let root = temp_dir();
    let env = LocalExecutionEnv::new(&root);
    env.create_dir("dir", CreateDirOptions::default())
        .expect("dir");
    env.write_file("dir/file.txt", b"hello").expect("file");
    unix_fs::symlink(root.join("dir/file.txt"), root.join("file-link")).expect("file symlink");
    unix_fs::symlink(root.join("dir"), root.join("dir-link")).expect("dir symlink");

    let dir_info = env.file_info("dir").expect("dir info");
    assert_eq!(dir_info.name, "dir");
    assert_eq!(dir_info.path, root.join("dir").to_string_lossy());
    assert_eq!(dir_info.kind, FileKind::Directory);
    assert!(dir_info.mtime_ms > 0);

    let file_info = env.file_info("dir/file.txt").expect("file info");
    assert_eq!(file_info.name, "file.txt");
    assert_eq!(file_info.path, root.join("dir/file.txt").to_string_lossy());
    assert_eq!(file_info.kind, FileKind::File);
    assert_eq!(file_info.size, 5);
    assert!(file_info.mtime_ms > 0);

    let file_link = env.file_info("file-link").expect("file link");
    assert_eq!(file_link.name, "file-link");
    assert_eq!(file_link.path, root.join("file-link").to_string_lossy());
    assert_eq!(file_link.kind, FileKind::Symlink);

    let dir_link = env.file_info("dir-link").expect("dir link");
    assert_eq!(dir_link.name, "dir-link");
    assert_eq!(dir_link.path, root.join("dir-link").to_string_lossy());
    assert_eq!(dir_link.kind, FileKind::Symlink);
    assert_eq!(
        env.canonical_path("file-link").expect("canonical"),
        fs::canonicalize(root.join("dir/file.txt"))
            .expect("realpath")
            .to_string_lossy()
    );

    let entries = env
        .list_dir(".")
        .expect("list")
        .into_iter()
        .map(|entry| (entry.name, entry.kind))
        .collect::<Vec<_>>();
    assert!(entries.contains(&("file-link".to_owned(), FileKind::Symlink)));
}

#[cfg(unix)]
#[test]
fn local_execution_env_lists_symlinks_as_symlinks() {
    let root = temp_dir();
    let env = LocalExecutionEnv::new(&root);
    env.write_file("target.txt", b"hello").expect("target");
    unix_fs::symlink(root.join("target.txt"), root.join("link.txt")).expect("symlink");

    let mut entries = env
        .list_dir(".")
        .expect("list")
        .into_iter()
        .map(|entry| (entry.name, entry.kind))
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    assert_eq!(
        entries,
        vec![
            ("link.txt".to_owned(), FileKind::Symlink),
            ("target.txt".to_owned(), FileKind::File)
        ]
    );
}

#[test]
fn local_execution_env_read_text_lines_stops_at_requested_limit() {
    let root = temp_dir();
    let env = LocalExecutionEnv::new(&root);
    env.write_file("file.txt", b"one\ntwo\nthree")
        .expect("write");

    assert_eq!(
        env.read_text_lines("file.txt", 1).expect("lines"),
        vec!["one".to_owned()]
    );
}

#[test]
fn local_execution_env_text_reads_replace_invalid_utf8_like_node() {
    let root = temp_dir();
    let env = LocalExecutionEnv::new(&root);
    env.write_file("invalid.txt", b"a\xffb\n\xfe")
        .expect("write invalid utf8");

    assert_eq!(
        env.read_text_file("invalid.txt").expect("read"),
        "a\u{fffd}b\n\u{fffd}"
    );
    assert_eq!(
        env.read_text_lines("invalid.txt", 2).expect("lines"),
        vec!["a\u{fffd}b".to_owned(), "\u{fffd}".to_owned()]
    );
}

#[test]
fn local_execution_env_returns_file_errors_for_missing_and_wrong_kinds() {
    let root = temp_dir();
    let env = LocalExecutionEnv::new(&root);

    let missing = env.file_info("missing.txt").expect_err("missing");
    assert_eq!(missing.code, FileErrorCode::NotFound);
    assert_eq!(missing.path, root.join("missing.txt").to_string_lossy());
    assert!(!env.exists("missing.txt").expect("exists"));

    env.write_file("file.txt", b"hello").expect("file");
    let not_dir = env.list_dir("file.txt").expect_err("not dir");
    assert_eq!(not_dir.code, FileErrorCode::NotDirectory);

    env.create_dir("dir", CreateDirOptions::default())
        .expect("dir");
    let is_dir = env.read_text_file("dir").expect_err("is dir");
    assert_eq!(is_dir.code, FileErrorCode::IsDirectory);

    let create = env
        .create_dir("missing/child", CreateDirOptions { recursive: false })
        .expect_err("missing parent");
    assert_eq!(create.code, FileErrorCode::NotFound);
}

#[test]
fn local_execution_env_appends_creates_temps_and_removes_recursively() {
    let root = temp_dir();
    let env = LocalExecutionEnv::new(&root);

    env.append_file("new/nested/file.txt", b"a")
        .expect("append a");
    env.append_file("new/nested/file.txt", b"b")
        .expect("append b");
    assert_eq!(
        env.read_text_file("new/nested/file.txt").expect("read"),
        "ab"
    );

    let temp_dir = env.create_temp_dir("node-env-test-").expect("temp dir");
    assert!(PathBuf::from(&temp_dir).is_dir());
    let temp_file = env.create_temp_file("prefix-", ".txt").expect("temp file");
    assert!(PathBuf::from(&temp_file).is_file());
    assert!(
        PathBuf::from(&temp_file)
            .parent()
            .and_then(|path| path.file_name())
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("tmp-"))
    );
    assert!(
        PathBuf::from(&temp_file)
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("prefix-"))
    );
    assert!(temp_file.ends_with(".txt"));

    env.write_file("dir/child/file.txt", b"hello")
        .expect("dir file");
    assert!(
        env.remove(
            "dir",
            RemoveOptions {
                recursive: false,
                force: true
            }
        )
        .is_err()
    );
    env.remove(
        "dir",
        RemoveOptions {
            recursive: true,
            force: true,
        },
    )
    .expect("recursive remove");
    assert!(!env.exists("dir").expect("exists"));

    assert_eq!(
        env.remove(
            "missing",
            RemoveOptions {
                recursive: false,
                force: false
            }
        )
        .expect_err("missing remove")
        .code,
        FileErrorCode::NotFound
    );
    assert_eq!(
        env.remove("missing", RemoveOptions::default())
            .expect_err("default remove is not forced")
            .code,
        FileErrorCode::NotFound
    );
    env.remove(
        "missing",
        RemoveOptions {
            recursive: false,
            force: true,
        },
    )
    .expect("force remove");
}

#[test]
fn local_execution_env_honors_create_dir_recursive_false_and_remove_options() {
    let root = temp_dir();
    let env = LocalExecutionEnv::new(&root);

    let create = env
        .create_dir("missing/child", CreateDirOptions { recursive: false })
        .expect_err("missing parent");
    assert_eq!(create.code, FileErrorCode::NotFound);

    env.write_file("dir/child/file.txt", b"hello")
        .expect("dir file");
    assert!(
        env.remove(
            "dir",
            RemoveOptions {
                recursive: false,
                force: true
            }
        )
        .is_err()
    );
    env.remove(
        "dir",
        RemoveOptions {
            recursive: true,
            force: true,
        },
    )
    .expect("recursive remove");
    assert!(!env.exists("dir").expect("exists"));

    let missing = env
        .remove(
            "missing",
            RemoveOptions {
                recursive: false,
                force: false,
            },
        )
        .expect_err("missing remove");
    assert_eq!(missing.code, FileErrorCode::NotFound);
    env.remove(
        "missing",
        RemoveOptions {
            recursive: false,
            force: true,
        },
    )
    .expect("force remove");
}

#[test]
fn local_execution_env_executes_shell_commands_in_cwd_with_env() {
    let root = temp_dir();
    fs::create_dir_all(root.join("child")).expect("child cwd");
    let env = LocalExecutionEnv::new(&root).with_shell_env(vec![
        ("BASE_ENV_TEST".to_owned(), "base".to_owned()),
        ("OVERRIDE_ENV_TEST".to_owned(), "base".to_owned()),
    ]);
    let output = env
        .exec(
            "printf '%s:%s:%s:%s' \"$PWD\" \"$NODE_ENV_TEST\" \"$BASE_ENV_TEST\" \"$OVERRIDE_ENV_TEST\"",
            ExecOptions {
                cwd: Some(PathBuf::from("child")),
                env: vec![("NODE_ENV_TEST".to_owned(), "ok".to_owned())],
                ..Default::default()
            },
        )
        .expect("exec");
    assert_eq!(
        output.stdout,
        format!("{}:ok:base:base", root.join("child").to_string_lossy())
    );
    assert_eq!(output.stderr, "");
    assert_eq!(output.exit_code, 0);

    let output = env
        .exec(
            "printf '%s' \"$OVERRIDE_ENV_TEST\"",
            ExecOptions {
                env: vec![("OVERRIDE_ENV_TEST".to_owned(), "extra".to_owned())],
                ..Default::default()
            },
        )
        .expect("env override");
    assert_eq!(output.stdout, "extra");

    let non_zero = env.exec("exit 7", ExecOptions::default()).expect("exit");
    assert_eq!(non_zero.exit_code, 7);

    let missing_shell = LocalExecutionEnv::new(&root).with_shell(root.join("missing-shell"));
    let error = missing_shell
        .exec("printf ok", ExecOptions::default())
        .expect_err("missing shell");
    assert_eq!(error.code, FileErrorCode::ShellUnavailable);

    #[cfg(unix)]
    {
        let custom_shell = root.join("custom-shell");
        let arg_file = root.join("custom-shell-arg");
        fs::write(
            &custom_shell,
            format!(
                "#!/bin/sh\nprintf '%s' \"$1\" > '{}'\nexec /bin/sh \"$@\"\n",
                arg_file.to_string_lossy()
            ),
        )
        .expect("write custom shell");
        fs::set_permissions(&custom_shell, fs::Permissions::from_mode(0o755))
            .expect("custom shell permissions");

        let custom = LocalExecutionEnv::new(&root).with_shell(&custom_shell);
        let output = custom
            .exec("printf custom", ExecOptions::default())
            .expect("custom shell");

        assert_eq!(output.stdout, "custom");
        assert_eq!(fs::read_to_string(arg_file).expect("arg file"), "-c");
    }
}

#[test]
fn local_execution_env_returns_non_zero_exit_codes_as_successful_execution_results() {
    let root = temp_dir();
    let env = LocalExecutionEnv::new(&root);

    let output = env
        .exec("printf out; printf err >&2; exit 7", ExecOptions::default())
        .expect("non-zero exit is still an execution result");

    assert_eq!(output.stdout, "out");
    assert_eq!(output.stderr, "err");
    assert_eq!(output.exit_code, 7);
}

#[cfg(unix)]
#[test]
fn local_execution_env_returns_spawn_error_for_non_executable_shell() {
    let root = temp_dir();
    let shell_path = root.join("not-executable-shell");
    fs::write(&shell_path, "not executable").expect("write shell");
    fs::set_permissions(&shell_path, fs::Permissions::from_mode(0o644)).expect("permissions");
    let env = LocalExecutionEnv::new(&root).with_shell(shell_path);

    let error = env
        .exec("printf ok", ExecOptions::default())
        .expect_err("spawn error");

    assert_eq!(error.code, FileErrorCode::Spawn);
}

#[test]
fn local_execution_env_cleanup_is_best_effort() {
    let root = temp_dir();
    let env = LocalExecutionEnv::new(&root);

    env.cleanup();
    env.cleanup();
}

#[test]
fn local_execution_env_streams_stdout_and_stderr_callbacks() {
    let root = temp_dir();
    let env = LocalExecutionEnv::new(&root);
    let stdout = Arc::new(Mutex::new(String::new()));
    let stderr = Arc::new(Mutex::new(String::new()));
    let stdout_callback = Arc::clone(&stdout);
    let stderr_callback = Arc::clone(&stderr);

    let output = env
        .exec(
            "printf out; printf err >&2",
            ExecOptions {
                on_stdout: Some(Arc::new(move |chunk| {
                    stdout_callback.lock().expect("stdout lock").push_str(chunk);
                    Ok(())
                })),
                on_stderr: Some(Arc::new(move |chunk| {
                    stderr_callback.lock().expect("stderr lock").push_str(chunk);
                    Ok(())
                })),
                ..Default::default()
            },
        )
        .expect("exec");

    assert_eq!(output.stdout, "out");
    assert_eq!(output.stderr, "err");
    assert_eq!(output.exit_code, 0);
    assert_eq!(&*stdout.lock().expect("stdout lock"), "out");
    assert_eq!(&*stderr.lock().expect("stderr lock"), "err");
}

#[test]
fn local_execution_env_returns_callback_errors_from_exec_handlers() {
    let root = temp_dir();
    let env = LocalExecutionEnv::new(&root);
    let error = env
        .exec(
            "printf out; sleep 5",
            ExecOptions {
                on_stdout: Some(Arc::new(|_| Err("callback failed".to_owned()))),
                ..Default::default()
            },
        )
        .expect_err("callback error");

    assert_eq!(error.code, FileErrorCode::CallbackError);
    assert_eq!(error.message, "callback failed");
}

#[test]
fn local_execution_env_times_out_long_running_commands() {
    let root = temp_dir();
    let env = LocalExecutionEnv::new(&root);
    let error = env
        .exec(
            "sleep 1",
            ExecOptions {
                timeout_ms: Some(10),
                ..Default::default()
            },
        )
        .expect_err("timeout");
    assert_eq!(error.code, FileErrorCode::Timeout);
}

#[test]
fn local_execution_env_returns_aborted_for_aborted_commands() {
    let root = temp_dir();
    let env = LocalExecutionEnv::new(&root);

    let pre_aborted = Arc::new(AtomicBool::new(true));
    let error = env
        .exec(
            "printf should-not-run > marker",
            ExecOptions {
                abort_flag: Some(Arc::clone(&pre_aborted)),
                ..Default::default()
            },
        )
        .expect_err("pre-aborted");
    assert_eq!(error.code, FileErrorCode::Aborted);
    assert!(!root.join("marker").exists());

    let abort_flag = Arc::new(AtomicBool::new(false));
    let abort_thread_flag = Arc::clone(&abort_flag);
    let abort_thread = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(10));
        abort_thread_flag.store(true, Ordering::SeqCst);
    });
    let error = env
        .exec(
            "sleep 1",
            ExecOptions {
                abort_flag: Some(abort_flag),
                ..Default::default()
            },
        )
        .expect_err("aborted");
    abort_thread.join().expect("abort thread");
    assert_eq!(error.code, FileErrorCode::Aborted);
}

#[test]
fn local_execution_env_returns_aborted_for_pre_aborted_file_operations() {
    let root = temp_dir();
    let env = LocalExecutionEnv::new(&root);
    env.write_file("file.txt", b"hello").expect("write");
    let abort_flag = AtomicBool::new(true);

    let results = [
        env.read_text_file_with_abort("file.txt", Some(&abort_flag))
            .map(|_| ()),
        env.read_text_lines_with_abort("file.txt", 1, Some(&abort_flag))
            .map(|_| ()),
        env.read_binary_file_with_abort("file.txt", Some(&abort_flag))
            .map(|_| ()),
        env.write_file_with_abort("other.txt", b"hello", Some(&abort_flag))
            .map(|_| ()),
        env.list_dir_with_abort(".", Some(&abort_flag)).map(|_| ()),
    ];

    for result in results {
        let error = result.expect_err("aborted operation");
        assert_eq!(error.code, FileErrorCode::Aborted);
    }
    assert!(!root.join("other.txt").exists());
}

#[test]
fn shell_capture_sanitizes_and_writes_large_output_file() {
    assert_eq!(sanitize_binary_output("a\u{0}b\tc\n\u{fff9}d"), "ab\tc\nd");

    let root = temp_dir();
    let env = LocalExecutionEnv::new(&root);
    let result =
        execute_shell_with_capture(&env, "yes line | head -n 15000", ExecOptions::default())
            .expect("capture");

    assert_eq!(result.exit_code, Some(0));
    assert!(!result.cancelled);
    assert!(result.truncated);
    let full_output_path = result.full_output_path.expect("full output path");
    let full_output = env.read_text_file(&full_output_path).expect("full output");
    assert!(full_output.split('\n').count() > 10_000);
    assert!(result.output.len() < full_output.len());
}

#[test]
fn shell_capture_returns_cancelled_result_with_partial_output_on_abort() {
    let root = temp_dir();
    let env = LocalExecutionEnv::new(&root);
    let abort_flag = Arc::new(AtomicBool::new(false));
    let callback_abort_flag = abort_flag.clone();

    let result = execute_shell_with_capture(
        &env,
        "printf tick; sleep 5",
        ExecOptions {
            abort_flag: Some(abort_flag),
            on_stdout: Some(Arc::new(move |_| {
                callback_abort_flag.store(true, Ordering::SeqCst);
                Ok(())
            })),
            ..Default::default()
        },
    )
    .expect("cancelled capture");

    assert!(result.cancelled);
    assert_eq!(result.exit_code, None);
    assert_eq!(result.output, "tick");
    assert!(!result.truncated);
}
