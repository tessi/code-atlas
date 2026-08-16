use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context, Result, bail};
use rayon::prelude::*;

use crate::model::{AnalyzerReport, Atlas, BuildTimings, Metric, Node, NodeKind, Rect};

#[derive(Debug)]
struct FileStats {
    path: String,
    bytes: u64,
    loc: u64,
    commits: u64,
    language: String,
}

pub fn scan_repository(
    root: &Path,
    metric: Metric,
    include_tests: bool,
    include_hidden: bool,
    excluded_paths: &[String],
) -> Result<Atlas> {
    let root = root
        .canonicalize()
        .with_context(|| format!("cannot open repository {}", root.display()))?;
    if !root.join(".git").exists() {
        bail!("{} is not a Git checkout", root.display());
    }

    let excluded_paths = normalize_excluded_paths(excluded_paths)?;
    let tracked = tracked_paths(&root, include_tests, include_hidden, &excluded_paths)?;
    let paths = tracked.paths;
    let commit_counts = commit_counts(&root)?;
    let stats: Result<Vec<_>> = paths
        .par_iter()
        .map(|path| read_file_stats(&root, path, &commit_counts))
        .collect();
    let mut stats = stats?;
    stats.sort_by(|left, right| left.path.cmp(&right.path));

    let revision = git_output(&root, &["rev-parse", "HEAD"])
        .unwrap_or_else(|_| "unborn".to_owned())
        .trim()
        .to_owned();
    let dirty = !git_output(&root, &["status", "--porcelain"])
        .unwrap_or_default()
        .trim()
        .is_empty();

    let mut nodes = vec![Node {
        id: 0,
        parent: None,
        children: Vec::new(),
        kind: NodeKind::Directory,
        name: root
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("repository")
            .to_owned(),
        path: String::new(),
        depth: 0,
        bytes: 0,
        loc: 0,
        commits: 0,
        weight: 0.0,
        language: "directory".to_owned(),
        rect: Rect::default(),
    }];
    let mut path_to_id = HashMap::new();
    path_to_id.insert(String::new(), 0);

    for file in stats {
        let file_path = Path::new(&file.path);
        let components: Vec<_> = file_path.components().collect();
        let mut parent = 0;
        let mut accumulated = PathBuf::new();

        for component in components.iter().take(components.len().saturating_sub(1)) {
            accumulated.push(component.as_os_str());
            let directory_path = accumulated.to_string_lossy().replace('\\', "/");
            if let Some(existing) = path_to_id.get(&directory_path) {
                parent = *existing;
                continue;
            }

            let id = nodes.len();
            nodes.push(Node {
                id,
                parent: Some(parent),
                children: Vec::new(),
                kind: NodeKind::Directory,
                name: component.as_os_str().to_string_lossy().to_string(),
                path: directory_path.clone(),
                depth: nodes[parent].depth + 1,
                bytes: 0,
                loc: 0,
                commits: 0,
                weight: 0.0,
                language: "directory".to_owned(),
                rect: Rect::default(),
            });
            nodes[parent].children.push(id);
            path_to_id.insert(directory_path, id);
            parent = id;
        }

        let id = nodes.len();
        let name = file_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(&file.path)
            .to_owned();
        let weight = metric.value(file.bytes, file.loc, file.commits);
        nodes.push(Node {
            id,
            parent: Some(parent),
            children: Vec::new(),
            kind: NodeKind::File,
            name,
            path: file.path.clone(),
            depth: nodes[parent].depth + 1,
            bytes: file.bytes,
            loc: file.loc,
            commits: file.commits,
            weight,
            language: file.language,
            rect: Rect::default(),
        });
        nodes[parent].children.push(id);
        path_to_id.insert(file.path, id);
    }

    roll_up_metrics(0, &mut nodes);

    Ok(Atlas {
        root_path: root,
        revision,
        dirty,
        excluded_test_files: tracked.excluded_test_files,
        excluded_hidden_files: tracked.excluded_hidden_files,
        excluded_custom_files: tracked.excluded_custom_files,
        excluded_paths,
        nodes,
        calls: Vec::new(),
        path_to_id,
        report: AnalyzerReport::default(),
        timings: BuildTimings::default(),
    })
}

struct TrackedPaths {
    paths: Vec<String>,
    excluded_test_files: usize,
    excluded_hidden_files: usize,
    excluded_custom_files: usize,
}

fn tracked_paths(
    root: &Path,
    include_tests: bool,
    include_hidden: bool,
    excluded_paths: &[String],
) -> Result<TrackedPaths> {
    let output = Command::new("git")
        .args(["-C"])
        .arg(root)
        .args(["ls-files", "-z"])
        .output()
        .context("failed to run git ls-files")?;
    if !output.status.success() {
        bail!("git ls-files failed");
    }
    let mut tracked: Vec<_> = output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .map(|path| String::from_utf8_lossy(path).into_owned())
        .collect();
    // `git ls-files` includes index entries deleted from the working tree. The
    // atlas represents the checkout as it exists now, so those entries are not
    // readable parcels and must not abort an otherwise valid dirty-tree scan.
    tracked.retain(|path| root.join(path).is_file());
    let mut excluded_test_files = 0;
    if !include_tests {
        tracked.retain(|path| {
            let include = !is_test_path(path);
            excluded_test_files += usize::from(!include);
            include
        });
    }

    let mut excluded_hidden_files = 0;
    if !include_hidden {
        tracked.retain(|path| {
            let include = !is_hidden_path(path);
            excluded_hidden_files += usize::from(!include);
            include
        });
    }

    let mut excluded_custom_files = 0;
    if !excluded_paths.is_empty() {
        tracked.retain(|path| {
            let include = !excluded_paths
                .iter()
                .any(|excluded| path == excluded || path.starts_with(&format!("{excluded}/")));
            excluded_custom_files += usize::from(!include);
            include
        });
    }

    Ok(TrackedPaths {
        paths: tracked,
        excluded_test_files,
        excluded_hidden_files,
        excluded_custom_files,
    })
}

fn normalize_excluded_paths(paths: &[String]) -> Result<Vec<String>> {
    let mut normalized = Vec::new();
    for path in paths {
        let path = path.trim().replace('\\', "/");
        let path = path.trim_matches('/');
        if path.is_empty() {
            bail!("custom exclusion paths cannot be empty");
        }
        if Path::new(path).is_absolute()
            || path
                .split('/')
                .any(|component| component == ".." || component.is_empty())
        {
            bail!("custom exclusion {path:?} must be a repository-relative path");
        }
        if !normalized.iter().any(|existing| existing == path) {
            normalized.push(path.to_owned());
        }
    }
    normalized.sort();
    Ok(normalized)
}

/// A hidden path has at least one file or directory component beginning with
/// a dot, such as `.github/workflows/ci.yml` or `lib/.generated/module.ex`.
pub fn is_hidden_path(path: &str) -> bool {
    path.replace('\\', "/")
        .split('/')
        .any(|component| component.starts_with('.') && component.len() > 1)
}

/// Classifies conventional, language-specific test paths without excluding
/// production files that merely contain words such as `test` or `spec`.
pub fn is_test_path(path: &str) -> bool {
    let normalized = path.replace('\\', "/").to_ascii_lowercase();
    let components: Vec<_> = normalized
        .split('/')
        .filter(|component| !component.is_empty())
        .collect();
    let Some(file_name) = components.last().copied() else {
        return false;
    };

    if components
        .iter()
        .take(components.len().saturating_sub(1))
        .any(|component| {
            matches!(
                *component,
                "test" | "tests" | "spec" | "specs" | "__tests__" | "testdata"
            )
        })
    {
        return true;
    }

    if matches!(file_name, "test_helper.exs" | "conftest.py")
        || file_name.starts_with("test.")
        || file_name.contains(".test.")
        || file_name.contains(".spec.")
    {
        return true;
    }

    const PREFIX_EXTENSIONS: &[&str] = &[
        ".py", ".rb", ".exs", ".rs", ".go", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts",
        ".cts",
    ];
    if file_name.starts_with("test_")
        && PREFIX_EXTENSIONS
            .iter()
            .any(|extension| file_name.ends_with(extension))
    {
        return true;
    }

    const SUFFIXES: &[&str] = &[
        "_test.ex",
        "_test.exs",
        "_test.rs",
        "_tests.rs",
        "_test.go",
        "_test.py",
        "_test.rb",
        "_spec.exs",
        "_spec.rb",
    ];
    SUFFIXES.iter().any(|suffix| file_name.ends_with(suffix))
}

fn commit_counts(root: &Path) -> Result<HashMap<String, u64>> {
    let output = Command::new("git")
        .args(["-C"])
        .arg(root)
        .args(["log", "--format=", "--name-only", "--no-renames", "HEAD"])
        .output()
        .context("failed to run git log")?;
    if !output.status.success() {
        return Ok(HashMap::new());
    }

    let mut counts = HashMap::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let path = line.trim();
        if !path.is_empty() {
            *counts.entry(path.to_owned()).or_insert(0) += 1;
        }
    }
    Ok(counts)
}

fn read_file_stats(
    root: &Path,
    path: &str,
    commit_counts: &HashMap<String, u64>,
) -> Result<FileStats> {
    let bytes =
        fs::read(root.join(path)).with_context(|| format!("cannot read tracked file {path}"))?;
    let binary = bytes.iter().take(8192).any(|byte| *byte == 0);
    let loc = if binary {
        0
    } else {
        String::from_utf8_lossy(&bytes).lines().count() as u64
    };
    Ok(FileStats {
        path: path.to_owned(),
        bytes: bytes.len() as u64,
        loc,
        commits: commit_counts.get(path).copied().unwrap_or(1),
        language: language_for_path(path).to_owned(),
    })
}

fn language_for_path(path: &str) -> &'static str {
    match Path::new(path)
        .extension()
        .and_then(|extension| extension.to_str())
    {
        Some("ex") | Some("exs") => "elixir",
        Some("rs") => "rust",
        Some("js") | Some("jsx") | Some("mjs") | Some("cjs") => "javascript",
        Some("ts") | Some("tsx") | Some("mts") | Some("cts") => "typescript",
        Some("gleam") => "gleam",
        Some("py") => "python",
        Some("go") => "go",
        Some("c") | Some("h") => "c",
        Some("cc") | Some("cpp") | Some("hpp") => "cpp",
        Some("java") => "java",
        Some("rb") => "ruby",
        Some("md") => "markdown",
        Some("toml") | Some("lock") => "config",
        Some("json") | Some("yml") | Some("yaml") => "data",
        Some("wit") | Some("wat") | Some("wasm") => "wasm",
        _ => "other",
    }
}

fn roll_up_metrics(id: usize, nodes: &mut [Node]) -> (u64, u64, u64, f64) {
    if nodes[id].is_file() {
        return (
            nodes[id].bytes,
            nodes[id].loc,
            nodes[id].commits,
            nodes[id].weight,
        );
    }

    let children = nodes[id].children.clone();
    let mut totals = (0, 0, 0, 0.0);
    for child in children {
        let child_totals = roll_up_metrics(child, nodes);
        totals.0 += child_totals.0;
        totals.1 += child_totals.1;
        totals.2 += child_totals.2;
        totals.3 += child_totals.3;
    }
    nodes[id].bytes = totals.0;
    nodes[id].loc = totals.1;
    nodes[id].commits = totals.2;
    nodes[id].weight = totals.3.max(1.0);
    totals
}

fn git_output(root: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .args(["-C"])
        .arg(root)
        .args(args)
        .output()
        .context("failed to invoke git")?;
    if !output.status.success() {
        bail!(
            "git command failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_conventional_test_directories_and_files() {
        for path in [
            "test/wasmex_test.exs",
            "tests/integration.rs",
            "spec/models/widget_spec.rb",
            "src/__tests__/widget.test.ts",
            "pkg/testdata/input.json",
            "lib/widget_test.exs",
            "src/widget_test.go",
            "src/test_widget.py",
            "src/widget.test.tsx",
            "src/widget.spec.js",
            "src/widget.test.mts",
            "src/widget.spec.cjs",
            "src/widget_spec.rb",
            "conftest.py",
            "test_helper.exs",
            "config/test.exs",
            "test.json",
        ] {
            assert!(
                is_test_path(path),
                "expected {path} to be classified as a test"
            );
        }
    }

    #[test]
    fn recognizes_analyzed_language_extensions() {
        for path in ["src/a.js", "src/a.jsx", "src/a.mjs", "src/a.cjs"] {
            assert_eq!(language_for_path(path), "javascript");
        }
        for path in ["src/a.ts", "src/a.tsx", "src/a.mts", "src/a.cts"] {
            assert_eq!(language_for_path(path), "typescript");
        }
        assert_eq!(language_for_path("src/a.gleam"), "gleam");
    }

    #[test]
    fn does_not_match_test_words_inside_production_names() {
        for path in [
            "src/contest.rs",
            "lib/testing_tools.ex",
            "docs/test-strategy.md",
            "src/specification.rs",
            "src/testimonial.ts",
            "lib/fixture_loader.ex",
        ] {
            assert!(
                !is_test_path(path),
                "expected {path} to remain in the production atlas"
            );
        }
    }

    #[test]
    fn recognizes_hidden_files_and_directories() {
        for path in [
            ".github/workflows/ci.yml",
            ".formatter.exs",
            "lib/.generated/module.ex",
            "assets/.cache/data.json",
        ] {
            assert!(
                is_hidden_path(path),
                "expected {path} to be classified as hidden"
            );
        }
        for path in [
            "github/workflows/ci.yml",
            "formatter.exs",
            "lib/generated/module.ex",
            "docs/dot-files.md",
        ] {
            assert!(!is_hidden_path(path), "expected {path} to remain visible");
        }
    }

    #[test]
    fn scanner_excludes_tests_and_hidden_paths_with_independent_overrides() {
        let directory = tempfile::tempdir().unwrap();
        fs::create_dir_all(directory.path().join("lib")).unwrap();
        fs::create_dir_all(directory.path().join("lib/.generated")).unwrap();
        fs::create_dir_all(directory.path().join("test")).unwrap();
        fs::create_dir_all(directory.path().join(".github/workflows")).unwrap();
        fs::write(
            directory.path().join("lib/widget.ex"),
            "defmodule Widget do\nend\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("lib/.generated/widget.ex"),
            "defmodule GeneratedWidget do\nend\n",
        )
        .unwrap();
        fs::write(directory.path().join(".formatter.exs"), "[inputs: []]\n").unwrap();
        fs::write(
            directory.path().join(".github/workflows/ci.yml"),
            "name: CI\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("lib/widget_test.exs"),
            "defmodule WidgetTest do\nend\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("test/widget_test.exs"),
            "defmodule WidgetTest do\nend\n",
        )
        .unwrap();

        let init = Command::new("git")
            .args(["init", "--quiet"])
            .arg(directory.path())
            .status()
            .unwrap();
        assert!(init.success());
        let add = Command::new("git")
            .args(["-C"])
            .arg(directory.path())
            .args(["add", "."])
            .status()
            .unwrap();
        assert!(add.success());

        let atlas = scan_repository(directory.path(), Metric::Loc, false, false, &[]).unwrap();
        assert_eq!(atlas.files().count(), 1);
        assert_eq!(atlas.excluded_test_files, 2);
        assert_eq!(atlas.excluded_hidden_files, 3);
        assert!(atlas.path_to_id.contains_key("lib/widget.ex"));

        let atlas = scan_repository(directory.path(), Metric::Loc, true, false, &[]).unwrap();
        assert_eq!(atlas.files().count(), 3);
        assert_eq!(atlas.excluded_test_files, 0);
        assert_eq!(atlas.excluded_hidden_files, 3);

        let atlas = scan_repository(directory.path(), Metric::Loc, false, true, &[]).unwrap();
        assert_eq!(atlas.files().count(), 4);
        assert_eq!(atlas.excluded_test_files, 2);
        assert_eq!(atlas.excluded_hidden_files, 0);

        let atlas = scan_repository(directory.path(), Metric::Loc, true, true, &[]).unwrap();
        assert_eq!(atlas.files().count(), 6);
        assert_eq!(atlas.excluded_test_files, 0);
        assert_eq!(atlas.excluded_hidden_files, 0);
    }

    #[test]
    fn scanner_excludes_repeatable_repository_relative_paths() {
        let directory = tempfile::tempdir().unwrap();
        fs::create_dir_all(directory.path().join("assets/src")).unwrap();
        fs::create_dir_all(directory.path().join("lib")).unwrap();
        fs::write(directory.path().join("assets/src/app.ts"), "run();\n").unwrap();
        fs::write(directory.path().join("assets/logo.svg"), "<svg/>\n").unwrap();
        fs::write(
            directory.path().join("lib/app.ex"),
            "defmodule App do\nend\n",
        )
        .unwrap();
        assert!(
            Command::new("git")
                .args(["init", "--quiet"])
                .arg(directory.path())
                .status()
                .unwrap()
                .success()
        );
        assert!(
            Command::new("git")
                .args(["-C"])
                .arg(directory.path())
                .args(["add", "."])
                .status()
                .unwrap()
                .success()
        );

        let atlas = scan_repository(
            directory.path(),
            Metric::Loc,
            false,
            false,
            &["assets".to_owned()],
        )
        .unwrap();
        assert_eq!(atlas.files().count(), 1);
        assert_eq!(atlas.excluded_custom_files, 2);
        assert_eq!(atlas.excluded_paths, vec!["assets"]);
        assert!(atlas.path_to_id.contains_key("lib/app.ex"));
    }

    #[test]
    fn scanner_skips_tracked_files_deleted_from_the_worktree() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("present.rs"), "fn present() {}\n").unwrap();
        fs::write(directory.path().join("deleted.rs"), "fn deleted() {}\n").unwrap();
        assert!(
            Command::new("git")
                .args(["init", "--quiet"])
                .arg(directory.path())
                .status()
                .unwrap()
                .success()
        );
        assert!(
            Command::new("git")
                .args(["-C"])
                .arg(directory.path())
                .args(["add", "."])
                .status()
                .unwrap()
                .success()
        );
        fs::remove_file(directory.path().join("deleted.rs")).unwrap();

        let atlas = scan_repository(directory.path(), Metric::Loc, false, false, &[]).unwrap();

        assert_eq!(atlas.files().count(), 1);
        assert!(atlas.path_to_id.contains_key("present.rs"));
        assert!(!atlas.path_to_id.contains_key("deleted.rs"));
        assert!(atlas.dirty);
    }
}
