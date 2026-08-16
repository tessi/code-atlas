use std::{
    collections::{HashMap, HashSet},
    env, fs,
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
};

use anyhow::Result;
use regex::Regex;
use syn::{Expr, spanned::Spanned, visit::Visit};

use serde::{Deserialize, Serialize};

use crate::{
    model::{AnalyzerReport, Atlas, Callsite, LanguageCoverage, NodeId},
    scip::{ScipDocument, ScipIndex, ScipRange, decode_index},
};

const CACHE_VERSION: u32 = 7;
const FILE_MARKER: &str = "__CODE_ATLAS_FILE__\t";
const END_MARKER: &str = "__CODE_ATLAS_END__";
const ERROR_MARKER: &str = "__CODE_ATLAS_ERROR__\t";
const SCRIPT_CALL_MARKER: &str = "__CODE_ATLAS_SCRIPT_CALL__\t";
const XREF_BATCH_EXPRESSION: &str = r#"
for path <- IO.stream(:stdio, :line) do
  path = String.trim(path)
  IO.puts("__CODE_ATLAS_FILE__\t" <> path)
  Mix.Task.reenable("xref")
  try do
    Mix.Tasks.Xref.run(["trace", path, "--no-compile", "--no-deps-check"])
  rescue
    error -> IO.puts("__CODE_ATLAS_ERROR__\t" <> Exception.message(error))
  catch
    kind, reason -> IO.puts("__CODE_ATLAS_ERROR__\t" <> Exception.format(kind, reason, __STACKTRACE__))
  end
  IO.puts("__CODE_ATLAS_END__")
end
"#;
const ELIXIR_SCRIPT_EXPRESSION: &str = r#"
for raw_path <- IO.stream(:stdio, :line) do
  path = String.trim(raw_path)
  IO.puts("__CODE_ATLAS_FILE__\t" <> path)
  case File.read(path) do
    {:ok, contents} ->
      case Code.string_to_quoted(contents, columns: true) do
        {:ok, ast} ->
          {_ast, aliases} = Macro.prewalk(ast, %{}, fn
            {:alias, _, [{:__aliases__, _, parts} | options]} = node, aliases ->
              full = Enum.map_join(parts, ".", &Atom.to_string/1)
              short = case Keyword.get(List.first(options) || [], :as) do
                {:__aliases__, _, as_parts} -> as_parts |> List.last() |> Atom.to_string()
                _ -> parts |> List.last() |> Atom.to_string()
              end
              {node, Map.put(aliases, short, full)}
            node, aliases -> {node, aliases}
          end)
          Macro.prewalk(ast, fn
            {{:., _, [{:__aliases__, _, parts}, function]}, meta, arguments} = node
                when is_atom(function) and is_list(arguments) ->
              names = Enum.map(parts, &Atom.to_string/1)
              module = case names do
                [first | rest] ->
                  case Map.get(aliases, first) do
                    nil -> Enum.join(names, ".")
                    resolved -> Enum.join([resolved | rest], ".")
                  end
                [] -> ""
              end
              IO.puts("__CODE_ATLAS_SCRIPT_CALL__\t#{meta[:line] || 1}\t#{module}\t#{function}\t#{length(arguments)}")
              node
            node -> node
          end)
        {:error, error} ->
          message = error |> inspect(limit: 10) |> String.replace("\n", " ") |> String.replace("\r", " ") |> String.replace("\t", " ")
          IO.puts("__CODE_ATLAS_ERROR__\t" <> message)
      end
    {:error, error} -> IO.puts("__CODE_ATLAS_ERROR__\t" <> inspect(error))
  end
  IO.puts("__CODE_ATLAS_END__")
end
"#;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XrefCall {
    pub source_line: u32,
    pub module: String,
    pub function: String,
    pub arity: u32,
    pub kind: String,
}

#[derive(Debug, Clone)]
struct RustlerTarget {
    file: NodeId,
    line: Option<u32>,
}

fn mix_project_roots(atlas: &Atlas) -> Vec<String> {
    let mut roots: Vec<_> = atlas
        .nodes
        .iter()
        .filter(|node| {
            node.is_file() && (node.path == "mix.exs" || node.path.ends_with("/mix.exs"))
        })
        .filter_map(|node| {
            Path::new(&node.path)
                .parent()
                .map(|parent| parent.to_string_lossy().trim_end_matches('/').to_owned())
        })
        .collect();
    roots.sort_by_key(|root| (root.matches('/').count(), root.clone()));
    roots.dedup();
    roots
}

fn elixir_files_for_project(
    atlas: &Atlas,
    project_root: &str,
    project_roots: &[String],
) -> Vec<(String, NodeId)> {
    atlas
        .nodes
        .iter()
        .filter(|node| node.is_file() && node.path.ends_with(".ex"))
        .filter(|node| nearest_project_root(&node.path, project_roots) == Some(project_root))
        .map(|node| {
            let input_path = if project_root.is_empty() {
                node.path.clone()
            } else {
                node.path
                    .strip_prefix(project_root)
                    .and_then(|path| path.strip_prefix('/'))
                    .unwrap_or(&node.path)
                    .to_owned()
            };
            (input_path, node.id)
        })
        .collect()
}

fn nearest_project_root<'a>(path: &str, project_roots: &'a [String]) -> Option<&'a str> {
    project_roots
        .iter()
        .filter(|root| {
            root.is_empty()
                || path == root.as_str()
                || path
                    .strip_prefix(root.as_str())
                    .is_some_and(|suffix| suffix.starts_with('/'))
        })
        .max_by_key(|root| root.len())
        .map(String::as_str)
}

fn display_project_root(root: &str) -> &str {
    if root.is_empty() { "." } else { root }
}

pub fn analyze_calls(atlas: &mut Atlas) -> Result<()> {
    if load_cached_analysis(atlas)? {
        exclude_same_file_calls(atlas);
        eprintln!(
            "Loaded {} cross-file callsites from the repository cache ({} same-file callsites excluded).",
            atlas.calls.len(),
            atlas.report.same_file_calls_excluded
        );
        return Ok(());
    }

    initialize_language_coverage(atlas);

    let module_files = elixir_module_files(atlas)?;
    let rustler_targets = rustler_targets(atlas, &module_files)?;
    let mut next_id = atlas.calls.len() as u64;

    let project_roots = mix_project_roots(atlas);
    let mut cacheable = true;
    if !project_roots.is_empty() {
        for project_root in &project_roots {
            let files = elixir_files_for_project(atlas, project_root, &project_roots);
            cacheable &= analyze_elixir_calls(
                atlas,
                &module_files,
                &rustler_targets,
                &mut next_id,
                project_root,
                files,
            );
        }
    } else if atlas
        .nodes
        .iter()
        .any(|node| node.is_file() && node.language == "elixir")
    {
        atlas
            .report
            .warnings
            .push("no mix.exs found; Elixir xref analyzer was skipped".to_owned());
    }
    analyze_elixir_scripts(atlas, &module_files, &mut next_id)?;

    cacheable &= analyze_typescript_calls(atlas, &mut next_id)?;
    analyze_gleam_calls(atlas, &mut next_id)?;
    cacheable &= analyze_rust_calls(atlas, &mut next_id)?;

    if cacheable {
        save_cached_analysis(atlas)?;
    }

    exclude_same_file_calls(atlas);

    Ok(())
}

fn initialize_language_coverage(atlas: &mut Atlas) {
    for language in ["elixir", "rust", "javascript", "typescript", "gleam"] {
        let files_discovered = atlas
            .nodes
            .iter()
            .filter(|node| node.is_file() && node.language == language)
            .count();
        atlas.report.language_coverage.insert(
            language.to_owned(),
            LanguageCoverage {
                provider: "none".to_owned(),
                fidelity: "not applicable".to_owned(),
                files_discovered,
                ..LanguageCoverage::default()
            },
        );
    }
    if coverage_mut(atlas, "elixir").files_discovered > 0 {
        let elixir = coverage_mut(atlas, "elixir");
        elixir.provider = "mix-xref + script-static".to_owned();
        elixir.fidelity = "compiler-backed project files; conservative scripts".to_owned();
    }
    if coverage_mut(atlas, "gleam").files_discovered > 0 {
        let gleam = coverage_mut(atlas, "gleam");
        gleam.provider = "gleam-import-static".to_owned();
        gleam.fidelity = "conservative explicit-import resolution".to_owned();
    }
}

fn coverage_mut<'a>(atlas: &'a mut Atlas, language: &str) -> &'a mut LanguageCoverage {
    atlas
        .report
        .language_coverage
        .entry(language.to_owned())
        .or_default()
}

fn exclude_same_file_calls(atlas: &mut Atlas) {
    let before = atlas.calls.len();
    atlas.calls.retain(|call| call.source != call.target);
    atlas.report.same_file_calls_excluded = before - atlas.calls.len();
}

fn analyze_elixir_calls(
    atlas: &mut Atlas,
    module_files: &HashMap<String, NodeId>,
    rustler_targets: &HashMap<(String, String), RustlerTarget>,
    next_id: &mut u64,
    project_root: &str,
    files: Vec<(String, NodeId)>,
) -> bool {
    // Prepare the whole project once. The subsequent traces share one Mix/BEAM
    // runtime and explicitly skip compilation.
    let compile = Command::new("mix")
        .arg("compile")
        .env("MIX_OS_CONCURRENCY_LOCK", "0")
        .current_dir(atlas.root_path.join(project_root))
        .output();
    match compile {
        Ok(output) if output.status.success() => {}
        Ok(output) => {
            atlas.report.warnings.push(format!(
                "mix compile failed in {} before xref analysis; prepare the checkout with its documented toolchain and dependencies: {}",
                display_project_root(project_root),
                String::from_utf8_lossy(&output.stderr).trim()
            ));
            return false;
        }
        Err(error) => {
            atlas.report.warnings.push(format!(
                "could not run mix compile in {} before xref analysis: {error}",
                display_project_root(project_root)
            ));
            return false;
        }
    }

    if files.is_empty() {
        return true;
    }
    eprintln!(
        "Tracing {} Elixir files in {} with one Mix runtime…",
        files.len(),
        display_project_root(project_root)
    );
    let source_ids: HashMap<_, _> = files
        .iter()
        .map(|(input_path, id)| (input_path.clone(), *id))
        .collect();
    let mut child = match Command::new("mix")
        .args([
            "run",
            "--no-compile",
            "--no-deps-check",
            "--no-start",
            "-e",
            XREF_BATCH_EXPRESSION,
        ])
        .env("MIX_OS_CONCURRENCY_LOCK", "0")
        .current_dir(atlas.root_path.join(project_root))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            atlas
                .report
                .warnings
                .push(format!("could not start batched Mix xref: {error}"));
            return false;
        }
    };
    let mut stdin = child.stdin.take().expect("piped stdin");
    let input_paths: Vec<_> = files
        .iter()
        .map(|(input_path, _)| input_path.clone())
        .collect();
    let writer = thread::spawn(move || {
        for path in input_paths {
            writeln!(stdin, "{path}")?;
        }
        Ok::<_, std::io::Error>(())
    });
    let mut stderr = child.stderr.take().expect("piped stderr");
    let stderr_reader = thread::spawn(move || {
        let mut contents = String::new();
        stderr.read_to_string(&mut contents).map(|_| contents)
    });
    let stdout = child.stdout.take().expect("piped stdout");
    let mut current: Option<(String, NodeId)> = None;
    let mut current_failed = false;
    let mut project_completed = 0_usize;
    for line in BufReader::new(stdout).lines() {
        let line = match line {
            Ok(line) => line,
            Err(error) => {
                atlas
                    .report
                    .warnings
                    .push(format!("could not read batched xref output: {error}"));
                break;
            }
        };
        if let Some(path) = line.strip_prefix(FILE_MARKER) {
            current = source_ids
                .get(path)
                .copied()
                .map(|id| (path.to_owned(), id));
            current_failed = false;
            continue;
        }
        if let Some(error) = line.strip_prefix(ERROR_MARKER) {
            let path = current
                .as_ref()
                .map(|(path, _)| path.as_str())
                .unwrap_or("unknown file");
            atlas
                .report
                .warnings
                .push(format!("mix xref trace failed for {path}: {error}"));
            current_failed = true;
            continue;
        }
        if line == END_MARKER {
            if current.is_some() && !current_failed {
                atlas.report.elixir_files_traced += 1;
                coverage_mut(atlas, "elixir").files_analyzed += 1;
                project_completed += 1;
                if project_completed.is_multiple_of(100) || project_completed == files.len() {
                    eprintln!(
                        "  traced {project_completed}/{} Elixir files in {}",
                        files.len(),
                        display_project_root(project_root)
                    );
                }
            }
            current = None;
            continue;
        }
        if let Some((_, source_id)) = current.as_ref() {
            let Some(call) = parse_xref_line(&line) else {
                continue;
            };
            let key = (call.module.clone(), call.function.clone());
            let (target, target_line, analyzer) = if let Some(target) = rustler_targets.get(&key) {
                atlas.report.rustler_calls += 1;
                (target.file, target.line, "elixir-xref+rustler")
            } else if let Some(target) = module_files.get(&call.module) {
                (
                    *target,
                    definition_line(
                        &atlas.root_path.join(&atlas.nodes[*target].path),
                        &call.function,
                    ),
                    "elixir-xref",
                )
            } else {
                atlas.report.unresolved_calls += 1;
                coverage_mut(atlas, "elixir").callsites_unresolved += 1;
                continue;
            };

            atlas.calls.push(Callsite {
                id: *next_id,
                source: *source_id,
                source_line: call.source_line,
                target,
                target_line,
                callee: format!("{}.{}/{}", call.module, call.function, call.arity),
                kind: call.kind,
                analyzer: analyzer.to_owned(),
                confidence: if analyzer.contains("rustler") {
                    0.98
                } else {
                    1.0
                },
            });
            *next_id += 1;
            atlas.report.resolved_calls += 1;
            coverage_mut(atlas, "elixir").callsites_resolved += 1;
        }
    }
    let writer_succeeded = match writer.join().expect("xref input writer panicked") {
        Ok(()) => true,
        Err(error) => {
            atlas
                .report
                .warnings
                .push(format!("could not send xref input: {error}"));
            false
        }
    };
    let status = child.wait();
    let stderr = stderr_reader
        .join()
        .expect("xref stderr reader panicked")
        .unwrap_or_default();
    let batch_succeeded = match status {
        Ok(status) if status.success() => true,
        Ok(status) => {
            atlas.report.warnings.push(format!(
                "batched Mix xref exited with {status}: {}",
                stderr.trim()
            ));
            false
        }
        Err(error) => {
            atlas
                .report
                .warnings
                .push(format!("could not wait for batched Mix xref: {error}"));
            false
        }
    };
    writer_succeeded && batch_succeeded
}

#[derive(Debug, Serialize, Deserialize)]
struct CachedAnalysis {
    version: u32,
    calls: Vec<CachedCallsite>,
    report: AnalyzerReport,
}

#[derive(Debug, Serialize, Deserialize)]
struct CachedCallsite {
    source_path: String,
    source_line: u32,
    target_path: String,
    target_line: Option<u32>,
    callee: String,
    kind: String,
    analyzer: String,
    confidence: f32,
}

fn load_cached_analysis(atlas: &mut Atlas) -> Result<bool> {
    let Some(path) = analysis_cache_path(atlas) else {
        return Ok(false);
    };
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            atlas
                .report
                .warnings
                .push(format!("could not read analysis cache: {error}"));
            return Ok(false);
        }
    };
    let cached: CachedAnalysis = match serde_json::from_slice(&bytes) {
        Ok(cached) => cached,
        Err(error) => {
            atlas
                .report
                .warnings
                .push(format!("ignored invalid analysis cache: {error}"));
            return Ok(false);
        }
    };
    if cached.version != CACHE_VERSION {
        return Ok(false);
    }
    let mut calls = Vec::with_capacity(cached.calls.len());
    for (id, call) in cached.calls.into_iter().enumerate() {
        let (Some(source), Some(target)) = (
            atlas.path_to_id.get(&call.source_path).copied(),
            atlas.path_to_id.get(&call.target_path).copied(),
        ) else {
            return Ok(false);
        };
        calls.push(Callsite {
            id: id as u64,
            source,
            source_line: call.source_line,
            target,
            target_line: call.target_line,
            callee: call.callee,
            kind: call.kind,
            analyzer: call.analyzer,
            confidence: call.confidence,
        });
    }
    atlas.calls = calls;
    atlas.report = cached.report;
    atlas.report.cache_hit = true;
    Ok(true)
}

fn save_cached_analysis(atlas: &mut Atlas) -> Result<()> {
    let Some(path) = analysis_cache_path(atlas) else {
        return Ok(());
    };
    let calls = atlas
        .calls
        .iter()
        .map(|call| CachedCallsite {
            source_path: atlas.nodes[call.source].path.clone(),
            source_line: call.source_line,
            target_path: atlas.nodes[call.target].path.clone(),
            target_line: call.target_line,
            callee: call.callee.clone(),
            kind: call.kind.clone(),
            analyzer: call.analyzer.clone(),
            confidence: call.confidence,
        })
        .collect();
    let mut report = atlas.report.clone();
    report.cache_hit = false;
    let payload = CachedAnalysis {
        version: CACHE_VERSION,
        calls,
        report,
    };
    if let Some(parent) = path.parent() {
        if let Err(error) = fs::create_dir_all(parent) {
            atlas.report.warnings.push(format!(
                "could not create analysis cache directory: {error}"
            ));
            return Ok(());
        }
    }
    if let Err(error) = fs::write(&path, serde_json::to_vec(&payload)?) {
        atlas
            .report
            .warnings
            .push(format!("could not write analysis cache: {error}"));
    }
    Ok(())
}

fn analysis_cache_path(atlas: &Atlas) -> Option<PathBuf> {
    if atlas.dirty {
        return None;
    }
    let output = Command::new("git")
        .args(["rev-parse", "--git-dir"])
        .current_dir(&atlas.root_path)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let git_dir = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    let git_dir = if git_dir.is_absolute() {
        git_dir
    } else {
        atlas.root_path.join(git_dir)
    };
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in atlas
        .revision
        .as_bytes()
        .iter()
        .copied()
        .chain(atlas.files().flat_map(|node| {
            node.path
                .as_bytes()
                .iter()
                .copied()
                .chain(std::iter::once(0))
        }))
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    for tool in [
        "elixir".to_owned(),
        rust_analyzer_binary(),
        scip_typescript_binary(atlas),
    ] {
        for byte in command_version(&tool).bytes().chain(std::iter::once(0)) {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    Some(
        git_dir
            .join("code-atlas")
            .join(format!("analysis-v{CACHE_VERSION}-{hash:016x}.json")),
    )
}

fn command_version(binary: &str) -> String {
    let output = Command::new(binary).arg("--version").output();
    match output {
        Ok(output) if output.status.success() => format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
        _ => "unavailable".to_owned(),
    }
}

fn analyze_elixir_scripts(
    atlas: &mut Atlas,
    module_files: &HashMap<String, NodeId>,
    next_id: &mut u64,
) -> Result<()> {
    if coverage_mut(atlas, "elixir").files_discovered == 0 {
        return Ok(());
    }
    match analyze_elixir_scripts_ast(atlas, module_files, next_id) {
        Ok(true) => {
            let coverage = coverage_mut(atlas, "elixir");
            coverage.provider = "mix-xref + script-ast".to_owned();
            coverage.fidelity = "compiler-backed project files; compiler-parsed scripts".to_owned();
            Ok(())
        }
        Ok(false) => {
            let coverage = coverage_mut(atlas, "elixir");
            coverage.provider = "mix-xref + script-static".to_owned();
            analyze_elixir_scripts_fallback(atlas, module_files, next_id)
        }
        Err(error) => {
            atlas.report.warnings.push(format!(
                "Elixir AST script analysis failed; scripts used the regex fallback: {error:#}"
            ));
            let coverage = coverage_mut(atlas, "elixir");
            coverage.provider = "mix-xref + script-static".to_owned();
            analyze_elixir_scripts_fallback(atlas, module_files, next_id)
        }
    }
}

fn analyze_elixir_scripts_ast(
    atlas: &mut Atlas,
    module_files: &HashMap<String, NodeId>,
    next_id: &mut u64,
) -> Result<bool> {
    if !command_exists("elixir") {
        return Ok(false);
    }
    let scripts: Vec<_> = atlas
        .nodes
        .iter()
        .filter(|node| node.is_file() && node.path.ends_with(".exs"))
        .map(|node| (node.id, node.path.clone()))
        .collect();
    if scripts.is_empty() {
        return Ok(true);
    }
    let source_ids: HashMap<_, _> = scripts
        .iter()
        .map(|(id, path)| (path.clone(), *id))
        .collect();
    let mut child = Command::new("elixir")
        .args(["-e", ELIXIR_SCRIPT_EXPRESSION])
        .current_dir(&atlas.root_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    {
        let mut stdin = child.stdin.take().expect("piped stdin");
        for (_, path) in &scripts {
            writeln!(stdin, "{path}")?;
        }
    }
    let output = child.wait_with_output()?;
    if !output.status.success() {
        anyhow::bail!("{}", String::from_utf8_lossy(&output.stderr).trim());
    }
    let mut current = None;
    let mut failed = false;
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if let Some(path) = line.strip_prefix(FILE_MARKER) {
            current = source_ids.get(path).copied();
            failed = false;
            continue;
        }
        if let Some(error) = line.strip_prefix(ERROR_MARKER) {
            atlas
                .report
                .warnings
                .push(format!("could not parse Elixir script: {error}"));
            failed = true;
            continue;
        }
        if line == END_MARKER {
            if current.is_some() && !failed {
                atlas.report.elixir_script_files_scanned += 1;
                coverage_mut(atlas, "elixir").files_analyzed += 1;
            }
            current = None;
            continue;
        }
        let Some(payload) = line.strip_prefix(SCRIPT_CALL_MARKER) else {
            continue;
        };
        let Some(source) = current else {
            continue;
        };
        let mut fields = payload.splitn(4, '\t');
        let Some(source_line) = fields.next().and_then(|value| value.parse::<u32>().ok()) else {
            continue;
        };
        let Some(module) = fields.next() else {
            continue;
        };
        let Some(function) = fields.next() else {
            continue;
        };
        let Some(arity) = fields.next().and_then(|value| value.parse::<u32>().ok()) else {
            continue;
        };
        let Some(target) = module_files.get(module).copied() else {
            atlas.report.unresolved_calls += 1;
            coverage_mut(atlas, "elixir").callsites_unresolved += 1;
            continue;
        };
        atlas.calls.push(Callsite {
            id: *next_id,
            source,
            source_line,
            target,
            target_line: definition_line(
                &atlas.root_path.join(&atlas.nodes[target].path),
                function,
            ),
            callee: format!("{module}.{function}/{arity}"),
            kind: "syntax".to_owned(),
            analyzer: "elixir-script-ast".to_owned(),
            confidence: 0.75,
        });
        *next_id += 1;
        atlas.report.resolved_calls += 1;
        coverage_mut(atlas, "elixir").callsites_resolved += 1;
    }
    Ok(true)
}

fn analyze_elixir_scripts_fallback(
    atlas: &mut Atlas,
    module_files: &HashMap<String, NodeId>,
    next_id: &mut u64,
) -> Result<()> {
    let alias_pattern =
        Regex::new(r"^\s*alias\s+([A-Z][A-Za-z0-9_.]*)(?:\s*,\s*as:\s*([A-Z][A-Za-z0-9_]*))?")?;
    let call_pattern = Regex::new(r"\b([A-Z][A-Za-z0-9_.]*)\.([a-z_][A-Za-z0-9_!?]*)\s*\(")?;
    let scripts: Vec<_> = atlas
        .nodes
        .iter()
        .filter(|node| node.is_file() && node.path.ends_with(".exs"))
        .map(|node| (node.id, node.path.clone()))
        .collect();
    for (source, path) in scripts {
        let contents = fs::read_to_string(atlas.root_path.join(&path)).unwrap_or_default();
        let mut aliases = HashMap::new();
        for line in contents.lines() {
            if let Some(captures) = alias_pattern.captures(line) {
                let full = captures[1].to_owned();
                let short = captures
                    .get(2)
                    .map(|value| value.as_str())
                    .or_else(|| full.rsplit('.').next())
                    .unwrap_or(&full)
                    .to_owned();
                aliases.insert(short, full);
            }
        }
        atlas.report.elixir_script_files_scanned += 1;
        coverage_mut(atlas, "elixir").files_analyzed += 1;
        for (index, line) in contents.lines().enumerate() {
            let code = line.split('#').next().unwrap_or("");
            for captures in call_pattern.captures_iter(code) {
                let written_module = &captures[1];
                let module = module_files
                    .contains_key(written_module)
                    .then(|| written_module.to_owned())
                    .or_else(|| aliases.get(written_module).cloned());
                let Some(module) = module else {
                    continue;
                };
                let Some(target) = module_files.get(&module).copied() else {
                    continue;
                };
                let function = captures[2].to_owned();
                atlas.calls.push(Callsite {
                    id: *next_id,
                    source,
                    source_line: index as u32 + 1,
                    target,
                    target_line: definition_line(
                        &atlas.root_path.join(&atlas.nodes[target].path),
                        &function,
                    ),
                    callee: format!("{module}.{function}/?"),
                    kind: "syntax".to_owned(),
                    analyzer: "elixir-script-static".to_owned(),
                    confidence: 0.55,
                });
                *next_id += 1;
                atlas.report.resolved_calls += 1;
                coverage_mut(atlas, "elixir").callsites_resolved += 1;
            }
        }
    }
    Ok(())
}

fn analyze_typescript_calls(atlas: &mut Atlas, next_id: &mut u64) -> Result<bool> {
    let file_count = atlas
        .nodes
        .iter()
        .filter(|node| matches!(node.language.as_str(), "javascript" | "typescript"))
        .count();
    if file_count == 0 {
        return Ok(true);
    }
    let cacheable = match generate_typescript_scip(atlas) {
        Ok(Some(indexes)) => {
            let stats = ingest_scip_calls(atlas, &indexes, &["javascript", "typescript"], next_id)?;
            for language in ["javascript", "typescript"] {
                let coverage = coverage_mut(atlas, language);
                coverage.provider = "scip-typescript".to_owned();
                coverage.fidelity = "compiler-backed semantic symbols".to_owned();
                coverage.files_analyzed =
                    stats.files_by_language.get(language).copied().unwrap_or(0);
                coverage.callsites_resolved = stats
                    .resolved_by_language
                    .get(language)
                    .copied()
                    .unwrap_or(0);
                coverage.callsites_unresolved = stats
                    .unresolved_by_language
                    .get(language)
                    .copied()
                    .unwrap_or(0);
            }
            atlas.report.typescript_files_scanned =
                stats.files_by_language.values().copied().sum::<usize>();
            atlas.report.typescript_calls =
                stats.resolved_by_language.values().copied().sum::<usize>();
            atlas.report.resolved_calls += atlas.report.typescript_calls;
            atlas.report.unresolved_calls += stats
                .unresolved_by_language
                .values()
                .copied()
                .sum::<usize>();
            return Ok(true);
        }
        Ok(None) => {
            atlas.report.warnings.push(
                "scip-typescript was not found; JavaScript/TypeScript used the relative-import fallback"
                    .to_owned(),
            );
            true
        }
        Err(error) => {
            atlas.report.warnings.push(format!(
                "scip-typescript indexing failed; JavaScript/TypeScript used the relative-import fallback: {error:#}"
            ));
            false
        }
    };
    for language in ["javascript", "typescript"] {
        let coverage = coverage_mut(atlas, language);
        coverage.provider = "relative-import-static".to_owned();
        coverage.fidelity = "conservative syntax fallback".to_owned();
    }
    analyze_typescript_calls_fallback(atlas, next_id)?;
    Ok(cacheable)
}

fn analyze_typescript_calls_fallback(atlas: &mut Atlas, next_id: &mut u64) -> Result<()> {
    let import_pattern = Regex::new(r#"^\s*import\s+(.+?)\s+from\s+["']([^"']+)["']\s*;?\s*$"#)?;
    let call_pattern =
        Regex::new(r"\b([A-Za-z_$][A-Za-z0-9_$]*)(?:\.([A-Za-z_$][A-Za-z0-9_$]*))?\s*\(")?;
    let files: Vec<_> = atlas
        .nodes
        .iter()
        .filter(|node| {
            node.is_file()
                && matches!(
                    Path::new(&node.path)
                        .extension()
                        .and_then(|value| value.to_str()),
                    Some("ts" | "tsx" | "mts" | "cts" | "js" | "jsx" | "mjs" | "cjs")
                )
        })
        .map(|node| (node.id, node.path.clone()))
        .collect();
    for (source, path) in files {
        atlas.report.typescript_files_scanned += 1;
        let language = atlas.nodes[source].language.clone();
        coverage_mut(atlas, &language).files_analyzed += 1;
        let contents = fs::read_to_string(atlas.root_path.join(&path)).unwrap_or_default();
        let mut bindings: HashMap<String, (NodeId, bool)> = HashMap::new();
        for line in contents.lines() {
            let Some(captures) = import_pattern.captures(line) else {
                continue;
            };
            let Some(target) = resolve_relative_module(atlas, &path, &captures[2]) else {
                continue;
            };
            let clause = captures[1].trim();
            if let Some(namespace) = clause.strip_prefix("* as ") {
                bindings.insert(namespace.trim().to_owned(), (target, true));
                continue;
            }
            let mut remainder = clause;
            if !remainder.starts_with('{') {
                let default = remainder.split([',', ' ']).next().unwrap_or("").trim();
                if !default.is_empty() {
                    bindings.insert(default.to_owned(), (target, false));
                }
                remainder = remainder
                    .split_once(',')
                    .map(|(_, rest)| rest)
                    .unwrap_or("");
            }
            if let Some(named) = remainder
                .trim()
                .strip_prefix('{')
                .and_then(|value| value.strip_suffix('}'))
            {
                for item in named.split(',') {
                    let mut parts = item.split_whitespace();
                    let original = parts.next().unwrap_or("");
                    let alias = match (parts.next(), parts.next()) {
                        (Some("as"), Some(alias)) => alias,
                        _ => original,
                    };
                    if !alias.is_empty() {
                        bindings.insert(alias.to_owned(), (target, false));
                    }
                }
            }
        }
        for (index, line) in contents.lines().enumerate() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("import ") || trimmed.starts_with("//") {
                continue;
            }
            let code = line.split("//").next().unwrap_or("");
            for captures in call_pattern.captures_iter(code) {
                let binding = &captures[1];
                let Some((target, namespace)) = bindings.get(binding).copied() else {
                    continue;
                };
                if namespace && captures.get(2).is_none() {
                    continue;
                }
                let function = captures
                    .get(2)
                    .map(|value| value.as_str())
                    .unwrap_or(binding);
                atlas.calls.push(Callsite {
                    id: *next_id,
                    source,
                    source_line: index as u32 + 1,
                    target,
                    target_line: None,
                    callee: format!("{}.{}", atlas.nodes[target].path, function),
                    kind: "runtime".to_owned(),
                    analyzer: "typescript-import-static".to_owned(),
                    confidence: 0.65,
                });
                *next_id += 1;
                atlas.report.resolved_calls += 1;
                atlas.report.typescript_calls += 1;
                coverage_mut(atlas, &language).callsites_resolved += 1;
            }
        }
    }
    Ok(())
}

fn resolve_relative_module(atlas: &Atlas, source_path: &str, import: &str) -> Option<NodeId> {
    if !import.starts_with('.') {
        return None;
    }
    let source_parent = Path::new(source_path)
        .parent()
        .unwrap_or_else(|| Path::new(""));
    let joined = source_parent.join(import);
    let mut components = Vec::new();
    for component in joined.components() {
        use std::path::Component;
        match component {
            Component::Normal(value) => components.push(value.to_string_lossy().to_string()),
            Component::ParentDir => {
                components.pop()?;
            }
            Component::CurDir => {}
            _ => return None,
        }
    }
    let base = components.join("/");
    let mut candidates = vec![base.clone()];
    for extension in ["ts", "tsx", "mts", "cts", "js", "jsx", "mjs", "cjs"] {
        candidates.push(format!("{base}.{extension}"));
        candidates.push(format!("{base}/index.{extension}"));
    }
    candidates
        .into_iter()
        .find_map(|candidate| atlas.path_to_id.get(&candidate).copied())
}

struct RootedScipIndex {
    root: String,
    index: ScipIndex,
}

#[derive(Default)]
struct ScipIngestStats {
    files_by_language: HashMap<String, usize>,
    resolved_by_language: HashMap<String, usize>,
    unresolved_by_language: HashMap<String, usize>,
}

fn generate_rust_scip(atlas: &Atlas) -> Result<Option<Vec<RootedScipIndex>>> {
    let binary = rust_analyzer_binary();
    if !command_exists(&binary) {
        return Ok(None);
    }
    let roots = outermost_project_roots(atlas, &["Cargo.toml"]);
    if roots.is_empty() {
        return Ok(None);
    }
    let temporary = tempfile::tempdir()?;
    let mut indexes = Vec::new();
    for (number, root) in roots.iter().enumerate() {
        let output_path = temporary.path().join(format!("rust-{number}.scip"));
        let project_path = if root.is_empty() {
            atlas.root_path.clone()
        } else {
            atlas.root_path.join(root)
        };
        let output = Command::new(&binary)
            .arg("scip")
            .arg(&project_path)
            .arg("--output")
            .arg(&output_path)
            .arg("--exclude-vendored-libraries")
            .current_dir(&atlas.root_path)
            .output()?;
        if !output.status.success() {
            anyhow::bail!("{}", String::from_utf8_lossy(&output.stderr).trim());
        }
        indexes.push(RootedScipIndex {
            root: root.clone(),
            index: decode_index(&fs::read(&output_path)?)?,
        });
    }
    Ok(Some(indexes))
}

fn generate_typescript_scip(atlas: &Atlas) -> Result<Option<Vec<RootedScipIndex>>> {
    let binary = scip_typescript_binary(atlas);
    if !command_exists(&binary) {
        return Ok(None);
    }
    let mut roots = outermost_project_roots(atlas, &["tsconfig.json"]);
    let infer_tsconfig = roots.is_empty();
    if roots.is_empty() {
        roots = outermost_project_roots(atlas, &["package.json"]);
    }
    if roots.is_empty() {
        return Ok(None);
    }
    let temporary = tempfile::tempdir()?;
    let mut indexes = Vec::new();
    for (number, root) in roots.iter().enumerate() {
        let output_path = temporary.path().join(format!("typescript-{number}.scip"));
        let project_path = if root.is_empty() {
            atlas.root_path.clone()
        } else {
            atlas.root_path.join(root)
        };
        let mut command = Command::new(&binary);
        command
            .arg("index")
            .arg("--output")
            .arg(&output_path)
            .current_dir(&project_path);
        if infer_tsconfig {
            command.arg("--infer-tsconfig");
        }
        let output = command.output()?;
        if !output.status.success() {
            anyhow::bail!("{}", String::from_utf8_lossy(&output.stderr).trim());
        }
        indexes.push(RootedScipIndex {
            root: root.clone(),
            index: decode_index(&fs::read(&output_path)?)?,
        });
    }
    Ok(Some(indexes))
}

fn rust_analyzer_binary() -> String {
    env::var("CODE_ATLAS_RUST_ANALYZER").unwrap_or_else(|_| "rust-analyzer".to_owned())
}

fn scip_typescript_binary(atlas: &Atlas) -> String {
    env::var("CODE_ATLAS_SCIP_TYPESCRIPT")
        .ok()
        .or_else(|| {
            let local = atlas.root_path.join("node_modules/.bin/scip-typescript");
            local.is_file().then(|| local.display().to_string())
        })
        .unwrap_or_else(|| "scip-typescript".to_owned())
}

fn command_exists(binary: &str) -> bool {
    Command::new(binary)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn outermost_project_roots(atlas: &Atlas, manifests: &[&str]) -> Vec<String> {
    let mut roots: Vec<_> = atlas
        .nodes
        .iter()
        .filter(|node| node.is_file())
        .filter(|node| manifests.iter().any(|manifest| node.name == *manifest))
        .map(|node| {
            Path::new(&node.path)
                .parent()
                .map(|parent| parent.to_string_lossy().trim_end_matches('/').to_owned())
                .unwrap_or_default()
        })
        .collect();
    roots.sort_by_key(|root| (root.matches('/').count(), root.clone()));
    let mut outermost = Vec::<String>::new();
    for root in roots {
        if outermost.iter().any(|parent| {
            parent.is_empty()
                || root == *parent
                || root
                    .strip_prefix(parent)
                    .is_some_and(|suffix| suffix.starts_with('/'))
        }) {
            continue;
        }
        outermost.push(root);
    }
    outermost
}

fn ingest_scip_calls(
    atlas: &mut Atlas,
    indexes: &[RootedScipIndex],
    languages: &[&str],
    next_id: &mut u64,
) -> Result<ScipIngestStats> {
    let mut documents = Vec::new();
    let mut analyzed_files: HashMap<String, HashSet<NodeId>> = HashMap::new();
    for rooted in indexes {
        for document in &rooted.index.documents {
            let Some(file) = scip_document_file(atlas, &rooted.root, document) else {
                continue;
            };
            let language = atlas.nodes[file].language.clone();
            if !languages.contains(&language.as_str()) {
                continue;
            }
            analyzed_files.entry(language).or_default().insert(file);
            documents.push((file, document));
        }
    }

    let mut definitions: HashMap<&str, (NodeId, u32)> = HashMap::new();
    for (file, document) in &documents {
        for occurrence in &document.occurrences {
            if occurrence.symbol.is_empty() || occurrence.symbol_roles & 1 == 0 {
                continue;
            }
            if let Some(range) = occurrence.range {
                definitions
                    .entry(&occurrence.symbol)
                    .or_insert((*file, range.start_line + 1));
            }
        }
    }

    let mut stats = ScipIngestStats::default();
    for (language, files) in analyzed_files {
        stats.files_by_language.insert(language, files.len());
    }
    let mut seen = HashSet::new();
    let mut source_cache: HashMap<NodeId, String> = HashMap::new();
    for (source, document) in documents {
        let language = atlas.nodes[source].language.clone();
        let contents = source_cache.entry(source).or_insert_with(|| {
            fs::read_to_string(atlas.root_path.join(&atlas.nodes[source].path)).unwrap_or_default()
        });
        for occurrence in &document.occurrences {
            if occurrence.symbol.is_empty() || occurrence.symbol_roles & (1 | 2 | 16 | 32) != 0 {
                continue;
            }
            let Some(range) = occurrence.range else {
                continue;
            };
            let Some(callee) = occurrence_call_text(contents, range, document.position_encoding)
            else {
                continue;
            };
            let Some((target, target_line)) = definitions.get(occurrence.symbol.as_str()).copied()
            else {
                *stats
                    .unresolved_by_language
                    .entry(language.clone())
                    .or_default() += 1;
                continue;
            };
            let identity = (
                source,
                range.start_line,
                range.start_character,
                target,
                target_line,
                occurrence.symbol.clone(),
            );
            if !seen.insert(identity) {
                continue;
            }
            atlas.calls.push(Callsite {
                id: *next_id,
                source,
                source_line: range.start_line + 1,
                target,
                target_line: Some(target_line),
                callee,
                kind: "semantic".to_owned(),
                analyzer: if language == "rust" {
                    "rust-analyzer-scip".to_owned()
                } else {
                    "scip-typescript".to_owned()
                },
                confidence: 0.98,
            });
            *next_id += 1;
            *stats
                .resolved_by_language
                .entry(language.clone())
                .or_default() += 1;
        }
    }
    Ok(stats)
}

fn scip_document_file(atlas: &Atlas, root: &str, document: &ScipDocument) -> Option<NodeId> {
    let relative = document
        .relative_path
        .replace('\\', "/")
        .trim_start_matches("./")
        .to_owned();
    let rooted = if root.is_empty() {
        relative.clone()
    } else {
        format!("{root}/{relative}")
    };
    atlas
        .path_to_id
        .get(&rooted)
        .or_else(|| atlas.path_to_id.get(&relative))
        .copied()
}

fn occurrence_call_text(contents: &str, range: ScipRange, encoding: i32) -> Option<String> {
    if range.start_line != range.end_line {
        return None;
    }
    let line = contents.lines().nth(range.start_line as usize)?;
    let start = encoded_column_to_byte(line, range.start_character, encoding)?;
    let end = encoded_column_to_byte(line, range.end_character, encoding)?;
    let callee = line.get(start..end)?.trim();
    if callee.is_empty() || !suffix_is_call(line.get(end..)?) {
        return None;
    }
    Some(callee.to_owned())
}

fn encoded_column_to_byte(line: &str, column: u32, encoding: i32) -> Option<usize> {
    if encoding != 2 {
        let byte = column as usize;
        return (byte <= line.len() && line.is_char_boundary(byte)).then_some(byte);
    }
    let mut utf16 = 0_u32;
    for (byte, character) in line.char_indices() {
        if utf16 == column {
            return Some(byte);
        }
        utf16 += character.len_utf16() as u32;
        if utf16 > column {
            return None;
        }
    }
    (utf16 == column).then_some(line.len())
}

fn suffix_is_call(suffix: &str) -> bool {
    let mut rest = suffix.trim_start();
    if let Some(optional) = rest.strip_prefix("?.") {
        rest = optional.trim_start();
    }
    if let Some(macro_call) = rest.strip_prefix('!') {
        rest = macro_call.trim_start();
    }
    if let Some(generic) = rest
        .strip_prefix("::")
        .and_then(|value| value.strip_prefix('<'))
    {
        let Some(after_generics) = skip_balanced_angles(generic) else {
            return false;
        };
        rest = after_generics;
    } else if let Some(generic) = rest.strip_prefix('<') {
        let Some(after_generics) = skip_balanced_angles(generic) else {
            return false;
        };
        rest = after_generics;
    }
    rest.trim_start().starts_with('(')
}

fn skip_balanced_angles(value: &str) -> Option<&str> {
    let mut depth = 1_usize;
    for (offset, character) in value.char_indices() {
        match character {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                if depth == 0 {
                    return value.get(offset + character.len_utf8()..);
                }
            }
            _ => {}
        }
    }
    None
}

fn analyze_gleam_calls(atlas: &mut Atlas, next_id: &mut u64) -> Result<()> {
    let import_pattern = Regex::new(
        r"^\s*import\s+([a-z][A-Za-z0-9_/]*)(?:\s+as\s+([a-z][A-Za-z0-9_]*))?(?:\.\{([^}]*)\})?",
    )?;
    let qualified_call = Regex::new(r"\b([a-z][A-Za-z0-9_]*)\.([a-z][A-Za-z0-9_]*)\s*\(")?;
    let unqualified_call = Regex::new(r"\b([a-z][A-Za-z0-9_]*)\s*\(")?;
    let gleam_files: Vec<_> = atlas
        .nodes
        .iter()
        .filter(|node| node.is_file() && node.path.ends_with(".gleam"))
        .map(|node| (node.id, node.path.clone()))
        .collect();
    let modules: HashMap<_, _> = gleam_files
        .iter()
        .filter_map(|(id, path)| gleam_module_name(path).map(|name| (name, *id)))
        .collect();
    for (source, path) in gleam_files {
        atlas.report.gleam_files_scanned += 1;
        coverage_mut(atlas, "gleam").files_analyzed += 1;
        let contents = fs::read_to_string(atlas.root_path.join(&path)).unwrap_or_default();
        let mut module_aliases = HashMap::new();
        let mut functions = HashMap::new();
        for line in contents.lines() {
            let Some(captures) = import_pattern.captures(line) else {
                continue;
            };
            let module = captures[1].to_owned();
            let Some(target) = modules.get(&module).copied() else {
                continue;
            };
            let alias = captures
                .get(2)
                .map(|value| value.as_str())
                .or_else(|| module.rsplit('/').next())
                .unwrap_or(&module);
            module_aliases.insert(alias.to_owned(), target);
            if let Some(selected) = captures.get(3) {
                for item in selected.as_str().split(',') {
                    let mut parts = item.split_whitespace();
                    let original = parts.next().unwrap_or("");
                    let alias = match (parts.next(), parts.next()) {
                        (Some("as"), Some(alias)) => alias,
                        _ => original,
                    };
                    if !alias.is_empty() {
                        functions.insert(alias.to_owned(), target);
                    }
                }
            }
        }
        for (index, line) in contents.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            for captures in qualified_call.captures_iter(code) {
                let Some(target) = module_aliases.get(&captures[1]).copied() else {
                    continue;
                };
                push_gleam_call(atlas, next_id, source, index, target, &captures[2]);
            }
            for captures in unqualified_call.captures_iter(code) {
                let Some(target) = functions.get(&captures[1]).copied() else {
                    continue;
                };
                push_gleam_call(atlas, next_id, source, index, target, &captures[1]);
            }
        }
    }
    Ok(())
}

fn gleam_module_name(path: &str) -> Option<String> {
    let without_extension = path.strip_suffix(".gleam")?;
    without_extension
        .split_once("/src/")
        .map(|(_, module)| module.to_owned())
        .or_else(|| without_extension.strip_prefix("src/").map(str::to_owned))
}

fn push_gleam_call(
    atlas: &mut Atlas,
    next_id: &mut u64,
    source: NodeId,
    line_index: usize,
    target: NodeId,
    function: &str,
) {
    atlas.calls.push(Callsite {
        id: *next_id,
        source,
        source_line: line_index as u32 + 1,
        target,
        target_line: None,
        callee: format!("{}.{}", atlas.nodes[target].path, function),
        kind: "runtime".to_owned(),
        analyzer: "gleam-import-static".to_owned(),
        confidence: 0.70,
    });
    *next_id += 1;
    atlas.report.resolved_calls += 1;
    atlas.report.gleam_calls += 1;
    coverage_mut(atlas, "gleam").callsites_resolved += 1;
}

#[derive(Debug, Clone, Copy)]
struct RustDefinition {
    file: NodeId,
    line: u32,
}

#[derive(Default)]
struct RustDefinitionVisitor {
    definitions: Vec<(String, u32)>,
}

impl<'ast> Visit<'ast> for RustDefinitionVisitor {
    fn visit_item_fn(&mut self, function: &'ast syn::ItemFn) {
        self.definitions.push((
            function.sig.ident.to_string(),
            span_line(function.sig.ident.span()),
        ));
        syn::visit::visit_item_fn(self, function);
    }

    fn visit_impl_item_fn(&mut self, function: &'ast syn::ImplItemFn) {
        self.definitions.push((
            function.sig.ident.to_string(),
            span_line(function.sig.ident.span()),
        ));
        syn::visit::visit_impl_item_fn(self, function);
    }
}

#[derive(Debug)]
struct RustCallCandidate {
    name: String,
    line: u32,
    qualified: bool,
}

#[derive(Default)]
struct RustCallVisitor {
    calls: Vec<RustCallCandidate>,
}

impl<'ast> Visit<'ast> for RustCallVisitor {
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if let Expr::Path(path) = call.func.as_ref() {
            if let Some(segment) = path.path.segments.last() {
                self.calls.push(RustCallCandidate {
                    name: segment.ident.to_string(),
                    line: span_line(call.span()),
                    qualified: path.path.segments.len() > 1,
                });
            }
        }
        syn::visit::visit_expr_call(self, call);
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        self.calls.push(RustCallCandidate {
            name: call.method.to_string(),
            line: span_line(call.span()),
            qualified: false,
        });
        syn::visit::visit_expr_method_call(self, call);
    }
}

fn analyze_rust_calls(atlas: &mut Atlas, next_id: &mut u64) -> Result<bool> {
    let file_count = atlas
        .nodes
        .iter()
        .filter(|node| node.is_file() && node.language == "rust")
        .count();
    if file_count == 0 {
        return Ok(true);
    }
    let cacheable = match generate_rust_scip(atlas) {
        Ok(Some(indexes)) => {
            let stats = ingest_scip_calls(atlas, &indexes, &["rust"], next_id)?;
            let files_analyzed = stats.files_by_language.get("rust").copied().unwrap_or(0);
            let callsites_resolved = stats.resolved_by_language.get("rust").copied().unwrap_or(0);
            let callsites_unresolved = stats
                .unresolved_by_language
                .get("rust")
                .copied()
                .unwrap_or(0);
            {
                let coverage = coverage_mut(atlas, "rust");
                coverage.provider = "rust-analyzer-scip".to_owned();
                coverage.fidelity = "compiler-backed semantic symbols".to_owned();
                coverage.files_analyzed = files_analyzed;
                coverage.callsites_resolved = callsites_resolved;
                coverage.callsites_unresolved = callsites_unresolved;
            }
            atlas.report.rust_files_scanned = files_analyzed;
            atlas.report.rust_calls = callsites_resolved;
            atlas.report.resolved_calls += callsites_resolved;
            atlas.report.unresolved_calls += callsites_unresolved;
            return Ok(true);
        }
        Ok(None) => {
            atlas.report.warnings.push(
                "rust-analyzer was not found; Rust used the globally-unique syntax fallback"
                    .to_owned(),
            );
            true
        }
        Err(error) => {
            atlas.report.warnings.push(format!(
                "rust-analyzer SCIP indexing failed; Rust used the globally-unique syntax fallback: {error:#}"
            ));
            false
        }
    };
    let coverage = coverage_mut(atlas, "rust");
    coverage.provider = "rust-syntax-unique".to_owned();
    coverage.fidelity = "conservative syntax fallback".to_owned();
    analyze_rust_calls_fallback(atlas, next_id)?;
    Ok(cacheable)
}

fn analyze_rust_calls_fallback(atlas: &mut Atlas, next_id: &mut u64) -> Result<()> {
    let rust_files: Vec<(NodeId, String)> = atlas
        .nodes
        .iter()
        .filter(|node| node.is_file() && node.path.ends_with(".rs"))
        .map(|node| (node.id, node.path.clone()))
        .collect();
    if rust_files.is_empty() {
        return Ok(());
    }

    let mut syntax_trees = Vec::new();
    let mut unique_definitions: HashMap<String, Option<RustDefinition>> = HashMap::new();
    for (file, path) in &rust_files {
        let source = fs::read_to_string(atlas.root_path.join(path)).unwrap_or_default();
        let syntax = match syn::parse_file(&source) {
            Ok(syntax) => syntax,
            Err(error) => {
                atlas
                    .report
                    .warnings
                    .push(format!("could not parse Rust file {path}: {error}"));
                continue;
            }
        };
        atlas.report.rust_files_scanned += 1;
        coverage_mut(atlas, "rust").files_analyzed += 1;
        let mut visitor = RustDefinitionVisitor::default();
        visitor.visit_file(&syntax);
        for (name, line) in visitor.definitions {
            unique_definitions
                .entry(name)
                .and_modify(|definition| *definition = None)
                .or_insert(Some(RustDefinition { file: *file, line }));
        }
        syntax_trees.push((*file, syntax));
    }

    for (source, syntax) in syntax_trees {
        let mut visitor = RustCallVisitor::default();
        visitor.visit_file(&syntax);
        for candidate in visitor.calls {
            if !candidate.qualified {
                atlas.report.unresolved_calls += 1;
                coverage_mut(atlas, "rust").callsites_unresolved += 1;
                continue;
            }
            let Some(Some(target)) = unique_definitions.get(&candidate.name) else {
                atlas.report.unresolved_calls += 1;
                coverage_mut(atlas, "rust").callsites_unresolved += 1;
                continue;
            };
            atlas.calls.push(Callsite {
                id: *next_id,
                source,
                source_line: candidate.line,
                target: target.file,
                target_line: Some(target.line),
                callee: candidate.name,
                kind: "syntax".to_owned(),
                analyzer: "rust-syntax-unique".to_owned(),
                confidence: 0.70,
            });
            *next_id += 1;
            atlas.report.resolved_calls += 1;
            atlas.report.rust_calls += 1;
            coverage_mut(atlas, "rust").callsites_resolved += 1;
        }
    }
    Ok(())
}

fn span_line(span: proc_macro2::Span) -> u32 {
    span.start().line.max(1) as u32
}

pub fn parse_xref_line(line: &str) -> Option<XrefCall> {
    let pattern = Regex::new(
        r"^.+?:(\d+): (?:import )?call ([A-Z][A-Za-z0-9_.]*)\.([a-zA-Z0-9_!?]+)/([0-9]+) \(([^)]+)\)$",
    )
    .expect("valid xref regex");
    let captures = pattern.captures(line)?;
    Some(XrefCall {
        source_line: captures.get(1)?.as_str().parse().ok()?,
        module: captures.get(2)?.as_str().to_owned(),
        function: captures.get(3)?.as_str().to_owned(),
        arity: captures.get(4)?.as_str().parse().ok()?,
        kind: captures.get(5)?.as_str().to_owned(),
    })
}

fn elixir_module_files(atlas: &Atlas) -> Result<HashMap<String, NodeId>> {
    let pattern = Regex::new(r"(?m)^\s*defmodule\s+([A-Z][A-Za-z0-9_.]*)\s+do\b")?;
    let mut modules = HashMap::new();
    for node in atlas.nodes.iter().filter(|node| {
        node.is_file() && (node.path.ends_with(".ex") || node.path.ends_with(".exs"))
    }) {
        let contents = fs::read_to_string(atlas.root_path.join(&node.path)).unwrap_or_default();
        for captures in pattern.captures_iter(&contents) {
            modules.insert(captures[1].to_owned(), node.id);
        }
    }
    Ok(modules)
}

fn rustler_targets(
    atlas: &Atlas,
    modules: &HashMap<String, NodeId>,
) -> Result<HashMap<(String, String), RustlerTarget>> {
    let init_pattern = Regex::new(r#"rustler::init!\(\s*"Elixir\.([^"]+)""#)?;
    let nif_pattern =
        Regex::new(r#"(?s)#\[rustler::nif(?:\(([^]]*)\))?\]\s*(?:pub\s+)?fn\s+([a-zA-Z0-9_]+)"#)?;
    let name_pattern = Regex::new(r#"name\s*=\s*"([^"]+)""#)?;

    let cargo_roots = project_roots_for_manifest(atlas, "Cargo.toml");
    let mut nif_modules = HashMap::new();
    for node in atlas
        .nodes
        .iter()
        .filter(|node| node.is_file() && node.path.ends_with(".rs"))
    {
        let contents = fs::read_to_string(atlas.root_path.join(&node.path)).unwrap_or_default();
        if let Some(captures) = init_pattern.captures(&contents) {
            let root = nearest_project_root(&node.path, &cargo_roots)
                .unwrap_or("")
                .to_owned();
            nif_modules.insert(root, captures[1].to_owned());
        }
    }

    if nif_modules.is_empty() {
        return Ok(HashMap::new());
    }

    let mut targets = HashMap::new();
    for node in atlas
        .nodes
        .iter()
        .filter(|node| node.is_file() && node.path.ends_with(".rs"))
    {
        let Some(nif_module) =
            nearest_project_root(&node.path, &cargo_roots).and_then(|root| nif_modules.get(root))
        else {
            continue;
        };
        if !modules.contains_key(nif_module) {
            continue;
        }
        let contents = fs::read_to_string(atlas.root_path.join(&node.path)).unwrap_or_default();
        for captures in nif_pattern.captures_iter(&contents) {
            let rust_name = captures.get(2).expect("function capture").as_str();
            let args = captures.get(1).map(|value| value.as_str()).unwrap_or("");
            let exported_name = name_pattern
                .captures(args)
                .and_then(|captures| captures.get(1))
                .map(|value| value.as_str())
                .unwrap_or(rust_name);
            let line = contents[..captures.get(0).expect("whole capture").start()]
                .lines()
                .count() as u32
                + 1;
            targets.insert(
                (nif_module.clone(), exported_name.to_owned()),
                RustlerTarget {
                    file: node.id,
                    line: Some(line),
                },
            );
        }
    }
    Ok(targets)
}

fn project_roots_for_manifest(atlas: &Atlas, manifest: &str) -> Vec<String> {
    let mut roots: Vec<_> = atlas
        .nodes
        .iter()
        .filter(|node| node.is_file() && node.name == manifest)
        .map(|node| {
            Path::new(&node.path)
                .parent()
                .map(|parent| parent.to_string_lossy().trim_end_matches('/').to_owned())
                .unwrap_or_default()
        })
        .collect();
    roots.sort_by_key(|root| (root.matches('/').count(), root.clone()));
    roots.dedup();
    roots
}

fn definition_line(path: &Path, function: &str) -> Option<u32> {
    let contents = fs::read_to_string(path).ok()?;
    let escaped = regex::escape(function);
    let pattern = Regex::new(&format!(
        r"(?m)^\s*(?:defp?|defmacrop?|defguardp?|defdelegate)\s+{escaped}(?:\s*\(|\s*,|\s+do|\s+when)"
    ))
    .ok()?;
    let found = pattern.find(&contents)?;
    Some(contents[..found.start()].lines().count() as u32 + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assigns_files_to_the_nearest_mix_project_root() {
        let roots = vec![
            String::new(),
            "apps/alpha".to_owned(),
            "apps/alpha/vendor/nested".to_owned(),
            "apps/beta".to_owned(),
        ];

        assert_eq!(nearest_project_root("lib/root.ex", &roots), Some(""));
        assert_eq!(
            nearest_project_root("apps/alpha/lib/alpha.ex", &roots),
            Some("apps/alpha")
        );
        assert_eq!(
            nearest_project_root("apps/alpha/vendor/nested/lib/nested.ex", &roots),
            Some("apps/alpha/vendor/nested")
        );
        assert_eq!(
            nearest_project_root("apps/beta/lib/beta.ex", &roots),
            Some("apps/beta")
        );
    }
    use std::collections::HashMap;

    use crate::model::{AnalyzerReport, BuildTimings, Node, NodeKind, Rect};

    #[test]
    fn parses_individual_xref_callsite() {
        assert_eq!(
            parse_xref_line("lib/wasmex.ex:252: call Wasmex.Module.compile/2 (runtime)"),
            Some(XrefCall {
                source_line: 252,
                module: "Wasmex.Module".to_owned(),
                function: "compile".to_owned(),
                arity: 2,
                kind: "runtime".to_owned(),
            })
        );
    }

    #[test]
    fn finds_elixir_function_macro_guard_and_delegate_targets() {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("targets.ex");
        fs::write(
            &source,
            "def run(), do: :ok\ndefmacro build(value), do: value\ndefguard valid(value) when is_atom(value)\ndefdelegate fetch(value), to: Other\n",
        )
        .unwrap();

        assert_eq!(definition_line(&source, "run"), Some(1));
        assert_eq!(definition_line(&source, "build"), Some(2));
        assert_eq!(definition_line(&source, "valid"), Some(3));
        assert_eq!(definition_line(&source, "fetch"), Some(4));
    }

    #[test]
    fn ignores_non_call_xref_records() {
        assert!(parse_xref_line("lib/a.ex:2: alias A (runtime)").is_none());
        assert!(parse_xref_line("lib/a.ex:3: struct A (export)").is_none());
    }

    #[test]
    fn excludes_same_file_calls_without_aggregating_cross_file_calls() {
        let directory = tempfile::tempdir().unwrap();
        let mut atlas = Atlas {
            root_path: directory.path().to_path_buf(),
            revision: "test".to_owned(),
            dirty: true,
            excluded_test_files: 0,
            excluded_hidden_files: 0,
            excluded_custom_files: 0,
            excluded_paths: Vec::new(),
            nodes: vec![
                Node {
                    id: 0,
                    parent: None,
                    children: vec![1, 2],
                    kind: NodeKind::Directory,
                    name: "fixture".to_owned(),
                    path: String::new(),
                    depth: 0,
                    bytes: 2,
                    loc: 2,
                    commits: 2,
                    weight: 2.0,
                    language: "directory".to_owned(),
                    rect: Rect::default(),
                },
                rust_node(1, "a.rs"),
                rust_node(2, "b.rs"),
            ],
            calls: vec![
                Callsite {
                    id: 11,
                    source: 1,
                    source_line: 3,
                    target: 1,
                    target_line: Some(8),
                    callee: "a::local".to_owned(),
                    kind: "syntax".to_owned(),
                    analyzer: "fixture".to_owned(),
                    confidence: 1.0,
                },
                Callsite {
                    id: 12,
                    source: 1,
                    source_line: 4,
                    target: 2,
                    target_line: Some(9),
                    callee: "b::remote".to_owned(),
                    kind: "syntax".to_owned(),
                    analyzer: "fixture".to_owned(),
                    confidence: 1.0,
                },
            ],
            path_to_id: HashMap::new(),
            report: AnalyzerReport::default(),
            timings: BuildTimings::default(),
        };

        exclude_same_file_calls(&mut atlas);

        assert_eq!(atlas.calls.len(), 1);
        assert_eq!(atlas.calls[0].id, 12);
        assert_eq!(atlas.report.same_file_calls_excluded, 1);
    }

    #[test]
    fn resolves_unique_cross_file_rust_call_without_aggregation() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(
            directory.path().join("a.rs"),
            "pub fn alpha() { b::beta(); }\n",
        )
        .unwrap();
        fs::write(directory.path().join("b.rs"), "pub fn beta() {}\n").unwrap();
        let mut atlas = Atlas {
            root_path: directory.path().to_path_buf(),
            revision: "test".to_owned(),
            dirty: false,
            excluded_test_files: 0,
            excluded_hidden_files: 0,
            excluded_custom_files: 0,
            excluded_paths: Vec::new(),
            nodes: vec![
                Node {
                    id: 0,
                    parent: None,
                    children: vec![1, 2],
                    kind: NodeKind::Directory,
                    name: "fixture".to_owned(),
                    path: String::new(),
                    depth: 0,
                    bytes: 0,
                    loc: 0,
                    commits: 0,
                    weight: 2.0,
                    language: "directory".to_owned(),
                    rect: Rect::default(),
                },
                rust_node(1, "a.rs"),
                rust_node(2, "b.rs"),
            ],
            calls: Vec::new(),
            path_to_id: HashMap::new(),
            report: AnalyzerReport::default(),
            timings: BuildTimings::default(),
        };
        let mut next_id = 0;

        initialize_language_coverage(&mut atlas);
        analyze_rust_calls_fallback(&mut atlas, &mut next_id).unwrap();

        assert_eq!(atlas.report.rust_files_scanned, 2);
        assert_eq!(atlas.report.rust_calls, 1);
        assert_eq!(atlas.calls.len(), 1);
        assert_eq!(atlas.calls[0].source, 1);
        assert_eq!(atlas.calls[0].target, 2);
    }

    #[test]
    fn clean_revision_cache_round_trips_calls_by_path() {
        let directory = tempfile::tempdir().unwrap();
        assert!(
            Command::new("git")
                .args(["init", "--quiet"])
                .arg(directory.path())
                .status()
                .unwrap()
                .success()
        );
        let nodes = vec![
            Node {
                id: 0,
                parent: None,
                children: vec![1, 2],
                kind: NodeKind::Directory,
                name: "fixture".to_owned(),
                path: String::new(),
                depth: 0,
                bytes: 2,
                loc: 2,
                commits: 2,
                weight: 2.0,
                language: "directory".to_owned(),
                rect: Rect::default(),
            },
            rust_node(1, "a.rs"),
            rust_node(2, "b.rs"),
        ];
        let mut path_to_id = HashMap::new();
        path_to_id.insert("a.rs".to_owned(), 1);
        path_to_id.insert("b.rs".to_owned(), 2);
        let mut atlas = Atlas {
            root_path: directory.path().to_path_buf(),
            revision: "abc123".to_owned(),
            dirty: false,
            excluded_test_files: 0,
            excluded_hidden_files: 0,
            excluded_custom_files: 0,
            excluded_paths: Vec::new(),
            nodes: nodes.clone(),
            calls: vec![Callsite {
                id: 7,
                source: 1,
                source_line: 3,
                target: 2,
                target_line: Some(8),
                callee: "b::run".to_owned(),
                kind: "syntax".to_owned(),
                analyzer: "fixture".to_owned(),
                confidence: 0.9,
            }],
            path_to_id: path_to_id.clone(),
            report: AnalyzerReport {
                resolved_calls: 1,
                ..AnalyzerReport::default()
            },
            timings: BuildTimings::default(),
        };
        save_cached_analysis(&mut atlas).unwrap();

        atlas.calls.clear();
        atlas.report = AnalyzerReport::default();
        atlas.nodes = nodes;
        atlas.path_to_id = path_to_id;
        assert!(load_cached_analysis(&mut atlas).unwrap());
        assert_eq!(atlas.calls.len(), 1);
        assert_eq!(atlas.calls[0].source, 1);
        assert_eq!(atlas.calls[0].target, 2);
        assert!(atlas.report.cache_hit);
    }

    #[test]
    fn resolves_typescript_and_gleam_imported_calls_without_aggregation() {
        let directory = tempfile::tempdir().unwrap();
        fs::create_dir_all(directory.path().join("src")).unwrap();
        fs::create_dir_all(directory.path().join("gleam/src")).unwrap();
        fs::write(
            directory.path().join("src/a.ts"),
            "import { run as execute } from './b';\nexecute();\nexecute();\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("src/b.ts"),
            "export function run() {}\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("gleam/src/a.gleam"),
            "import b\npub fn start() { b.run() }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("gleam/src/b.gleam"),
            "pub fn run() { Nil }\n",
        )
        .unwrap();
        let paths = [
            "src/a.ts",
            "src/b.ts",
            "gleam/src/a.gleam",
            "gleam/src/b.gleam",
        ];
        let mut nodes = vec![Node {
            id: 0,
            parent: None,
            children: (1..=paths.len()).collect(),
            kind: NodeKind::Directory,
            name: "fixture".to_owned(),
            path: String::new(),
            depth: 0,
            bytes: 4,
            loc: 4,
            commits: 4,
            weight: 4.0,
            language: "directory".to_owned(),
            rect: Rect::default(),
        }];
        let mut path_to_id = HashMap::new();
        for (index, path) in paths.iter().enumerate() {
            let id = index + 1;
            nodes.push(Node {
                id,
                parent: Some(0),
                children: Vec::new(),
                kind: NodeKind::File,
                name: path.rsplit('/').next().unwrap().to_owned(),
                path: (*path).to_owned(),
                depth: 1,
                bytes: 1,
                loc: 1,
                commits: 1,
                weight: 1.0,
                language: if path.ends_with(".gleam") {
                    "gleam".to_owned()
                } else {
                    "typescript".to_owned()
                },
                rect: Rect::default(),
            });
            path_to_id.insert((*path).to_owned(), id);
        }
        let mut atlas = Atlas {
            root_path: directory.path().to_path_buf(),
            revision: "test".to_owned(),
            dirty: true,
            excluded_test_files: 0,
            excluded_hidden_files: 0,
            excluded_custom_files: 0,
            excluded_paths: Vec::new(),
            nodes,
            calls: Vec::new(),
            path_to_id,
            report: AnalyzerReport::default(),
            timings: BuildTimings::default(),
        };
        let mut next_id = 0;
        initialize_language_coverage(&mut atlas);
        analyze_typescript_calls_fallback(&mut atlas, &mut next_id).unwrap();
        analyze_gleam_calls(&mut atlas, &mut next_id).unwrap();

        assert_eq!(atlas.report.typescript_files_scanned, 2);
        assert_eq!(atlas.report.typescript_calls, 2);
        assert_eq!(atlas.report.gleam_files_scanned, 2);
        assert_eq!(atlas.report.gleam_calls, 1);
        assert_eq!(atlas.report.language_coverage["gleam"].files_discovered, 2);
        assert_eq!(atlas.report.language_coverage["gleam"].files_analyzed, 2);
        assert_eq!(
            atlas.report.language_coverage["gleam"].callsites_resolved,
            1
        );
        assert_eq!(atlas.calls.len(), 3);
        assert_eq!(atlas.calls[0].source, 1);
        assert_eq!(atlas.calls[0].target, 2);
        assert_eq!(atlas.calls[1].source, 1);
        assert_eq!(atlas.calls[1].target, 2);
        assert_eq!(atlas.calls[2].source, 3);
        assert_eq!(atlas.calls[2].target, 4);
    }

    #[test]
    fn semantic_scip_ingestion_keeps_individual_calls_and_exact_targets() {
        let directory = tempfile::tempdir().unwrap();
        fs::create_dir_all(directory.path().join("src")).unwrap();
        fs::write(
            directory.path().join("src/a.ts"),
            "execute();\nexecute();\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("src/b.ts"),
            "export function execute() {}\n",
        )
        .unwrap();
        let paths = ["src/a.ts", "src/b.ts"];
        let mut path_to_id = HashMap::new();
        let mut nodes = vec![Node {
            id: 0,
            parent: None,
            children: vec![1, 2],
            kind: NodeKind::Directory,
            name: "fixture".to_owned(),
            path: String::new(),
            depth: 0,
            bytes: 2,
            loc: 2,
            commits: 2,
            weight: 2.0,
            language: "directory".to_owned(),
            rect: Rect::default(),
        }];
        for (index, path) in paths.iter().enumerate() {
            let id = index + 1;
            nodes.push(Node {
                id,
                parent: Some(0),
                children: Vec::new(),
                kind: NodeKind::File,
                name: path.rsplit('/').next().unwrap().to_owned(),
                path: (*path).to_owned(),
                depth: 1,
                bytes: 1,
                loc: 1,
                commits: 1,
                weight: 1.0,
                language: "typescript".to_owned(),
                rect: Rect::default(),
            });
            path_to_id.insert((*path).to_owned(), id);
        }
        let symbol = "scip-typescript npm fixture 1.0 src/b.ts/execute().";
        let reference = |line| crate::scip::ScipOccurrence {
            range: Some(ScipRange {
                start_line: line,
                start_character: 0,
                end_line: line,
                end_character: 7,
            }),
            symbol: symbol.to_owned(),
            symbol_roles: 0,
        };
        let indexes = vec![RootedScipIndex {
            root: String::new(),
            index: ScipIndex {
                documents: vec![
                    ScipDocument {
                        relative_path: "src/a.ts".to_owned(),
                        language: "typescript".to_owned(),
                        position_encoding: 2,
                        occurrences: vec![reference(0), reference(1)],
                    },
                    ScipDocument {
                        relative_path: "src/b.ts".to_owned(),
                        language: "typescript".to_owned(),
                        position_encoding: 2,
                        occurrences: vec![crate::scip::ScipOccurrence {
                            range: Some(ScipRange {
                                start_line: 0,
                                start_character: 16,
                                end_line: 0,
                                end_character: 23,
                            }),
                            symbol: symbol.to_owned(),
                            symbol_roles: 1,
                        }],
                    },
                ],
            },
        }];
        let mut atlas = Atlas {
            root_path: directory.path().to_path_buf(),
            revision: "test".to_owned(),
            dirty: true,
            excluded_test_files: 0,
            excluded_hidden_files: 0,
            excluded_custom_files: 0,
            excluded_paths: Vec::new(),
            nodes,
            calls: Vec::new(),
            path_to_id,
            report: AnalyzerReport::default(),
            timings: BuildTimings::default(),
        };
        let mut next_id = 0;

        let stats = ingest_scip_calls(&mut atlas, &indexes, &["typescript"], &mut next_id).unwrap();

        assert_eq!(stats.files_by_language["typescript"], 2);
        assert_eq!(stats.resolved_by_language["typescript"], 2);
        assert_eq!(atlas.calls.len(), 2);
        assert_eq!(atlas.calls[0].source_line, 1);
        assert_eq!(atlas.calls[1].source_line, 2);
        assert_eq!(atlas.calls[0].target, 2);
        assert_eq!(atlas.calls[0].target_line, Some(1));
        assert_eq!(atlas.calls[0].analyzer, "scip-typescript");
    }

    #[test]
    fn semantic_scip_ingestion_supports_javascript_documents() {
        let directory = tempfile::tempdir().unwrap();
        fs::create_dir_all(directory.path().join("src")).unwrap();
        fs::write(directory.path().join("src/a.js"), "execute();\n").unwrap();
        fs::write(
            directory.path().join("src/b.js"),
            "export function execute() {}\n",
        )
        .unwrap();
        let paths = ["src/a.js", "src/b.js"];
        let mut path_to_id = HashMap::new();
        let mut nodes = vec![Node {
            id: 0,
            parent: None,
            children: vec![1, 2],
            kind: NodeKind::Directory,
            name: "fixture".to_owned(),
            path: String::new(),
            depth: 0,
            bytes: 2,
            loc: 2,
            commits: 2,
            weight: 2.0,
            language: "directory".to_owned(),
            rect: Rect::default(),
        }];
        for (index, path) in paths.iter().enumerate() {
            let id = index + 1;
            nodes.push(Node {
                id,
                parent: Some(0),
                children: Vec::new(),
                kind: NodeKind::File,
                name: path.rsplit('/').next().unwrap().to_owned(),
                path: (*path).to_owned(),
                depth: 1,
                bytes: 1,
                loc: 1,
                commits: 1,
                weight: 1.0,
                language: "javascript".to_owned(),
                rect: Rect::default(),
            });
            path_to_id.insert((*path).to_owned(), id);
        }
        let symbol = "scip-typescript npm fixture 1.0 src/b.js/execute().";
        let indexes = vec![RootedScipIndex {
            root: String::new(),
            index: ScipIndex {
                documents: vec![
                    ScipDocument {
                        relative_path: "src/a.js".to_owned(),
                        language: "javascript".to_owned(),
                        position_encoding: 2,
                        occurrences: vec![crate::scip::ScipOccurrence {
                            range: Some(ScipRange {
                                start_line: 0,
                                start_character: 0,
                                end_line: 0,
                                end_character: 7,
                            }),
                            symbol: symbol.to_owned(),
                            symbol_roles: 0,
                        }],
                    },
                    ScipDocument {
                        relative_path: "src/b.js".to_owned(),
                        language: "javascript".to_owned(),
                        position_encoding: 2,
                        occurrences: vec![crate::scip::ScipOccurrence {
                            range: Some(ScipRange {
                                start_line: 0,
                                start_character: 16,
                                end_line: 0,
                                end_character: 23,
                            }),
                            symbol: symbol.to_owned(),
                            symbol_roles: 1,
                        }],
                    },
                ],
            },
        }];
        let mut atlas = Atlas {
            root_path: directory.path().to_path_buf(),
            revision: "test".to_owned(),
            dirty: true,
            excluded_test_files: 0,
            excluded_hidden_files: 0,
            excluded_custom_files: 0,
            excluded_paths: Vec::new(),
            nodes,
            calls: Vec::new(),
            path_to_id,
            report: AnalyzerReport::default(),
            timings: BuildTimings::default(),
        };
        let mut next_id = 0;

        let stats = ingest_scip_calls(&mut atlas, &indexes, &["javascript"], &mut next_id).unwrap();

        assert_eq!(stats.files_by_language["javascript"], 2);
        assert_eq!(stats.resolved_by_language["javascript"], 1);
        assert_eq!(atlas.calls.len(), 1);
        assert_eq!(atlas.calls[0].source_line, 1);
        assert_eq!(atlas.calls[0].target_line, Some(1));
        assert_eq!(atlas.calls[0].analyzer, "scip-typescript");
    }

    #[test]
    fn call_suffix_classifier_handles_methods_generics_macros_and_optional_calls() {
        assert!(suffix_is_call("()"));
        assert!(suffix_is_call("::<Value>()"));
        assert!(suffix_is_call("<Value>()"));
        assert!(suffix_is_call("!()"));
        assert!(suffix_is_call("?.()"));
        assert!(!suffix_is_call(";"));
        assert!(!suffix_is_call("::<Value>"));
    }

    fn rust_node(id: usize, path: &str) -> Node {
        Node {
            id,
            parent: Some(0),
            children: Vec::new(),
            kind: NodeKind::File,
            name: path.to_owned(),
            path: path.to_owned(),
            depth: 1,
            bytes: 1,
            loc: 1,
            commits: 1,
            weight: 1.0,
            language: "rust".to_owned(),
            rect: Rect::default(),
        }
    }
}
