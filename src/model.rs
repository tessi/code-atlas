use std::{
    collections::{BTreeMap, HashMap},
    fmt,
    path::PathBuf,
    str::FromStr,
};

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

pub type NodeId = usize;

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Rect {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

impl Rect {
    pub fn width(self) -> f64 {
        (self.x1 - self.x0).max(0.0)
    }

    pub fn height(self) -> f64 {
        (self.y1 - self.y0).max(0.0)
    }

    pub fn area(self) -> f64 {
        self.width() * self.height()
    }

    pub fn center(self) -> Point {
        Point {
            x: (self.x0 + self.x1) * 0.5,
            y: (self.y0 + self.y1) * 0.5,
        }
    }

    pub fn inset(self, amount: f64) -> Self {
        let x = amount.min(self.width() * 0.22);
        let y = amount.min(self.height() * 0.22);
        Self {
            x0: self.x0 + x,
            y0: self.y0 + y,
            x1: self.x1 - x,
            y1: self.y1 - y,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeKind {
    Directory,
    File,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub parent: Option<NodeId>,
    pub children: Vec<NodeId>,
    pub kind: NodeKind,
    pub name: String,
    pub path: String,
    pub depth: usize,
    pub bytes: u64,
    pub loc: u64,
    pub commits: u64,
    pub weight: f64,
    pub language: String,
    pub rect: Rect,
}

impl Node {
    pub fn is_file(&self) -> bool {
        self.kind == NodeKind::File
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Callsite {
    pub id: u64,
    pub source: NodeId,
    pub source_line: u32,
    pub target: NodeId,
    pub target_line: Option<u32>,
    pub callee: String,
    pub kind: String,
    pub analyzer: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnalyzerReport {
    pub elixir_files_traced: usize,
    pub elixir_script_files_scanned: usize,
    pub typescript_files_scanned: usize,
    pub gleam_files_scanned: usize,
    pub rust_files_scanned: usize,
    pub resolved_calls: usize,
    pub unresolved_calls: usize,
    pub rustler_calls: usize,
    pub rust_calls: usize,
    pub typescript_calls: usize,
    pub gleam_calls: usize,
    #[serde(default)]
    pub same_file_calls_excluded: usize,
    pub cache_hit: bool,
    pub warnings: Vec<String>,
    #[serde(default)]
    pub language_coverage: BTreeMap<String, LanguageCoverage>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LanguageCoverage {
    pub provider: String,
    pub fidelity: String,
    pub files_discovered: usize,
    pub files_analyzed: usize,
    pub callsites_resolved: usize,
    pub callsites_unresolved: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BuildTimings {
    pub scan_ms: u64,
    pub analyze_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Atlas {
    pub root_path: PathBuf,
    pub revision: String,
    pub dirty: bool,
    pub excluded_test_files: usize,
    pub excluded_hidden_files: usize,
    pub excluded_custom_files: usize,
    pub excluded_paths: Vec<String>,
    pub nodes: Vec<Node>,
    pub calls: Vec<Callsite>,
    pub path_to_id: HashMap<String, NodeId>,
    pub report: AnalyzerReport,
    pub timings: BuildTimings,
}

impl Atlas {
    pub fn files(&self) -> impl Iterator<Item = &Node> {
        self.nodes.iter().filter(|node| node.is_file())
    }

    pub fn total_loc(&self) -> u64 {
        self.files().map(|node| node.loc).sum()
    }

    pub fn hierarchy_path(&self, source: NodeId, target: NodeId) -> Vec<NodeId> {
        let mut source_chain = Vec::new();
        let mut current = Some(source);
        while let Some(id) = current {
            source_chain.push(id);
            current = self.nodes[id].parent;
        }

        let mut target_chain = Vec::new();
        current = Some(target);
        while let Some(id) = current {
            target_chain.push(id);
            current = self.nodes[id].parent;
        }

        let target_positions: HashMap<NodeId, usize> = target_chain
            .iter()
            .copied()
            .enumerate()
            .map(|(index, id)| (id, index))
            .collect();

        let (source_lca_index, target_lca_index) = source_chain
            .iter()
            .enumerate()
            .find_map(|(source_index, id)| {
                target_positions
                    .get(id)
                    .copied()
                    .map(|target_index| (source_index, target_index))
            })
            .unwrap_or((source_chain.len() - 1, target_chain.len() - 1));

        let mut path = source_chain[..=source_lca_index].to_vec();
        path.extend(target_chain[..target_lca_index].iter().rev().copied());
        path
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    Loc,
    Bytes,
    Commits,
}

impl Metric {
    pub fn value(self, bytes: u64, loc: u64, commits: u64) -> f64 {
        let raw = match self {
            Self::Loc => loc,
            Self::Bytes => bytes,
            Self::Commits => commits,
        };
        (raw.max(1) as f64).sqrt()
    }
}

impl fmt::Display for Metric {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Loc => "loc",
            Self::Bytes => "bytes",
            Self::Commits => "commits",
        })
    }
}

impl FromStr for Metric {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "loc" => Ok(Self::Loc),
            "bytes" => Ok(Self::Bytes),
            "commits" => Ok(Self::Commits),
            _ => bail!("unknown metric {value:?}; expected loc, bytes, or commits"),
        }
    }
}
