use std::path::{Component, Path, PathBuf};
use std::pin::Pin;

use anyhow::{Context, Result, bail};
use futures::Stream;
pub use protocol::ToolApprovalRequest;

use crate::cancel::CancelToken;
use crate::ports::FileSystem;

pub const TOOL_REJECTED_MESSAGE: &str = "Tool call rejected by user";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalRequirement {
    NotRequired,
    Required { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolApprovalDecision {
    pub request_id: String,
    pub approved: bool,
}

/// Delivers user approval decisions for tool calls.
///
/// The returned stream yields one `ToolApprovalDecision` per request, in
/// whatever order decisions arrive (not request order). The runner consumes
/// decisions incrementally so each approved tool starts executing the moment
/// its decision lands, rather than waiting for the full batch to resolve.
pub trait ToolApprover: Send + Sync {
    fn approve(
        &self,
        requests: Vec<ToolApprovalRequest>,
        cancel: CancelToken,
    ) -> Pin<Box<dyn Stream<Item = Result<ToolApprovalDecision>> + Send + '_>>;
}

pub struct NoApprovalRequired;

impl ToolApprover for NoApprovalRequired {
    fn approve(
        &self,
        requests: Vec<ToolApprovalRequest>,
        _cancel: CancelToken,
    ) -> Pin<Box<dyn Stream<Item = Result<ToolApprovalDecision>> + Send + '_>> {
        Box::pin(futures::stream::iter(requests.into_iter().map(|request| {
            Ok(ToolApprovalDecision {
                request_id: request.request_id,
                approved: true,
            })
        })))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingPathPolicy {
    MustExist,
    AllowMissingTarget,
}

pub async fn path_approval_requirement<F: FileSystem>(
    fs: &F,
    workspace_root: &Path,
    path: &str,
    missing_policy: MissingPathPolicy,
) -> Result<ApprovalRequirement> {
    if path.is_empty() {
        bail!("path must not be empty");
    }

    let candidate = crate::tools::resolve_path(workspace_root, path);
    let lexical_workspace = normalize_lexical(workspace_root);
    let lexical_candidate = normalize_lexical(&candidate);
    let lexical_inside = lexical_candidate.starts_with(&lexical_workspace);

    let canonical_workspace = fs
        .canonicalize(workspace_root)
        .await
        .with_context(|| format!("canonicalizing workspace root {}", workspace_root.display()))?;
    let canonical_candidate = match fs.canonicalize(&candidate).await {
        Ok(path) => path,
        Err(err) => match missing_policy {
            MissingPathPolicy::MustExist => {
                return Err(err).with_context(|| {
                    format!("canonicalizing requested path {}", candidate.display())
                });
            }
            MissingPathPolicy::AllowMissingTarget => {
                canonicalize_missing_target(fs, &candidate).await?
            }
        },
    };
    let canonical_inside = canonical_candidate.starts_with(&canonical_workspace);

    if lexical_inside && canonical_inside {
        Ok(ApprovalRequirement::NotRequired)
    } else {
        Ok(ApprovalRequirement::Required {
            reason: format!(
                "Tool targets a path outside the session worktree: {}",
                candidate.display()
            ),
        })
    }
}

async fn canonicalize_missing_target<F: FileSystem>(fs: &F, target: &Path) -> Result<PathBuf> {
    let mut missing_suffix = PathBuf::new();
    let mut cursor = target.to_path_buf();

    loop {
        match fs.canonicalize(&cursor).await {
            Ok(existing) => return Ok(existing.join(missing_suffix)),
            Err(last_err) => {
                let Some(name) = cursor.file_name() else {
                    return Err(last_err)
                        .with_context(|| format!("canonicalizing {}", target.display()));
                };
                let mut next_suffix = PathBuf::from(name);
                next_suffix.push(missing_suffix);
                missing_suffix = next_suffix;
                if !cursor.pop() {
                    return Err(last_err)
                        .with_context(|| format!("canonicalizing {}", target.display()));
                }
            }
        }
    }
}

pub(crate) fn normalize_lexical(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            Component::Normal(part) => normalized.push(part),
            Component::RootDir | Component::Prefix(_) => normalized.push(component.as_os_str()),
        }
    }
    normalized
}
