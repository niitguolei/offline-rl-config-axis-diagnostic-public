# F1 novelty retrieval for fresh offline-RL candidates (round 2), via PowerShell.
$ProgressPreference = 'SilentlyContinue'

# NOTE: arXiv API returned 429 after several rapid rounds. Back off hard:
# fewer queries, one combined query per candidate, long sleep between calls,
# and retry-once-on-429 instead of hammering.

$queries = [ordered]@{
  'C1 state-density reweighting / balanced sampling' = 'all:"offline reinforcement learning" AND (all:"state distribution" OR all:"balanced sampling" OR all:"reweighting" OR all:"importance weighting")'
  'C2 state-dependent TD(lambda) / adaptive backup'  = 'all:"reinforcement learning" AND (all:"partial sweeping" OR all:"state-dependent lambda" OR all:"learned lambda" OR all:"adaptive bootstrapping")'
  'C3 low-data regime scaling'                       = 'all:"offline reinforcement learning" AND (all:"data efficiency" OR all:"limited data" OR all:"few trajectories" OR all:"scaling laws")'
  'C6 reward-free / unlabeled auxiliary objective'    = 'all:"offline reinforcement learning" AND (all:"reward-free" OR all:"unsupervised" OR all:"unlabeled" OR all:"self-supervised")'
  'C6b inverse-dynamics auxiliary task'              = 'all:"offline reinforcement learning" AND (all:"inverse dynamics" OR all:"auxiliary task" OR all:"representation learning")'
  'C7 evaluation protocol / reproducibility'         = 'all:"offline reinforcement learning" AND (all:"reproducib" OR all:"evaluation protocol" OR all:"variance" OR all:"hyperparameter budget")'
}

foreach ($label in $queries.Keys) {
  Write-Output ("=" * 78)
  Write-Output $label
  Write-Output ("-" * 78)
  $q = [uri]::EscapeDataString($queries[$label])
  $url = "http://export.arxiv.org/api/query?search_query=$q&sortBy=submittedDate&sortOrder=descending&max_results=12"
  try {
    [xml]$xml = (Invoke-WebRequest -Uri $url -Headers @{ 'User-Agent' = 'codex-research-check' } -TimeoutSec 45).Content
    $ns = New-Object System.Xml.XmlNamespaceManager($xml.NameTable)
    $ns.AddNamespace('a', 'http://www.w3.org/2005/Atom')
    $entries = $xml.SelectNodes('//a:entry', $ns)
    if (-not $entries -or $entries.Count -eq 0) { Write-Output '  (no results)' }
    foreach ($e in $entries) {
      $d = $e.published.Substring(0, 10)
      $t = ($e.title -replace '\s+', ' ').Trim()
      Write-Output "  [$d] $t"
    }
  } catch {
    Write-Output "  ERROR: $($_.Exception.Message)"
  }
  Start-Sleep -Seconds 15
}
