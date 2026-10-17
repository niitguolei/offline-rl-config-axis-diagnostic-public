# Round-2b: verify the shortlisted new candidates. Wider/looser queries than
# round 2a, because "no results" there may just mean the query was too narrow.
$ProgressPreference = 'SilentlyContinue'

$queries = [ordered]@{
  'N1a n-step / lambda returns in OFFLINE RL (loose)' = 'all:"offline reinforcement learning" AND (all:"n-step" OR all:"lambda return" OR all:"eligibility trace")'
  'N1b adaptive / learned backup horizon, any RL'        = '(all:"state-dependent" OR all:"adaptive" OR all:"learned") AND (all:"lambda return" OR all:"n-step target" OR all:"backup length") AND cat:cs.LG'
  'N1c partial sweeping / prioritised sweeping (classic)' = 'all:"partial sweeping" OR all:"prioritised sweeping" OR all:"prioritized sweeping"'
  'N3  state-only vs action-only coverage gap taxonomy'   = 'all:"offline reinforcement learning" AND (all:"coverage gap" OR all:"state coverage" OR all:"action coverage" OR all:"support gap")'
  'N2  state density ignored / balanced replay offline'   = 'all:"offline reinforcement learning" AND (all:"state marginal" OR all:"balanced replay" OR all:"density ratio" OR all:"resampling")'
}

foreach ($label in $queries.Keys) {
  Write-Output ("=" * 78)
  Write-Output $label
  Write-Output ("-" * 78)
  $q = [uri]::EscapeDataString($queries[$label])
  $url = "http://export.arxiv.org/api/query?search_query=$q&sortBy=submittedDate&sortOrder=descending&max_results=10"
  $ok = $false
  foreach ($attempt in 1..2) {
    try {
      [xml]$xml = (Invoke-WebRequest -Uri $url -Headers @{ 'User-Agent' = 'codex-research-check' } -TimeoutSec 60).Content
      $ns = New-Object System.Xml.XmlNamespaceManager($xml.NameTable)
      $ns.AddNamespace('a', 'http://www.w3.org/2005/Atom')
      $entries = $xml.SelectNodes('//a:entry', $ns)
      if (-not $entries -or $entries.Count -eq 0) { Write-Output '  (no results)' }
      foreach ($e in $entries) {
        Write-Output ("  [{0}] {1}" -f $e.published.Substring(0,10), (($e.title -replace '\s+',' ').Trim()))
      }
      $ok = $true
      break
    } catch {
      Write-Output "  attempt $attempt failed: $($_.Exception.Message)"
      Start-Sleep -Seconds 25
    }
  }
  if (-not $ok) { Write-Output '  (gave up after retries)' }
  Start-Sleep -Seconds 18
}
