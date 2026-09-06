<!--
  Road bar: a rounded rect spanning x=0..24, y=0..8 (a wide/short design grid,
  not the 24x24 square the other icons use), so the caller can scale its
  width to an edge's pixel length and rotate it to the edge's angle. Uses
  currentColor -- see SettlementIcon.svelte. Centered on (12, 4) so a
  translate(-12, -4) recenters it on its own midpoint before rotating.
-->
<rect x="0" y="0" width="24" height="8" rx="2.5" fill="currentColor" stroke="#000000" stroke-width="0.8" />
