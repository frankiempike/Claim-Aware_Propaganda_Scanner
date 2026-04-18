window.createTechniquesTable = function(spans) {
  const table = document.createElement("table");
  const headerRow = document.createElement("tr");
  const headers = ["Text", "Detected Techniques"];
  headers.forEach((header) => {
    const th = document.createElement("th");
    th.textContent = header;
    headerRow.appendChild(th);
  });
  table.appendChild(headerRow);

  spans.forEach((span) => {
    const row = document.createElement("tr");
    const textCell = document.createElement("td");
    textCell.textContent = span.text;
    const techniqueCell = document.createElement("td");
    const techniqueArray = span.techniques.map((item) => 
      `${item.technique.replaceAll("_", " ")} (${(item.probability * 100).toFixed(1)}%)`
  );
    techniqueCell.textContent = techniqueArray.join(", ");
    row.appendChild(textCell);
    row.appendChild(techniqueCell);
    table.appendChild(row);
  });

  return table;
};

// Helper function to highlight propaganda spans on the page
window.highlightPropagandaSpans = function (fullPageText, propagandaSpans) {
  if (!Array.isArray(propagandaSpans) || propagandaSpans.length === 0) {
    return;
  }

  // Inject CSS styles for hover effect
  if (!document.getElementById("propaganda-highlight-styles")) {
    const style = document.createElement("style");
    style.id = "propaganda-highlight-styles";
    style.textContent = `
      mark[data-propaganda-highlight="true"] {
        background-color: #ffef86;
        padding: 0 2px;
        cursor: pointer;
        border-radius: 2px;
        transition: all 0.2s ease;
      }
      mark[data-propaganda-highlight="true"]:hover {
        background-color: #ffd700;
        box-shadow: 0 0 4px rgba(255, 215, 0, 0.8);
        padding: 0 2px;
      }
    `;
    document.head.appendChild(style);
  }

  // Clear existing highlights
  document.querySelectorAll("mark[data-propaganda-highlight='true']").forEach((markNode) => {
    const textNode = document.createTextNode(markNode.textContent || "");
    markNode.replaceWith(textNode);
  });

  const excludedParents = ["SCRIPT", "STYLE", "NOSCRIPT", "TEXTAREA", "MARK"];
  let searchStartPos = 0; // Track position for sequential search
  
  // Sort spans by their start position for consistent ordering
  const sortedSpans = [...propagandaSpans].sort((a, b) => (a.start || 0) - (b.start || 0));
  
  sortedSpans.forEach((span) => {
    let searchText = span.text;
    let normalizedSearchText = searchText.replace(/\s+/g, ' ').trim();
    if (!searchText) return;

    // Find this span's text in the remaining text (try exact match first, then normalized whitespace)
    let searchResult = fullPageText.indexOf(searchText, searchStartPos);
    
    if (searchResult === -1) {
      // Try with normalized whitespace (collapse multiple spaces/newlines to single space)
      const normalizedFullText = fullPageText.replace(/\s+/g, ' ');
      const normalizedStartPos = Math.max(0, searchStartPos - 50); // Account for position shift from normalization
      
      searchResult = normalizedFullText.indexOf(normalizedSearchText, normalizedStartPos);
      if (searchResult !== -1) {
        console.log(`[Propaganda Highlighter] Found span with normalized whitespace: "${searchText}"`);
        // We'll use normalizedSearchText for DOM matching since fullPageText had different whitespace
      } else {
        console.log(`[Propaganda Highlighter] Could not find span: "${searchText}" starting from position ${searchStartPos}`);
        // Skip forward by a reasonable amount (~100 chars or 10% of remaining text) and try again
        const skipDistance = Math.max(100, Math.floor((fullPageText.length - searchStartPos) * 0.1));
        searchStartPos += skipDistance;
        searchResult = fullPageText.indexOf(searchText, searchStartPos);
        if (searchResult === -1) {
          console.log(`[Propaganda Highlighter] Still not found after skip. Giving up on: "${searchText}"`);
          return;
        }
        console.log(`[Propaganda Highlighter] Found span after skip: "${searchText}" at position ${searchResult}`);
      }
    } else {
      console.log(`[Propaganda Highlighter] Found span: "${searchText}" at position ${searchResult}`);
    }
    searchStartPos = searchResult + searchText.length; // Move past this match for next search

    // Now find and highlight this specific text in the DOM
    const stack = [document.body];

    while (stack.length > 0) {
      const node = stack.pop();
      if (!node) continue;

      if (node.nodeType === Node.TEXT_NODE) {
        const parentNodeName = node.parentNode?.nodeName;
        if (excludedParents.includes(parentNodeName)) continue;

        let nodeText = node.textContent;
        
        // Try exact match first, then normalized whitespace match
        let index = nodeText.indexOf(searchText);
        let matchLength = searchText.length;
        let matchText = searchText;
        
        if (index === -1) {
          // Try with normalized whitespace
          const normalizedNodeText = nodeText.replace(/\s+/g, ' ');
          const normalizedSearchText = searchText.replace(/\s+/g, ' ');
          index = normalizedNodeText.indexOf(normalizedSearchText);
          if (index !== -1) {
            matchLength = normalizedSearchText.length;
            matchText = normalizedSearchText;
            nodeText = normalizedNodeText; // Use normalized for subsequent operations
          }
        }
        
        // Find this specific occurrence only (first match with word boundaries)
        if (index !== -1) {
          // Check word boundaries for this occurrence
          const beforeChar = index > 0 ? nodeText[index - 1] : " ";
          const afterIndex = index + matchLength;
          const afterChar = afterIndex < nodeText.length ? nodeText[afterIndex] : " ";
          
          // Allow punctuation or whitespace around the span, just not word characters
          const isWordCharBefore = /\w/.test(beforeChar);
          const isWordCharAfter = /\w/.test(afterChar);
          
          // Only highlight if it's a complete word (not in middle of another word)
          if (!isWordCharBefore && !isWordCharAfter) {
            const fragments = [];

            // Add text before match
            if (index > 0) {
              fragments.push(document.createTextNode(nodeText.slice(0, index)));
            }

            // Create highlight mark
            const mark = document.createElement("mark");
            mark.dataset.propagandaHighlight = "true";
            mark.dataset.techniques = span.techniques.map((t) => t.technique).join(", ");
            mark.title = `Techniques: ${span.techniques
              .map((t) => `${t.technique.replaceAll("_", " ")} (${(t.probability * 100).toFixed(1)}%)`)
              .join(", ")}`;
            mark.textContent = matchText;
            fragments.push(mark);

            // Add remaining text
            if (afterIndex < nodeText.length) {
              fragments.push(document.createTextNode(nodeText.slice(afterIndex)));
            }

            // Replace node with fragments
            const fragment = document.createDocumentFragment();
            fragments.forEach((frag) => fragment.appendChild(frag));
            node.replaceWith(fragment);
            
            // Done with this span, move to next
            return;
          }
        }
      } else {
        // Add children to stack (in reverse for correct order)
        for (let i = node.childNodes.length - 1; i >= 0; i -= 1) {
          stack.push(node.childNodes[i]);
        }
      }
    }
  });
}