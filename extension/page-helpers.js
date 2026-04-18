// Extract main article/content text from the page
window.extractArticleText = function() {
  // Try common content selectors in order of preference
  const contentSelectors = [
    "article",
    "main",
    "[role='main']",
    ".article",
    ".post",
    ".content",
    "#content",
    "#article",
    ".post-content",
    ".entry-content",
  ];

  let contentElement = null;
  for (const selector of contentSelectors) {
    const elem = document.querySelector(selector);
    if (elem) {
      contentElement = elem;
      break;
    }
  }

  // If no content element found, use body but filter out nav/footer/sidebar
  if (!contentElement) {
    contentElement = document.body;
  }

  // Get text and clean it
  let text = contentElement.innerText || contentElement.textContent || "";

  // Basic cleanup: remove extra whitespace, but preserve paragraphs
  text = text
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .join("\n");

  // Return at least some text, even if minimal
  return text.length > 0 ? text : "Unable to extract article text";
};

window.highlightPropagandaSpansInPage = function(fullPageText, propagandaSpans) {
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
  
  propagandaSpans.forEach((span) => {
    const searchText = span.text;
    if (!searchText) return;

    // Walk the DOM to find and highlight this text
    const stack = [document.body];

    while (stack.length > 0) {
      const node = stack.pop();
      if (!node) continue;

      if (node.nodeType === Node.TEXT_NODE) {
        const parentNodeName = node.parentNode?.nodeName;
        if (excludedParents.includes(parentNodeName)) continue;

        let nodeText = node.textContent;
        
        // Find all valid matches (complete words only)
        const matches = [];
        let index = nodeText.indexOf(searchText);
        while (index !== -1) {
          // Check word boundaries
          const beforeChar = index > 0 ? nodeText[index - 1] : " ";
          const afterIndex = index + searchText.length;
          const afterChar = afterIndex < nodeText.length ? nodeText[afterIndex] : " ";
          
          const isWordCharBefore = /\w/.test(beforeChar);
          const isWordCharAfter = /\w/.test(afterChar);
          
          // Only match if it's a complete word
          if (!isWordCharBefore && !isWordCharAfter) {
            matches.push({ start: index, end: afterIndex });
          }
          
          index = nodeText.indexOf(searchText, index + 1);
        }

        // Highlight all valid matches
        if (matches.length > 0) {
          const fragments = [];
          let lastIndex = 0;

          matches.forEach((match) => {
            // Add text before match
            if (match.start > lastIndex) {
              fragments.push(document.createTextNode(nodeText.slice(lastIndex, match.start)));
            }

            // Create highlight mark
            const mark = document.createElement("mark");
            mark.dataset.propagandaHighlight = "true";
            mark.dataset.techniques = span.techniques.map((t) => t.technique).join(", ");
            mark.title = `Techniques: ${span.techniques
              .map((t) => `${t.technique.replaceAll("_", " ")} (${(t.probability * 100).toFixed(1)}%)`)
              .join(", ")}`;
            mark.textContent = searchText;
            fragments.push(mark);

            lastIndex = match.end;
          });

          // Add remaining text
          if (lastIndex < nodeText.length) {
            fragments.push(document.createTextNode(nodeText.slice(lastIndex)));
          }

          // Replace node with fragments
          const fragment = document.createDocumentFragment();
          fragments.forEach((frag) => fragment.appendChild(frag));
          node.replaceWith(fragment);
          
          // Continue searching from the parent since we just modified the DOM
          continue;
        }
      } else {
        // Add children to stack (in reverse for correct order)
        for (let i = node.childNodes.length - 1; i >= 0; i -= 1) {
          stack.push(node.childNodes[i]);
        }
      }
    }
  });
};
