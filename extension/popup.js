// Restore previous results on popup open
chrome.storage.local.get(["lastResult", "fullResult", "pageText", "tabUrl"], ({ lastResult, fullResult, pageText, tabUrl }) => {
  if (lastResult) {
    const table = createTechniquesTable(lastResult);
    const outputDiv = document.getElementById("output");
    outputDiv.innerHTML = "";
    outputDiv.appendChild(table);

    // Re-highlight propaganda on page if we have the full results, text, and matching URL
    if (fullResult && pageText && tabUrl) {
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (tabs[0].url === tabUrl) {
          console.log("Re-highlighting propaganda spans on page with stored results (URL matches)");
          chrome.scripting.executeScript({
            target: { tabId: tabs[0].id },
            func: (pageText, spans) => {
              window.highlightPropagandaSpansInPage(pageText, spans);
            },
            args: [pageText, fullResult],
          });
        } else {
          console.log("URL mismatch - not re-highlighting. Stored:", tabUrl, "Current:", tabs[0].url);
        }
      });
    }
  }
});

document.getElementById("clearBtn").addEventListener("click", () => {
  chrome.storage.local.remove(["lastResult", "fullResult", "pageText", "tabUrl"]);
  document.getElementById("output").innerHTML = "";

  // Remove highlights from page
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    chrome.scripting.executeScript({
      target: { tabId: tabs[0].id },
      func: () => {
        document.querySelectorAll("mark[data-propaganda-highlight='true']").forEach((markNode) => {
          const textNode = document.createTextNode(markNode.textContent || "");
          markNode.replaceWith(textNode);
        });
      },
    });
  });
});

document.getElementById("scanBtn").addEventListener("click", () => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const currentUrl = tabs[0].url;
    chrome.scripting.executeScript(
      {
        target: { tabId: tabs[0].id },
        func: () => document.body.innerText,
      },
      async (results) => {
        document.getElementById("output").textContent = "Processing...";
        const fullText = results[0].result;
        const response = await fetch(API_ENDPOINT, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ text: fullText }),
        });
        const data = await response.json();
        
        // Keep full results with positions for highlighting
        const fullResult = data;
        
        // Create simplified results for table display
        const result = data.map((item) => {
          return {
            text: item.text,
            techniques: item.techniques,
          };
        });

        chrome.storage.local.set({ lastResult: result, fullResult: fullResult, pageText: fullText, tabUrl: currentUrl });
        console.log("Stored results in local storage for persistence with URL:", currentUrl);

        console.log("Executing script to highlight propaganda spans on page with new results");
        chrome.scripting.executeScript(
          {
            target: { tabId: tabs[0].id },
            func: (pageText, spans) => {
              window.highlightPropagandaSpansInPage(pageText, spans);
            },
            args: [fullText, fullResult],
          }
        );

        const table = createTechniquesTable(result);
        const outputDiv = document.getElementById("output");
        outputDiv.innerHTML = "";
        outputDiv.appendChild(table);
      }
    );
  });
});