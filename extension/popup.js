// Restore previous results on popup open
chrome.storage.local.get(["lastResult", "fullResult", "pageText", "tabUrl"], ({ lastResult, fullResult, pageText, tabUrl }) => {
  try {
    if (lastResult) {
      const table = createTechniquesTable(lastResult);
      const outputDiv = document.getElementById("output");
      outputDiv.innerHTML = "";
      outputDiv.appendChild(table);

      // Re-highlight propaganda on page if we have the full results, text, and matching URL
      if (fullResult && pageText && tabUrl) {
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
          if (!tabs || tabs.length === 0) {
            console.warn("No active tab found for re-highlighting");
            return;
          }

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
  } catch (error) {
    console.error("Error restoring previous results:", error);
  }
});

document.getElementById("clearBtn").addEventListener("click", () => {
  try {
    chrome.storage.local.remove(["lastResult", "fullResult", "pageText", "tabUrl"]);
    document.getElementById("output").innerHTML = "";

    // Remove highlights from page
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (!tabs || tabs.length === 0) {
        console.warn("No active tab found");
        return;
      }

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
  } catch (error) {
    console.error("Error clearing results:", error);
    const outputDiv = document.getElementById("output");
    outputDiv.innerHTML = "";
    const errorDiv = document.createElement("div");
    errorDiv.style.color = "#d32f2f";
    errorDiv.style.padding = "10px";
    errorDiv.innerHTML = `<strong>Error:</strong> Failed to clear results`;
    outputDiv.appendChild(errorDiv);
  }
});

document.getElementById("scanBtn").addEventListener("click", () => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const currentUrl = tabs[0].url;
    chrome.scripting.executeScript(
      {
        target: { tabId: tabs[0].id },
        func: () => {
          // Try to use extractArticleText if available, fallback to body
          if (typeof window.extractArticleText === 'function') {
            return window.extractArticleText();
          }
          return document.body.innerText;
        },
      },
      async (results) => {
        try {
          if (!results || !results[0] || !results[0].result) {
            throw new Error("Could not extract text from page");
          }

          document.getElementById("output").textContent = "Processing...";
          const fullText = results[0].result;
          console.log("Extracted full page text for processing. Length:", fullText.length);
          console.log("Full text:", fullText.substring(0, 500), "...");
          
          const response = await fetch(API_ENDPOINT, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ text: fullText }),
          });

          if (!response.ok) {
            throw new Error(`API error: ${response.status} ${response.statusText}`);
          }

          const data = await response.json();
          
          if (!Array.isArray(data)) {
            throw new Error("Invalid API response format");
          }

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
            },
            (scriptResults) => {
              if (chrome.runtime.lastError) {
                console.error("Script execution error:", chrome.runtime.lastError);
              }
            }
          );

          const table = createTechniquesTable(result);
          const outputDiv = document.getElementById("output");
          outputDiv.innerHTML = "";
          outputDiv.appendChild(table);
        } catch (error) {
          console.error("Error during scan:", error);
          const outputDiv = document.getElementById("output");
          outputDiv.innerHTML = "";
          const errorDiv = document.createElement("div");
          errorDiv.style.color = "#d32f2f";
          errorDiv.style.padding = "10px";
          errorDiv.style.backgroundColor = "#ffebee";
          errorDiv.style.borderRadius = "4px";
          errorDiv.style.marginTop = "10px";
          errorDiv.innerHTML = `<strong>Error:</strong> ${error.message}`;
          outputDiv.appendChild(errorDiv);
        }
      }
    );
  });
});