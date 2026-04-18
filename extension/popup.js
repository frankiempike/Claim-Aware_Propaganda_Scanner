// Restore previous results on popup open
chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
  if (!tabs || tabs.length === 0) {
    console.warn("No active tab found");
    return;
  }

  const currentUrl = tabs[0].url;
  chrome.storage.local.get(["scanResults"], ({ scanResults = {} }) => {
    try {
      const urlResults = scanResults[currentUrl];
      if (urlResults && urlResults.lastResult) {
        console.log(`Found stored results for URL: ${currentUrl}`);
        const outputDiv = document.getElementById("output");
        outputDiv.innerHTML = "";

        if (urlResults.lastResult.length === 0) {
          // No propaganda detected
          const messageDiv = document.createElement("div");
          messageDiv.classList.add("message-div");
          messageDiv.textContent = "No propaganda detected on this page.";
          outputDiv.appendChild(messageDiv);
        } else {
          const table = createTechniquesTable(urlResults.lastResult);
          outputDiv.appendChild(table);
        }

        // Re-highlight propaganda on page with stored results
        if (urlResults.fullResult && urlResults.pageText) {
          console.log("Re-highlighting propaganda spans on page with stored results");
          chrome.scripting.executeScript({
            target: { tabId: tabs[0].id },
            func: window.highlightPropagandaSpans,
            args: [urlResults.pageText, urlResults.fullResult],
          });
        }
      } else {
        // No stored results for this URL
        const outputDiv = document.getElementById("output");
        outputDiv.innerHTML = "";
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message-div");
        messageDiv.textContent = "Click 'Scan Page' to analyze this page for propaganda techniques.";
        outputDiv.appendChild(messageDiv);
      }
    } catch (error) {
      console.error("Error restoring previous results:", error);
    }
  });
});

document.getElementById("clearBtn").addEventListener("click", () => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (!tabs || tabs.length === 0) {
      console.warn("No active tab found");
      return;
    }

    const currentUrl = tabs[0].url;
    try {
      chrome.storage.local.get(["scanResults"], ({ scanResults = {} }) => {
        // Remove results for current URL
        delete scanResults[currentUrl];
        chrome.storage.local.set({ scanResults });
        console.log(`Cleared scan results for URL: ${currentUrl}`);
      });

      document.getElementById("output").innerHTML = "";

      // Remove highlights from page
      chrome.scripting.executeScript({
        target: { tabId: tabs[0].id },
        func: () => {
          document.querySelectorAll("mark[data-propaganda-highlight='true']").forEach((markNode) => {
            const textNode = document.createTextNode(markNode.textContent || "");
            markNode.replaceWith(textNode);
          });
        },
      });
    } catch (error) {
      console.error("Error clearing results:", error);
      const outputDiv = document.getElementById("output");
      outputDiv.innerHTML = "";
      const errorDiv = document.createElement("div");
      errorDiv.classList.add("error-div");
      errorDiv.innerHTML = `<strong>Error:</strong> Failed to clear results`;
      outputDiv.appendChild(errorDiv);
    }
  });
});

document.getElementById("scanBtn").addEventListener("click", () => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const currentUrl = tabs[0].url;
    chrome.scripting.executeScript(
      {
        target: { tabId: tabs[0].id },
        func: () => {
          // Embedded extraction logic
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

          if (!contentElement) {
            contentElement = document.body;
          }

          let text = contentElement.innerText || contentElement.textContent || "";
          text = text
            .split("\n")
            .map((line) => line.trim())
            .filter((line) => line.length > 0)
            .join("\n");

          return text.length > 0 ? text : "Unable to extract article text";
        },
      },
      async (results) => {
        try {
          if (!results || !results[0] || !results[0].result) {
            throw new Error("Could not extract text from page");
          }

          document.getElementById("output").textContent = "Processing...";
          let fullText = results[0].result;
          console.log("Extracted full page text for processing. Length:", fullText.length);
          
          // Split text into chunks (from config.js)
          const chunkSize = typeof MAX_TEXT_LENGTH !== "undefined" ? MAX_TEXT_LENGTH : 5000;
          const chunks = [];
          for (let i = 0; i < fullText.length; i += chunkSize) {
            chunks.push(fullText.substring(i, i + chunkSize));
          }
          
          console.log(`Processing ${chunks.length} chunk(s) of ~${chunkSize} characters each`);
          
          // Process each chunk and combine results
          let allResults = [];
          let charOffset = 0;
          
          for (let i = 0; i < chunks.length; i++) {
            // Add 2 second pause between requests (not before first request)
            if (i > 0) {
              console.log("Waiting 2 seconds before next chunk...");
              await new Promise(resolve => setTimeout(resolve, 2000));
            }
            
            console.log(`Processing chunk ${i + 1}/${chunks.length}`);
            document.getElementById("output").textContent = `Processing... (${i + 1}/${chunks.length})`;
            
            const response = await fetch(API_ENDPOINT, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({ text: chunks[i] }),
            });

            if (!response.ok) {
              throw new Error(`API error on chunk ${i + 1}: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();
            
            if (!Array.isArray(data)) {
              throw new Error(`Invalid API response format for chunk ${i + 1}`);
            }
            
            // Adjust positions for chunks beyond the first
            if (i > 0) {
              data.forEach(item => {
                item.start += charOffset;
                item.end += charOffset;
              });
            }
            
            allResults = allResults.concat(data);
            charOffset += chunks[i].length;
          }

          const data = allResults;
          
          // Keep full results with positions for highlighting
          const fullResult = data;
          
          // Create simplified results for table display
          const result = data.map((item) => {
            return {
              text: item.text,
              techniques: item.techniques,
            };
          });

          // Store results under URL key
          chrome.storage.local.get(["scanResults"], ({ scanResults = {} }) => {
            scanResults[currentUrl] = {
              lastResult: result,
              fullResult: fullResult,
              pageText: fullText,
            };
            chrome.storage.local.set({ scanResults });
            console.log(`Stored scan results for URL: ${currentUrl}`);
          });

          console.log("Executing script to highlight propaganda spans on page with new results");
          chrome.scripting.executeScript(
            {
              target: { tabId: tabs[0].id },
              func: window.highlightPropagandaSpans,
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

          if (result.length === 0) {
            // No propaganda detected
            const messageDiv = document.createElement("div");
            messageDiv.classList.add("message-div");
            messageDiv.textContent = "No propaganda detected on this page (for now).";
            outputDiv.appendChild(messageDiv);
          } else {
            outputDiv.appendChild(table);
          }
        } catch (error) {
          console.error("Error during scan:", error);
          const outputDiv = document.getElementById("output");
          outputDiv.innerHTML = "";
          const errorDiv = document.createElement("div");
          errorDiv.classList.add("error-div");
          errorDiv.innerHTML = `<strong>Error:</strong> ${error.message}`;
          outputDiv.appendChild(errorDiv);
        }
      }
    );
  });
});