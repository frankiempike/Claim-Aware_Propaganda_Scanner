function createTechniquesTable(spans) {
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
    console.log("Techniques for text:", span.text, span.techniques);
    techniqueArray = span.techniques.map((item) => item.technique.replaceAll("_", " "));
    techniqueCell.textContent = techniqueArray.join(", ");
    row.appendChild(textCell);
    row.appendChild(techniqueCell);
    table.appendChild(row);
  });

  console.log("Created techniques table:", table);

  return table;
}

// Restore previous results on popup open
chrome.storage.local.get("lastResult", ({ lastResult }) => {
  if (lastResult) {
    const table = createTechniquesTable(lastResult);
    const outputDiv = document.getElementById("output");
    outputDiv.innerHTML = "";
    outputDiv.appendChild(table);
  }
});

document.getElementById("clearBtn").addEventListener("click", () => {
  chrome.storage.local.remove("lastResult");
  document.getElementById("output").innerHTML = "";
});

document.getElementById("scanBtn").addEventListener("click", () => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    chrome.scripting.executeScript(
      {
        target: { tabId: tabs[0].id },
        func: () => document.body.innerText,
      },
      async (results) => {
        document.getElementById("output").textContent = "Processing...";
        const response = await fetch(API_ENDPOINT, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ text: results[0].result }),
        });
        const data = await response.json();
        console.log("Processed data:", data);
        result = data.map((item) => {
          return {
            text: item.text,
            techniques: item.techniques,
          };
        });

        chrome.storage.local.set({ lastResult: result });
        const table = createTechniquesTable(result);
        const outputDiv = document.getElementById("output");
        outputDiv.innerHTML = "";
        outputDiv.appendChild(table);
      }
    );
  });
});