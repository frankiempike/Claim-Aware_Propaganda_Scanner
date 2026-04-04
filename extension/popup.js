document.getElementById("scanBtn").addEventListener("click", () => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    chrome.scripting.executeScript(
      {
        target: { tabId: tabs[0].id },
        func: () => document.body.innerText,
      },
      async (results) => {
        console.log("Page text:", results[0].result);
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
            technique: item.techniques,
          };
        });

        document.getElementById("output").textContent = JSON.stringify(result, null, 2);
      }
    );
  });
});