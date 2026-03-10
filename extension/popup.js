document.getElementById("scan-btn").addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  chrome.tabs.sendMessage(tab.id, { action: "getPageText" }, (response) => {
    const resultEl = document.getElementById("result");
    if (chrome.runtime.lastError || !response) {
      resultEl.textContent = "Error: could not read page.";
    } else {
      resultEl.textContent = response.text;
    }
    resultEl.style.display = "block";
  });
});
