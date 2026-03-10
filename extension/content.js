// Extracts the visible text content from the current page
function getPageText() {
  return document.body.innerText;
}

// Listen for messages from the popup or background
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "getPageText") {
    sendResponse({ text: getPageText() });
  }
});
