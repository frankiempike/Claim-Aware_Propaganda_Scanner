// Load saved theme preference on startup
chrome.storage.local.get(["darkMode"], ({ darkMode }) => {
  if (darkMode) {
    document.body.classList.add("dark-mode");
  }
  updateThemeButton();
});

// Theme toggle button handler
document.getElementById("themeToggle").addEventListener("click", () => {
  const isDarkMode = document.body.classList.toggle("dark-mode");
  chrome.storage.local.set({ darkMode: isDarkMode });
  updateThemeButton();
});

function updateThemeButton() {
  const btn = document.getElementById("themeToggle");
  btn.textContent = document.body.classList.contains("dark-mode") ? "Light Mode" : "Dark Mode";
}
