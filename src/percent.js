
chrome.storage.local.get(["percent"]).then((result) => {
    document.getElementById("percent").innerText = result.percent.toFixed(2) * 100;
  });