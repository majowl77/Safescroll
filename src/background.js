let lastUrl = "";

chrome.tabs.onUpdated.addListener(function (tabId, changeInfo, tab) {
  if (changeInfo.status === "complete" && tab && tab.id && tab.url) {
    if (lastUrl === tab.url) {
      console.log("Skipping duplicate URL:", tab.url);
      return;
    }

    lastUrl = tab.url;

    chrome.tabs.sendMessage(tab.id, { type: "hideBlackBox" });
    console.log("Tab updated:", tab.url, tab.id);
    fetch("http://localhost:5000/extract_url", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ url: tab.url }),
    })
      .then((response) => response.json())
      .then((data) => {
        console.log(data["output"]);

        if (data["output"] > 0.5) {
          chrome.action.setPopup({ tabId: tab.id, popup: "newpopupred.html" });
          chrome.tabs.sendMessage(tab.id, { type: "showBlackBox" });
          chrome.storage.local.set({ percent: data["output"] }).then(() => {
            console.log("Value is set");
          });
          chrome.action.setBadgeText({ text: "Alrt" });
          chrome.action.setBadgeBackgroundColor({ color: "#F28379" });
        } else {
          chrome.action.setPopup({
            tabId: tab.id,
            popup: "newpopupgreen.html",
          });
          chrome.tabs.sendMessage(tab.id, { type: "hideBlackBox" });
          chrome.storage.local.set({ percent: data["output"] }).then(() => {
            console.log("Value is set");
          });
          chrome.action.setBadgeText({ text: "Safe" });
          chrome.action.setBadgeBackgroundColor({ color: "#155E63" });
        }
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  }
});
