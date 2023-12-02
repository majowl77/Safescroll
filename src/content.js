// Create a div element for the black box
let blackBox = document.createElement("div");

// Add styles to the black box
blackBox.style.position = "fixed";
blackBox.style.top = "0";
blackBox.style.bottom = "0";
blackBox.style.left = "0";
blackBox.style.right = "0";
blackBox.style.margin = "auto";
blackBox.style.width = "32%";
blackBox.style.height = "90%";
blackBox.style.backgroundColor = "#0f0f0f";
blackBox.style.backgroundImage = "url('background.png')";
blackBox.style.opacity = "1";
blackBox.style.zIndex = "10000";
blackBox.id = "blackBoxId"; // Assign an id for future reference

// Initially set it to hidden
blackBox.style.display = "none";

// Append the black box to the body of the webpage
document.body.appendChild(blackBox);

// Listen for messages from background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "showBlackBox") {
    document.getElementById("blackBoxId").style.display = "block";
  } else if (message.type === "hideBlackBox") {
    document.getElementById("blackBoxId").style.display = "none";
  }
});
