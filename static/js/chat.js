const input = document.getElementById("question");
const sendBtn = document.getElementById("sendBtn");
const landing = document.getElementById("landing");
const messages = document.getElementById("messages");

let firstMessage = true;

function createMessage(text, sender) {

    const div = document.createElement("div");
    div.classList.add("message");

    if (sender === "user") {

        div.innerHTML = `
            <div class="user">
                <strong>You</strong><br><br>
                ${text}
            </div>
        `;

    } else {

        div.innerHTML = `
            <div class="bot">
                <strong>CourseGPT</strong><br><br>
                ${text}
            </div>
        `;

    }

    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

function showThinking() {

    const div = document.createElement("div");
    div.classList.add("message");
    div.id = "thinking";

    div.innerHTML = `
        <div class="bot">
            <strong>CourseGPT</strong><br><br>
            Thinking...
        </div>
    `;

    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

function removeThinking() {

    const thinking = document.getElementById("thinking");

    if (thinking) {
        thinking.remove();
    }

}

async function sendMessage() {

    const question = input.value.trim();

    if (question === "") return;

    // Hide landing page after first message
    if (firstMessage) {

        landing.style.display = "none";
        messages.style.display = "block";

        firstMessage = false;
    }

    createMessage(question, "user");

    input.value = "";

    showThinking();

    try {

        const response = await fetch("/ask", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                question: question
            })

        });

        const data = await response.json();

        removeThinking();

        createMessage(data.answer, "bot");

    } catch (error) {

        removeThinking();

        createMessage(
            "❌ Unable to connect to the backend. Make sure FastAPI and Ollama are running.",
            "bot"
        );

        console.error(error);
    }

}

sendBtn.addEventListener("click", sendMessage);

input.addEventListener("keypress", function (e) {

    if (e.key === "Enter") {

        sendMessage();

    }

});