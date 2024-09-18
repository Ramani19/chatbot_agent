const modeToggle = document.getElementById('modeToggle');
const questionInput = document.getElementById('questionInput');
const submitBtn = document.getElementById('submitBtn');
const voiceBtn = document.getElementById('voiceBtn');
const chatContainer = document.getElementById('chatContainer');
const languageSelect = document.getElementById('languageSelect');

let mediaRecorder;
let audioChunks = [];

function addMessage(content, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');
    messageDiv.classList.add(isUser ? 'user-message' : 'assistant-message');
    messageDiv.textContent = content;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addThinkingAnimation() {
    const thinkingDiv = document.createElement('div');
    thinkingDiv.classList.add('thinking');
    thinkingDiv.innerHTML = `
        <div class="thinking-dot"></div>
        <div class="thinking-dot"></div>
        <div class="thinking-dot"></div>
    `;
    chatContainer.appendChild(thinkingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return thinkingDiv;
}

// ... (keep the existing code)

async function sendMessage(message) {
    addMessage(message, true);
    const thinkingDiv = addThinkingAnimation();
    const mode = modeToggle.checked ? 'agent' : 'rag-chatbot';
    const url = `/api/${mode}`; // Update the URL to use the /api prefix
    
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: message }),
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        chatContainer.removeChild(thinkingDiv);
        addMessage(data.response);
    } catch (error) {
        console.error('Error:', error);
        chatContainer.removeChild(thinkingDiv);
        addMessage('An error occurred while fetching the response.');
    }
}

// ... (keep the rest of the code)
submitBtn.addEventListener('click', () => {
    const question = questionInput.value.trim();
    if (question) {
        sendMessage(question);
        questionInput.value = '';
    }
});

questionInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        submitBtn.click();
    }
});

voiceBtn.addEventListener('click', () => {
    const selectedLanguage = languageSelect.value;
    if (!selectedLanguage) {
        alert('Please select a language before using voice search.');
        return;
    }

    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        voiceBtn.textContent = '🎤';
        voiceBtn.classList.remove('recording');
    } else {
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.start();
                voiceBtn.textContent = '⏹️';
                voiceBtn.classList.add('recording');

                audioChunks = [];
                mediaRecorder.addEventListener("dataavailable", event => {
                    audioChunks.push(event.data);
                });

                mediaRecorder.addEventListener("stop", () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    sendAudioToBackend(audioBlob, selectedLanguage);
                });
            });
    }
});

async function sendAudioToBackend(audioBlob, languageCode) {
    const formData = new FormData();
    formData.append('file', audioBlob, 'audio.wav');
    formData.append('language_code', languageCode);
    formData.append('model', 'saarika:v1');

    try {
        const response = await fetch('https://api.sarvam.ai/speech-to-text', {
            method: 'POST',
            headers: {
                'API-Subscription-Key': 'c587d363-845f-4aa3-9f56-ac2735609bcc'  // Replace with your actual API key
            },
            body: formData
        });

        if (!response.ok) {
            throw new Error('Speech to text API request failed');
        }

        const data = await response.json();
        questionInput.value = data.transcript;
        sendMessage(data.transcript);
    } catch (error) {
        console.error('Error:', error);
        addMessage('An error occurred during speech recognition.');
    }
}