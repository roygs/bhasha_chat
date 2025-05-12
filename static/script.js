document.addEventListener('DOMContentLoaded', () => {
    const messageInput = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');
    const chatMessagesDiv = document.getElementById('chat-messages');
    const uploadBtn = document.getElementById('upload-btn');
    const fileUploadInput = document.getElementById('file-upload-input');
    const recordBtn = document.getElementById('record-btn');

    let isRecording = false;
    let isProcessing = false;
    let mediaRecorder;
    let audioChunks = [];
    let mediaStream = null; // To keep track of the stream for stopping

    // --- Helper function to add message to chat display ---
    function addMessageToChat(sender, content, type = 'text', data = null) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `message-${type}`);
        // If message is from 'You', align it to the right (handled by CSS if .message-text has margin-left: auto)
        if (sender === "You" && (type === 'text' || type === 'file' || type === 'audio')) {
             messageDiv.style.marginLeft = "auto"; // Example direct style, or use a class
             messageDiv.style.textAlign = "right";
        }

        if (sender === "Agent" && (type === 'text' || type === 'file' || type === 'audio')) {
            messageDiv.style.marginLeft = "auto"; // Example direct style, or use a class
            messageDiv.style.textAlign = "left";
       }


        let messageContent = `<strong>${sender}:</strong> ${content}`;

        if (type === 'audio' && data && data.path) {
            // Use the path received from the server, which should be served by /uploads
            messageContent += `<br><audio controls src="${data.path}"></audio>`;
        } 
        // else if (type === 'file' && data && data.filename) {
        //      messageContent += `<br><a href="/uploads/${data.filename}" target="_blank">Download ${data.filename}</a>`;
        // }

        messageDiv.innerHTML = messageContent;
        chatMessagesDiv.appendChild(messageDiv);
        chatMessagesDiv.scrollTop = chatMessagesDiv.scrollHeight; // Auto-scroll
    }

    // --- Send Text Message ---
    async function sendTextMessage() {
        const messageText = messageInput.value.trim();
        //if (messageText === '') return;
        if (messageText !== '') 
            addMessageToChat('You', messageText, 'text');

        const formData = new FormData();
        formData.append('message', messageText);

        try {
            addMessageToChat('System', 'Processing', 'system');
            isProcessing = true
            sendBtn.setAttribute("disabled","true");
            uploadBtn.setAttribute("disabled","true");
            recordBtn.setAttribute("disabled", "true")
            
            const response = await fetch('/send_message/', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json()
            isProcessing = false
            

            addMessageToChat('Agent',result.message,'text')
            addMessageToChat('Agent', `Answer:  ${result.filename}`, 'audio', 
                { path: result.path, filename: result.filename });
            sendBtn.removeAttribute("disabled");
            uploadBtn.removeAttribute("disabled");
            recordBtn.removeAttribute("disabled")

            //addMessageToChat('Agent', ``, 'audio', 
            //    { path: result.path, filename: result.filename });                
            //addMessageToChat('You', `Answer: ${result.filename}`, 'audio', { path: result.path, filename: result.filename });
            // if (!response.ok) {
            //     const result = await response.json().catch(() => ({ status: 'Server error' }));
            //     addMessageToChat('System', `Error: ${result.status || response.statusText}`, 'system');
            // }
 
            // Server doesn't need to send the message back if we're not using WebSockets
        } catch (error) {
            console.error('Error sending message:', error);
            addMessageToChat('System', 'Error: Could not connect to server.', 'system');
        }
        messageInput.value = '';
    }

    sendBtn.addEventListener('click', sendTextMessage);
    messageInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) { // Send on Enter, allow Shift+Enter for newline
            event.preventDefault(); // Prevent default Enter behavior (like newline in some inputs)
            sendTextMessage();
        }
    });

    // --- File Upload ---
    uploadBtn.addEventListener('click', () => {
        fileUploadInput.click();
    });

    fileUploadInput.addEventListener('change', async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        const optimisticMsgDiv = addMessageToChat('You', `Uploading: ${file.name}...`, 'file');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/upload_file/', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            console.log('Server response (file):', result);

            // Find the optimistic message and update it
            const existingMsg = Array.from(chatMessagesDiv.children).find(
                el => el.textContent.includes(`Uploading: ${file.name}`)
            );

            if (response.ok) {
                //if (existingMsg) {
                //    existingMsg.innerHTML = `<strong>You:</strong> Uploaded: ${result.filename} <br><a href="/uploads/${result.filename}" target="_blank">Download ${result.filename}</a>`;
                //} else { // Fallback
                    addMessageToChat('You', `Uploaded: ${result.filename}`, 'file', { filename: result.filename });
                //}
            } else {
                 if (existingMsg) {
                    existingMsg.innerHTML = `<strong>System:</strong> Error uploading ${file.name}: ${result.error || result.status}`;
                    existingMsg.classList.replace('message-file', 'message-system');
                 } else {
                    addMessageToChat('System', `Error uploading ${file.name}: ${result.error || result.status}`, 'system');
                 }
            }
        } catch (error) {
            console.error('Error uploading file:', error);
            const existingMsg = Array.from(chatMessagesDiv.children).find(
                el => el.textContent.includes(`Uploading: ${file.name}`)
            );
            if (existingMsg) {
                existingMsg.innerHTML = `<strong>System:</strong> Error: Could not upload ${file.name}. Check connection.`;
                existingMsg.classList.replace('message-file', 'message-system');
            } else {
                addMessageToChat('System', `Error: Could not upload ${file.name}.`, 'system');
            }
        }
        fileUploadInput.value = ''; // Reset file input
    });

    // --- Audio Recording ---
    recordBtn.addEventListener('click', async () => {
        if (!isRecording) {
            // Start recording
            try {
                // Stop any existing stream tracks before starting a new one
                if (mediaStream) {
                    mediaStream.getTracks().forEach(track => track.stop());
                }

                mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
                // Determine preferred MIME type
                const options = { mimeType: 'audio/wav' };
                if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                    console.log(options.mimeType + ' is not Supported');
                    options.mimeType = 'audio/ogg; codecs=opus';
                    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                        console.log(options.mimeType + ' is not Supported');
                        options.mimeType = ''; // Let the browser choose
                    }
                }

                mediaRecorder = new MediaRecorder(mediaStream, options.mimeType ? options : undefined);
                audioChunks = [];

                mediaRecorder.addEventListener('dataavailable', event => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                });

                mediaRecorder.addEventListener('stop', async () => {
                    // Tracks are stopped after blob creation and upload attempt
                    const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/wav' });
                    audioChunks = [];

                    //addMessageToChat('System', 'Sending audio...', 'system');

                    const formData = new FormData();
                    const fileExtension = (mediaRecorder.mimeType || 'audio/wav').split('/')[1].split(';')[0];
                    formData.append('audio', audioBlob, `recording.${fileExtension}`);

                    try {
                        const response = await fetch('/upload_audio/', {
                            method: 'POST',
                            body: formData
                        });
                        const result = await response.json();
                        console.log('Server response (audio):', result);
                        if (response.ok) {
                            //addMessageToChat('You', `Recorded audio: ${result.filename}`, 'audio', { path: result.path, filename: result.filename });
                            addMessageToChat('You', ``, 'audio', { path: result.path, filename: result.filename });

                            // //invoke agent with audiofile
                            // const formData = new FormData();
                            // formData.append('message', "");
                    
                            // try {
                            //     addMessageToChat('System', 'Processing', 'system');
                            //     const response = await fetch('/send_message/', {
                            //         method: 'POST',
                            //         body: formData
                            //     });
                                //const data = await response.json()
                                //addMessageToChat('Agent',JSON.stringify(data['message']),'text')
                                // if (!response.ok) {
                                //     const result = await response.json().catch(() => ({ status: 'Server error' }));
                                //     addMessageToChat('System', `Error: ${result.status || response.statusText}`, 'system');
                                // }
                     
                                // Server doesn't need to send the message back if we're not using WebSockets
                            // } catch (error) {
                            //     console.error('Error sending message:', error);
                            //     addMessageToChat('System', 'Error: Could not connect to server.', 'system');
                            // }                            
                        } else {
                            addMessageToChat('System', `Error uploading audio: ${result.error || result.status}`, 'system');
                        }
                    } catch (error) {
                        console.error('Error uploading audio:', error);
                        addMessageToChat('System', 'Error: Could not upload audio.', 'system');
                    } finally {
                        // Ensure microphone is released
                        if (mediaStream) {
                            mediaStream.getTracks().forEach(track => track.stop());
                            mediaStream = null;
                        }
                    }
                });

                mediaRecorder.start();
                isRecording = true;
                messageInput.value=""
                messageInput.disabled = true;
                sendBtn.setAttribute("disabled","true");
                uploadBtn.setAttribute("disabled","true");
                recordBtn.classList.add('recording');
                recordBtn.title = "Stop Recording";
                recordBtn.textContent = "◼️"; // Stop icon
                //addMessageToChat('System', 'Recording started...', 'system');

            } catch (err) {
                console.error('Error accessing microphone:', err);
                addMessageToChat('System', `Mic Error: ${err.name} - ${err.message}`, 'system');
                isRecording = false;
                recordBtn.classList.remove('recording');
                recordBtn.title = "Record Audio";
                recordBtn.textContent = "🎤";
                if (mediaStream) { // Clean up stream if acquisition failed mid-way
                    mediaStream.getTracks().forEach(track => track.stop());
                    mediaStream = null;
                }
            }
        } else {
            // Stop recording
            if (mediaRecorder && mediaRecorder.state !== "inactive") {
                mediaRecorder.stop(); // This will trigger the 'stop' event listener
                // Stream tracks are stopped inside the 'stop' event listener after processing
            }
            isRecording = false;
            messageInput.disabled = false;
            sendBtn.removeAttribute("disabled");
            uploadBtn.removeAttribute("disabled");

            recordBtn.classList.remove('recording');
            recordBtn.title = "Record Audio";
            recordBtn.textContent = "🎤"; // Mic icon
            // The "Recording stopped" message can be added here or after successful upload
            // addMessageToChat('System', 'Recording stopped.', 'system');
            messageInput.value=""
        }
    });
});